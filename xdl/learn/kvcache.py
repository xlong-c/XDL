import numpy as np

class SimpleLLMWithKVCache:
    def __init__(self, hidden_dim=64, head_dim=16):
        self.hidden_dim = hidden_dim  # 模型隐藏层维度
        self.head_dim = head_dim      # 注意力头维度
        self.num_heads = hidden_dim // head_dim  # 注意力头数量
        
        # 初始化随机参数(模拟模型权重)
        self.W_q = np.random.randn(hidden_dim, hidden_dim)  # Query 权重
        self.W_k = np.random.randn(hidden_dim, hidden_dim)  # Key 权重
        self.W_v = np.random.randn(hidden_dim, hidden_dim)  # Value 权重
        self.W_o = np.random.randn(hidden_dim, hidden_dim)  # 输出权重
        
        # KV Cache 存储(键和值的缓存, 格式：[batch_size, num_heads, seq_len, head_dim])
        self.cache_k = None
        self.cache_v = None

    def softmax(self, x, axis=-1):
        """手动实现softmax函数"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # 减去最大值防止数值溢出
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def attention(self, query, key, value):
        """修复维度对齐的注意力计算"""
        # query shape: [batch_size, num_heads, seq_len_q, head_dim]
        # key shape: [batch_size, num_heads, seq_len_k, head_dim]
        # 转换 key 维度为 [batch_size, num_heads, head_dim, seq_len_k] 以满足矩阵乘法
        key_transposed = key.transpose(0, 1, 3, 2)  # 修正转置维度
        
        # 注意力分数：(batch_size, num_heads, seq_len_q, seq_len_k)
        scores = np.matmul(query, key_transposed) / np.sqrt(self.head_dim)
        attn_weights = self.softmax(scores, axis=-1)  # 对最后一个维度(seq_len_k)归一化
        
        # 加权求和：(batch_size, num_heads, seq_len_q, head_dim)
        output = np.matmul(attn_weights, value)
        return output

    def forward(self, x, use_cache=True):
        """前向传播(含 KV Cache 逻辑)"""
        batch_size, seq_len, _ = x.shape
        
        # 1. 计算 Query、Key、Value(投影到注意力头维度)
        q = np.matmul(x, self.W_q).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = np.matmul(x, self.W_k).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = np.matmul(x, self.W_v).reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # 调整维度顺序：[batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # 2. 处理 KV Cache
        if use_cache and self.cache_k is not None and self.cache_v is not None:
            # 复用历史缓存：拼接新的k/v和缓存的k/v(在seq_len维度拼接)
            k = np.concatenate([self.cache_k, k], axis=2)
            v = np.concatenate([self.cache_v, v], axis=2)
        # print(k.shape)
        # print(v.shape)
        # 更新缓存
        if use_cache:
            self.cache_k = k
            self.cache_v = v
        
        # 3. 计算注意力输出
        attn_output = self.attention(q, k, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 4. 拼接注意力头并投影到输出维度
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_dim)
        output = np.matmul(attn_output, self.W_o)  # [batch_size, seq_len, hidden_dim]
        
        return output

    def reset_cache(self):
        """重置KV Cache"""
        self.cache_k = None
        self.cache_v = None


# 演示：模拟文本生成过程
if __name__ == "__main__":
    model = SimpleLLMWithKVCache(hidden_dim=64, head_dim=16)
    
    # 第1个token(无缓存)
    token1_emb = np.random.randn(1, 1, 64)  # [batch=1, seq_len=1, hidden_dim=64]
    output1 = model.forward(token1_emb, use_cache=True)
    assert model.cache_k is not None, "缓存应该已经被初始化"
    print(f"生成第1个token后, 缓存长度: {model.cache_k.shape[2]}")  # 输出：1
    
    # 第2个token(复用第1个的缓存)
    token2_emb = np.random.randn(1, 1, 64)
    output2 = model.forward(token2_emb, use_cache=True)
    print(f"生成第2个token后, 缓存长度: {model.cache_k.shape[2]}")  # 输出：2
    
    # 第3个token(复用前2个的缓存)
    token3_emb = np.random.randn(1, 1, 64)
    output3 = model.forward(token3_emb, use_cache=True)
    print(f"生成第3个token后, 缓存长度: {model.cache_k.shape[2]}")  # 输出：3
    
    # 重置缓存
    model.reset_cache()
    print(f"重置后缓存是否为空: {model.cache_k is None}")  # 输出：True