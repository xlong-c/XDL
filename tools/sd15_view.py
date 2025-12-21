import os
from peft import get_peft_model, LoraConfig
# 设置环境变量, 使用 hfmirror 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

# 加载预训练的UNet模型, 使用本地缓存
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="unet",
    revision=None,
)

print("原始UNet模型结构:")
print(unet)
print(f"\n模型参数总数: {sum(p.numel() for p in unet.parameters())}")

# 配置LoRA (Low-Rank Adaptation) 参数
lora_config = LoraConfig(
    r=16,  # LoRA的rank, 控制低秩矩阵的大小
    lora_alpha=32,  # LoRA的缩放因子
    target_modules=['attn1.to_k'],  # 仅attn1的目标模块列表
    lora_dropout=0.1,  # LoRA的dropout率
    bias="none",  # 不训练bias参数
    task_type="FEATURE_EXTRACTION",  # 使用特征提取任务类型
)

# 应用PEFT到UNet模型
print("\n应用PEFT配置到UNet模型...")
peft_model = get_peft_model(unet, lora_config)

peft_model.print_trainable_parameters()

# 收集UNet模型中的LoRA参数
unet_lora_param = []
for name, param in peft_model.named_parameters():
    if param.requires_grad:
        unet_lora_param.append((name, param))
        print(f"可训练LoRA参数: {name}, 形状: {param.shape}")

print(f"\n总共收集到 {len(unet_lora_param)} 个LoRA参数")
print(f"LoRA参数总数: {sum(p.numel() for (_, p) in unet_lora_param)}")