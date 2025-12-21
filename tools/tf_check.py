import sys
import pkg_resources

# 1. 检查当前环境是否安装了 tensorflow
try:
    import tensorflow as tf
    print(f"✅ 环境中安装了 TensorFlow，版本：{tf.__version__}")
except ImportError:
    print("❌ 环境中未安装 TensorFlow")

# 2. 检查已安装的依赖包（找可能依赖 TF 的库）
print("\n=== 已安装的可能关联 TensorFlow 的包 ===")
for pkg in pkg_resources.working_set:
    pkg_name = pkg.key.lower()
    # 筛选可能依赖 TF 的包关键词
    if any(key in pkg_name for key in ["tensorflow", "tf", "keras", "paddle", "torch"]):
        print(f"{pkg.project_name} == {pkg.version}")

# 3. 检查当前进程已加载的模块（找谁加载了 TF）
print("\n=== 当前进程已加载的模块（含 TF 相关）===")
for module in sys.modules:
    if "tensorflow" in module or "tf" in module:
        print(module)