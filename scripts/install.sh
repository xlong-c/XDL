#!/bin/bash
# XDL 安装脚本
# 使用方法: bash scripts/install.sh [option]
# 选项: base, full, dev, cpu, cuda

set -e

echo "🚀 开始安装 XDL 深度学习框架..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取选项
OPTION=${1:-"full"}

# 检查 Python 版本
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "📦 Python 版本: $PYTHON_VERSION"

if (( $(echo "$PYTHON_VERSION < 3.8" | bc -l 2>/dev/null || echo "0") )); then
    echo -e "${RED}❌ 错误: 需要 Python 3.8 或更高版本${NC}"
    exit 1
fi

# 激活虚拟环境 (如果存在)
if [ -d "xdl_env" ]; then
    echo -e "${YELLOW}💡 发现虚拟环境，正在激活...${NC}"
    source xdl_env/bin/activate
fi

install_base() {
    echo -e "\n${GREEN}📦 安装核心依赖...${NC}"
    pip install --upgrade pip
    pip install torch torchvision numpy pyyaml tqdm typing-extensions
}

install_full() {
    echo -e "\n${GREEN}📦 安装完整依赖...${NC}"
    pip install --upgrade pip
    pip install -r requirements-full.txt
}

install_dev() {
    echo -e "\n${GREEN}📦 安装开发依赖...${NC}"
    pip install --upgrade pip
    pip install -r requirements-full.txt
    pip install pytest pytest-cov black isort flake8 mypy pre-commit
}

install_cpu() {
    echo -e "\n${GREEN}📦 安装 CPU 版本...${NC}"
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install numpy pyyaml tqdm typing-extensions opencv-python albumentations matplotlib
}

install_cuda() {
    echo -e "\n${GREEN}📦 安装 CUDA 版本...${NC}"
    echo "请选择 CUDA 版本:"
    echo "1) CUDA 11.8"
    echo "2) CUDA 12.1"
    echo "3) CUDA 12.4"
    read -p "输入选项 [1-3]: " cuda_option

    case $cuda_option in
        1)
            CUDA_VERSION="cu118"
            ;;
        2)
            CUDA_VERSION="cu121"
            ;;
        3)
            CUDA_VERSION="cu124"
            ;;
        *)
            echo -e "${RED}❌ 无效选项，使用默认 CUDA 11.8${NC}"
            CUDA_VERSION="cu118"
            ;;
    esac

    echo -e "${YELLOW}💡 安装 PyTorch with $CUDA_VERSION...${NC}"
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/$CUDA_VERSION
    pip install -r requirements-full.txt
}

# 根据选项执行安装
case $OPTION in
    base)
        install_base
        ;;
    full)
        install_full
        ;;
    dev)
        install_dev
        ;;
    cpu)
        install_cpu
        ;;
    cuda)
        install_cuda
        ;;
    *)
        echo -e "${RED}❌ 未知选项: $OPTION${NC}"
        echo "可用选项:"
        echo "  base  - 仅核心依赖"
        echo "  full  - 完整依赖 (推荐)"
        echo "  dev   - 开发环境"
        echo "  cpu   - CPU 版本"
        echo "  cuda  - CUDA 版本"
        exit 1
        ;;
esac

# 安装 XDL 包
echo -e "\n${GREEN}📦 安装 XDL 包...${NC}"
pip install -e .

# 验证安装
echo -e "\n${GREEN}✅ 验证安装...${NC}"
python -c "
import torch
import xdl
from xdl.trainer import Trainer
from xdl.callbacks import ModelCheckpoint
print(f'PyTorch: {torch.__version__}')
print(f'XDL: {xdl.__version__}')
print('✅ 安装成功！')
"

echo -e "\n${GREEN}🎉 安装完成！${NC}"
echo ""
echo "下一步:"
echo "  1. 查看文档: cat docs/INSTALL.md"
echo "  2. 运行测试: pytest tests/"
echo "  3. 查看示例: python examples/final_accelerate_test.py"
echo ""
echo "常用命令:"
echo "  python train.py --config config/vgg_cifar100.yaml"
echo "  tensorboard --logdir others/logs"