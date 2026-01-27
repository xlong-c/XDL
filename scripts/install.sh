#!/bin/bash

# XDL 安装脚本
# 支持基础安装、完整安装以及特定硬件版本的安装

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    error "Python 3 未安装，请先安装 Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
log "检测到 Python 版本: $PYTHON_VERSION"

# 安装选项
MODE=${1:-"base"}

# 升级 pip
log "正在升级 pip..."
python3 -m pip install --upgrade pip

case $MODE in
    "base")
        log "正在进行基础安装..."
        python3 -m pip install -e .
        ;;
    "full"|"all")
        log "正在进行完整安装 (包含所有可选依赖)..."
        python3 -m pip install -e ".[all]"
        ;;
    "cuda")
        log "正在安装针对 CUDA 的依赖..."
        # 默认安装最新的稳定版 torch (CUDA 支持)
        python3 -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
        python3 -m pip install -e ".[all]"
        ;;
    "cpu")
        log "正在安装针对 CPU 的依赖..."
        python3 -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
        python3 -m pip install -e ".[all]"
        ;;
    "dev")
        log "正在安装开发环境依赖..."
        python3 -m pip install -e ".[all,dev]"
        ;;
    *)
        warn "未知的安装模式: $MODE. 默认执行基础安装."
        python3 -m pip install -e .
        ;;
esac

# 检查必要的第三方库 (如 loguru)
if ! python3 -c "import loguru" &> /dev/null; then
    log "正在安装 loguru (xdl 核心依赖)..."
    python3 -m pip install loguru
fi

log "XDL 安装完成!"