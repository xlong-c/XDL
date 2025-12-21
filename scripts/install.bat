@echo off
REM XDL 安装脚本 (Windows)
REM 使用方法: install.bat [option]
REM 选项: base, full, dev, cpu, cuda

setlocal enabledelayedexpansion

echo 🚀 开始安装 XDL 深度学习框架...

REM 获取选项
set "OPTION=%~1"
if "%OPTION%"=="" set "OPTION=full"

REM 检查 Python 版本
for /f "tokens=2 delims=." %%i in ('python -c "import sys; print(sys.version_info.minor)"') do set PY_MINOR=%%i
for /f "tokens=1 delims=." %%i in ('python -c "import sys; print(sys.version_info.major)"') do set PY_MAJOR=%%i

echo 📦 Python 版本: %PY_MAJOR%.%PY_MINOR%

if %PY_MAJOR% LSS 3 (
    echo ❌ 错误: 需要 Python 3.8 或更高版本
    exit /b 1
)
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 8 (
    echo ❌ 错误: 需要 Python 3.8 或更高版本
    exit /b 1
)

REM 激活虚拟环境 (如果存在)
if exist "xdl_env\Scripts\activate.bat" (
    echo 💡 发现虚拟环境，正在激活...
    call xdl_env\Scripts\activate.bat
)

goto :%OPTION%

:base
    echo.
    echo 📦 安装核心依赖...
    pip install --upgrade pip
    pip install torch torchvision numpy pyyaml tqdm typing-extensions
    goto :install_xdl

:full
    echo.
    echo 📦 安装完整依赖...
    pip install --upgrade pip
    pip install -r requirements-full.txt
    goto :install_xdl

:dev
    echo.
    echo 📦 安装开发依赖...
    pip install --upgrade pip
    pip install -r requirements-full.txt
    pip install pytest pytest-cov black isort flake8 mypy pre-commit
    goto :install_xdl

:cpu
    echo.
    echo 📦 安装 CPU 版本...
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install numpy pyyaml tqdm typing-extensions opencv-python albumentations matplotlib
    goto :install_xdl

:cuda
    echo.
    echo 📦 安装 CUDA 版本...
    echo 请选择 CUDA 版本:
    echo 1) CUDA 11.8
    echo 2) CUDA 12.1
    echo 3) CUDA 12.4
    set /p cuda_option=输入选项 [1-3]:

    if "%cuda_option%"=="1" (
        set "CUDA_VERSION=cu118"
    ) elif "%cuda_option%"=="2" (
        set "CUDA_VERSION=cu121"
    ) elif "%cuda_option%"=="3" (
        set "CUDA_VERSION=cu124"
    ) else (
        echo ❌ 无效选项，使用默认 CUDA 11.8
        set "CUDA_VERSION=cu118"
    )

    echo 💡 安装 PyTorch with !CUDA_VERSION!...
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/!CUDA_VERSION!
    pip install -r requirements-full.txt
    goto :install_xdl

:install_xdl
    echo.
    echo 📦 安装 XDL 包...
    pip install -e .

    echo.
    echo ✅ 验证安装...
    python -c "import torch; import xdl; from xdl.trainer import Trainer; from xdl.callbacks import ModelCheckpoint; print(f'PyTorch: {torch.__version__}'); print(f'XDL: {xdl.__version__}'); print('✅ 安装成功！')"

    echo.
    echo 🎉 安装完成！
    echo.
    echo 下一步:
    echo   1. 查看文档: type docs\INSTALL.md
    echo   2. 运行测试: pytest tests\
    echo   3. 查看示例: python examples\final_accelerate_test.py
    echo.
    echo 常用命令:
    echo   python train.py --config config\vgg_cifar100.yaml
    echo   tensorboard --logdir others\logs
    goto :end

:unknown
    echo ❌ 未知选项: %OPTION%
    echo 可用选项:
    echo   base  - 仅核心依赖
    echo   full  - 完整依赖 (推荐)
    echo   dev   - 开发环境
    echo   cpu   - CPU 版本
    echo   cuda  - CUDA 版本
    exit /b 1

:end
    endlocal