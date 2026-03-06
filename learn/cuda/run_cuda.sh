#!/bin/sh

set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BUILD_ROOT="${SCRIPT_DIR}/build"

find_first_existing_file_dir() {
    target_file=$1
    shift

    for candidate in "$@"; do
        [ -n "$candidate" ] || continue
        if [ -f "$candidate/$target_file" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

find_cuda_related_dir() {
    target_file=$1
    NVCC_BIN=$(command -v nvcc 2>/dev/null || true)
    NVCC_ROOT=
    if [ -n "$NVCC_BIN" ]; then
        NVCC_ROOT=$(cd "$(dirname "$NVCC_BIN")/.." && pwd)
    fi

    find_first_existing_file_dir \
        "$target_file" \
        "${CUDA_HOME:-}/include" \
        "${CONDA_PREFIX:-}/include" \
        "$NVCC_ROOT/include" \
        "$NVCC_ROOT/targets/x86_64-linux/include" \
        /usr/local/cuda/include \
        /usr/local/cuda-13.0/include \
        /usr/local/cuda-12.9/include \
        /usr/local/cuda-12.8/include \
        /usr/local/cuda-12.6/include \
        /usr/local/cuda-12.4/include \
        "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include" \
        "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include" \
        "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include" \
        "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/include" \
        "/mnt/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/include"
}

find_cuda_related_lib_dir() {
    target_pattern=$1
    NVCC_BIN=$(command -v nvcc 2>/dev/null || true)
    NVCC_ROOT=
    if [ -n "$NVCC_BIN" ]; then
        NVCC_ROOT=$(cd "$(dirname "$NVCC_BIN")/.." && pwd)
    fi

    for candidate in \
        "${CUDA_HOME:-}" \
        "${CONDA_PREFIX:-}" \
        "$NVCC_ROOT" \
        /usr/local/cuda \
        /usr/local/cuda-13.0 \
        /usr/local/cuda-12.9 \
        /usr/local/cuda-12.8 \
        /usr/local/cuda-12.6 \
        /usr/local/cuda-12.4
    do
        [ -n "$candidate" ] || continue
        match=$(find "$candidate" -maxdepth 8 -name "$target_pattern" 2>/dev/null | head -n 1 || true)
        if [ -n "$match" ]; then
            dirname "$match"
            return 0
        fi
    done

    return 1
}

usage() {
    cat <<'USAGE'
用法:
  sh learn/cuda/run_cuda.sh <source.cu> [-- <program args...>]

示例:
  sh learn/cuda/run_cuda.sh learn/cuda/kernels/gemm/sgemm.cu
  sh learn/cuda/run_cuda.sh learn/cuda/kernels/softmax/softmax_1.cu -- --size 4096

说明:
  - 仅支持包含 main 函数的 .cu 可执行 demo
  - 编译产物会保存在 learn/cuda/build/ 下，并保留原有子目录结构
  - 可通过环境变量 NVCC_FLAGS 追加额外编译参数
USAGE
}

if [ "$#" -lt 1 ]; then
    usage
    exit 1
fi

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 0
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "错误: 未找到 nvcc, 请先安装 CUDA toolkit 或激活正确环境" >&2
    exit 1
fi

SOURCE_INPUT=$1
shift

if [ "$#" -gt 0 ] && [ "$1" = "--" ]; then
    shift
fi

SOURCE_PATH=$(realpath "$SOURCE_INPUT")

if [ ! -f "$SOURCE_PATH" ]; then
    echo "错误: 源文件不存在: $SOURCE_INPUT" >&2
    exit 1
fi

case "$SOURCE_PATH" in
    *.cu) ;;
    *)
        echo "错误: 仅支持 .cu 文件: $SOURCE_PATH" >&2
        exit 1
        ;;
esac

case "$SOURCE_PATH" in
    "$SCRIPT_DIR"/*) ;;
    *)
        echo "错误: 仅支持编译 $SCRIPT_DIR 目录下的 .cu 文件" >&2
        exit 1
        ;;
esac

if ! grep -Eq 'int[[:space:]]+main[[:space:]]*\(' "$SOURCE_PATH"; then
    echo "错误: $SOURCE_PATH 中未检测到 main 函数, 当前脚本仅支持可直接运行的 demo" >&2
    exit 1
fi

SOURCE_REL=${SOURCE_PATH#"$SCRIPT_DIR"/}
SOURCE_DIR_REL=$(dirname "$SOURCE_REL")
TARGET_DIR="$BUILD_ROOT/$SOURCE_DIR_REL"
TARGET_NAME=$(basename "$SOURCE_PATH" .cu)
TARGET_PATH="$TARGET_DIR/$TARGET_NAME"
SOURCE_DIR=$(dirname "$SOURCE_PATH")
CUDA_INCLUDE_DIR=$(find_cuda_related_dir cuda_runtime.h || true)
CUBLAS_INCLUDE_DIR=$(find_cuda_related_dir cublas_v2.h || true)
CUBLAS_LIB_DIR=$(find_cuda_related_lib_dir 'libcublas.so*' || true)
CUBLAS_LT_LIB_DIR=$(find_cuda_related_lib_dir 'libcublasLt.so*' || true)
CUBLAS_LIB_FILE=""
CUBLAS_LT_LIB_FILE=""
EXTRA_LINK_DIR_FLAGS=""
EXTRA_LINK_FLAGS=""
RUNTIME_LIB_DIRS=""
NEED_CUBLAS=0
NEED_CUBLAS_LT=0

if [ -n "$CUBLAS_LIB_DIR" ]; then
    CUBLAS_LIB_FILE=$(find "$CUBLAS_LIB_DIR" -maxdepth 1 -name 'libcublas.so*' 2>/dev/null | head -n 1 || true)
fi
if [ -n "$CUBLAS_LT_LIB_DIR" ]; then
    CUBLAS_LT_LIB_FILE=$(find "$CUBLAS_LT_LIB_DIR" -maxdepth 1 -name 'libcublasLt.so*' 2>/dev/null | head -n 1 || true)
fi

if grep -Eq 'cublas_v2\.h|cublas[A-Za-z0-9_]+' "$SOURCE_PATH"; then
    NEED_CUBLAS=1
fi
if grep -Eq 'cublasLt\.h|cublasLt[A-Za-z0-9_]+' "$SOURCE_PATH"; then
    NEED_CUBLAS=1
    NEED_CUBLAS_LT=1
fi

if [ "$NEED_CUBLAS" -eq 1 ]; then
    if [ -n "$CUBLAS_LIB_DIR" ]; then
        EXTRA_LINK_DIR_FLAGS="$EXTRA_LINK_DIR_FLAGS -L$CUBLAS_LIB_DIR"
        RUNTIME_LIB_DIRS="$RUNTIME_LIB_DIRS:$CUBLAS_LIB_DIR"
    fi
    if [ -n "$CUBLAS_LIB_FILE" ]; then
        EXTRA_LINK_FLAGS="$EXTRA_LINK_FLAGS -l:$(basename "$CUBLAS_LIB_FILE")"
    else
        EXTRA_LINK_FLAGS="$EXTRA_LINK_FLAGS -lcublas"
    fi
fi

if [ "$NEED_CUBLAS_LT" -eq 1 ]; then
    if [ -n "$CUBLAS_LT_LIB_DIR" ] && [ "$CUBLAS_LT_LIB_DIR" != "$CUBLAS_LIB_DIR" ]; then
        EXTRA_LINK_DIR_FLAGS="$EXTRA_LINK_DIR_FLAGS -L$CUBLAS_LT_LIB_DIR"
        RUNTIME_LIB_DIRS="$RUNTIME_LIB_DIRS:$CUBLAS_LT_LIB_DIR"
    elif [ -n "$CUBLAS_LT_LIB_DIR" ] && [ -z "$CUBLAS_LIB_DIR" ]; then
        EXTRA_LINK_DIR_FLAGS="$EXTRA_LINK_DIR_FLAGS -L$CUBLAS_LT_LIB_DIR"
        RUNTIME_LIB_DIRS="$RUNTIME_LIB_DIRS:$CUBLAS_LT_LIB_DIR"
    fi
    if [ -n "$CUBLAS_LT_LIB_FILE" ]; then
        EXTRA_LINK_FLAGS="$EXTRA_LINK_FLAGS -l:$(basename "$CUBLAS_LT_LIB_FILE")"
    else
        EXTRA_LINK_FLAGS="$EXTRA_LINK_FLAGS -lcublasLt"
    fi
fi

EXTRA_LINK_DIR_FLAGS=$(printf '%s' "$EXTRA_LINK_DIR_FLAGS" | sed 's/^ *//')
EXTRA_LINK_FLAGS=$(printf '%s' "$EXTRA_LINK_FLAGS" | sed 's/^ *//')
RUNTIME_LIB_DIRS=$(printf '%s' "$RUNTIME_LIB_DIRS" | sed 's/^://')

mkdir -p "$TARGET_DIR"

echo "[build] $SOURCE_REL -> $TARGET_PATH"
[ -n "$CUDA_INCLUDE_DIR" ] && echo "[cuda] runtime include: $CUDA_INCLUDE_DIR"
[ -n "$CUBLAS_INCLUDE_DIR" ] && echo "[cuda] cublas include: $CUBLAS_INCLUDE_DIR"
[ -n "$CUBLAS_LIB_DIR" ] && echo "[cuda] cublas lib: $CUBLAS_LIB_DIR"
[ -n "$CUBLAS_LIB_FILE" ] && echo "[cuda] cublas file: $CUBLAS_LIB_FILE"
[ -n "$CUBLAS_LT_LIB_DIR" ] && echo "[cuda] cublasLt lib: $CUBLAS_LT_LIB_DIR"
[ -n "$CUBLAS_LT_LIB_FILE" ] && echo "[cuda] cublasLt file: $CUBLAS_LT_LIB_FILE"
[ -n "$EXTRA_LINK_DIR_FLAGS" ] && echo "[link] extra dir flags: $EXTRA_LINK_DIR_FLAGS"
[ -n "$EXTRA_LINK_FLAGS" ] && echo "[link] extra flags: $EXTRA_LINK_FLAGS"

# NVCC_FLAGS 允许用户以环境变量形式追加额外参数。
# shellcheck disable=SC2086
if [ -n "$CUDA_INCLUDE_DIR" ] && [ -n "$CUBLAS_INCLUDE_DIR" ] && [ "$CUDA_INCLUDE_DIR" != "$CUBLAS_INCLUDE_DIR" ]; then
    nvcc "$SOURCE_PATH" \
        -O3 \
        -std=c++17 \
        -I"$PROJECT_ROOT" \
        -I"$SOURCE_DIR" \
        -I"$CUDA_INCLUDE_DIR" \
        -I"$CUBLAS_INCLUDE_DIR" \
        ${NVCC_FLAGS:-} \
        ${EXTRA_LINK_DIR_FLAGS:-} \
        ${EXTRA_LINK_FLAGS:-} \
        -o "$TARGET_PATH"
elif [ -n "$CUDA_INCLUDE_DIR" ]; then
    nvcc "$SOURCE_PATH" \
        -O3 \
        -std=c++17 \
        -I"$PROJECT_ROOT" \
        -I"$SOURCE_DIR" \
        -I"$CUDA_INCLUDE_DIR" \
        ${NVCC_FLAGS:-} \
        ${EXTRA_LINK_DIR_FLAGS:-} \
        ${EXTRA_LINK_FLAGS:-} \
        -o "$TARGET_PATH"
elif [ -n "$CUBLAS_INCLUDE_DIR" ]; then
    nvcc "$SOURCE_PATH" \
        -O3 \
        -std=c++17 \
        -I"$PROJECT_ROOT" \
        -I"$SOURCE_DIR" \
        -I"$CUBLAS_INCLUDE_DIR" \
        ${NVCC_FLAGS:-} \
        ${EXTRA_LINK_DIR_FLAGS:-} \
        ${EXTRA_LINK_FLAGS:-} \
        -o "$TARGET_PATH"
else
    echo "[warn] 未找到额外 CUDA include 路径, 将仅使用 nvcc 默认搜索路径" >&2
    nvcc "$SOURCE_PATH" \
        -O3 \
        -std=c++17 \
        -I"$PROJECT_ROOT" \
        -I"$SOURCE_DIR" \
        ${NVCC_FLAGS:-} \
        ${EXTRA_LINK_DIR_FLAGS:-} \
        ${EXTRA_LINK_FLAGS:-} \
        -o "$TARGET_PATH"
fi

echo "[run] $TARGET_PATH${*:+ }$*"
if [ -n "$RUNTIME_LIB_DIRS" ]; then
    LD_LIBRARY_PATH="$RUNTIME_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$TARGET_PATH" "$@"
else
    "$TARGET_PATH" "$@"
fi
