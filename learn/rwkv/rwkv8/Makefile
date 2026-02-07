# RWKV-8 CUDA 算子实现 - 构建和测试 Makefile

.PHONY: all build test clean docs benchmark format lint help

# 默认目标
all: build

# 帮助信息
help:
	@echo "RWKV-8 CUDA 算子实现 - 可用命令:"
	@echo ""
	@echo "  make build       - 构建 CUDA 扩展"
	@echo "  make test        - 运行所有测试"
	@echo "  make benchmark   - 运行性能基准测试"
	@echo "  make docs        - 生成文档"
	@echo "  make clean       - 清理构建文件"
	@echo "  make format      - 格式化代码"
	@echo "  make lint        - 代码检查"
	@echo ""

# 构建 CUDA 扩展
build:
	@echo "Building RWKV CUDA extensions..."
	cd cuda && python setup.py build_ext --inplace
	@echo "Build complete!"

# 运行测试
test:
	@echo "Running tests..."
	cd tests && python -m pytest test_wkv.py -v
	@echo "Tests complete!"

# 运行基准测试
benchmark:
	@echo "Running benchmarks..."
	cd benchmarks && python compare_speed.py
	@echo "Benchmarks complete!"

# 内存分析
memory:
	@echo "Running memory analysis..."
	cd benchmarks && python memory_profile.py
	@echo "Memory analysis complete!"

# 生成可视化
visualize:
	@echo "Generating visualizations..."
	cd experiments && python visualize_wkv.py
	@echo "Visualizations complete!"

# 生成文档
docs:
	@echo "Generating documentation..."
	@echo "Documentation is in the docs/ directory"
	@echo "- 01_wkv_math.md: WKV 数学推导"
	@echo "- 02_rwkv_architecture.md: RWKV 架构详解"
	@echo "- 03_rwkv_vs_others.md: 与其他架构对比"

# 代码格式化
format:
	@echo "Formatting code..."
	find . -name "*.py" -type f -exec black {} \;
	@echo "Formatting complete!"

# 代码检查
lint:
	@echo "Linting code..."
	find . -name "*.py" -type f -exec pylint {} \;
	@echo "Linting complete!"

# 清理构建文件
clean:
	@echo "Cleaning build files..."
	rm -rf cuda/build
	rm -rf cuda/*.egg-info
	rm -rf cuda/__pycache__
	rm -rf cuda/*.so
	rm -rf reference/__pycache__
	rm -rf tests/__pycache__
	rm -rf benchmarks/__pycache__
	rm -rf experiments/__pycache__
	rm -rf .pytest_cache
	@echo "Clean complete!"

# 安装依赖
install-deps:
	@echo "Installing dependencies..."
	pip install torch numpy matplotlib seaborn pandas pytest black pylint jupyter
	@echo "Dependencies installed!"

# 完整测试流程 (CI/CD)
ci: clean build test benchmark
	@echo "CI/CD pipeline complete!"

# 发布检查
release-check:
	@echo "Running release checks..."
	make clean
	make build
	make test
	@echo "All checks passed! Ready for release."
