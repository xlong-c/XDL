#include "cublas_demo_utils.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

enum class RunnerKind {
  kDense,
  kFp8,
  kFp4,
};

enum class DataKind {
  kFp32,
  kFp16,
  kBf16,
  kFp8E4M3,
  kFp8E5M2,
  kFp4E2M1,
};

enum class ComputeKind {
  kFp16,
  kFp32,
  kTf32,
};

enum class ScaleModeKind {
  kAuto,
  kScalar32F,
  kVec16Ue4M3,
};

struct BenchmarkResult {
  float max_abs_err = 0.0f;
  float avg_time_ms = 0.0f;
  double tflops = 0.0;
  bool passed = false;
};

struct RunSpec {
  DataKind a_kind = DataKind::kFp32;
  DataKind b_kind = DataKind::kFp32;
  DataKind c_kind = DataKind::kFp32;
  DataKind d_kind = DataKind::kFp32;
  ComputeKind compute_kind = ComputeKind::kFp32;
  ScaleModeKind a_scale_kind = ScaleModeKind::kAuto;
  ScaleModeKind b_scale_kind = ScaleModeKind::kAuto;
  int m = 256;
  int n = 256;
  int k = 256;
  float tolerance = 1e-4f;
  bool tolerance_overridden = false;
  std::string label;
};

constexpr int kDefaultDim = 256;
constexpr int kWarmupIters = 5;
constexpr int kRepeatIters = 30;
constexpr size_t kWorkspaceBytes = 32ull * 1024ull * 1024ull;

const char *data_kind_name(const DataKind kind) {
  switch (kind) {
  case DataKind::kFp32:
    return "fp32";
  case DataKind::kFp16:
    return "fp16";
  case DataKind::kBf16:
    return "bf16";
  case DataKind::kFp8E4M3:
    return "fp8e4m3";
  case DataKind::kFp8E5M2:
    return "fp8e5m2";
  case DataKind::kFp4E2M1:
    return "fp4";
  }

  return "unknown";
}

const char *compute_kind_name(const ComputeKind kind) {
  switch (kind) {
  case ComputeKind::kFp16:
    return "fp16";
  case ComputeKind::kFp32:
    return "fp32";
  case ComputeKind::kTf32:
    return "tf32";
  }

  return "unknown";
}

const char *scale_mode_name(const ScaleModeKind kind) {
  switch (kind) {
  case ScaleModeKind::kAuto:
    return "auto";
  case ScaleModeKind::kScalar32F:
    return "scalar32f";
  case ScaleModeKind::kVec16Ue4M3:
    return "vec16ue4m3";
  }

  return "unknown";
}

bool is_dense_data_kind(const DataKind kind) {
  return kind == DataKind::kFp32 || kind == DataKind::kFp16 ||
         kind == DataKind::kBf16;
}

bool is_fp8_data_kind(const DataKind kind) {
  return kind == DataKind::kFp8E4M3 || kind == DataKind::kFp8E5M2;
}

bool is_fp4_data_kind(const DataKind kind) {
  return kind == DataKind::kFp4E2M1;
}

bool is_supported_output_kind(const DataKind kind) {
  return kind == DataKind::kFp32 || kind == DataKind::kFp16 ||
         kind == DataKind::kBf16;
}

cudaDataType to_cuda_data_type(const DataKind kind) {
  switch (kind) {
  case DataKind::kFp32:
    return CUDA_R_32F;
  case DataKind::kFp16:
    return CUDA_R_16F;
  case DataKind::kBf16:
    return CUDA_R_16BF;
  case DataKind::kFp8E4M3:
    return CUDA_R_8F_E4M3;
  case DataKind::kFp8E5M2:
    return CUDA_R_8F_E5M2;
  case DataKind::kFp4E2M1:
    return CUDA_R_4F_E2M1;
  }

  return CUDA_R_32F;
}

cublasComputeType_t to_cublas_compute_type(const ComputeKind kind,
                                           const DataKind a_kind,
                                           const DataKind b_kind) {
  switch (kind) {
  case ComputeKind::kFp16:
    return CUBLAS_COMPUTE_16F;
  case ComputeKind::kFp32:
    if (a_kind == DataKind::kFp16 && b_kind == DataKind::kFp16) {
      return CUBLAS_COMPUTE_32F_FAST_16F;
    }
    if (a_kind == DataKind::kBf16 && b_kind == DataKind::kBf16) {
      return CUBLAS_COMPUTE_32F_FAST_16BF;
    }
    return CUBLAS_COMPUTE_32F;
  case ComputeKind::kTf32:
    return CUBLAS_COMPUTE_32F_FAST_TF32;
  }

  return CUBLAS_COMPUTE_32F;
}

cudaDataType to_scale_type(const ComputeKind kind) {
  return kind == ComputeKind::kFp16 ? CUDA_R_16F : CUDA_R_32F;
}

cublasLtMatmulMatrixScale_t to_cublaslt_scale_mode(const ScaleModeKind kind) {
  switch (kind) {
  case ScaleModeKind::kAuto:
  case ScaleModeKind::kScalar32F:
    return CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
  case ScaleModeKind::kVec16Ue4M3:
    return CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  }

  return CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
}

bool parse_data_kind(const std::string &text, DataKind &kind) {
  if (text == "fp32") {
    kind = DataKind::kFp32;
    return true;
  }
  if (text == "fp16" || text == "half") {
    kind = DataKind::kFp16;
    return true;
  }
  if (text == "bf16") {
    kind = DataKind::kBf16;
    return true;
  }
  if (text == "fp8" || text == "fp8e4m3" || text == "fp8_e4m3") {
    kind = DataKind::kFp8E4M3;
    return true;
  }
  if (text == "fp8e5m2" || text == "fp8_e5m2") {
    kind = DataKind::kFp8E5M2;
    return true;
  }
  if (text == "fp4" || text == "fp4e2m1" || text == "fp4_e2m1") {
    kind = DataKind::kFp4E2M1;
    return true;
  }
  return false;
}

bool parse_compute_kind(const std::string &text, ComputeKind &kind) {
  if (text == "fp16") {
    kind = ComputeKind::kFp16;
    return true;
  }
  if (text == "fp32") {
    kind = ComputeKind::kFp32;
    return true;
  }
  if (text == "tf32") {
    kind = ComputeKind::kTf32;
    return true;
  }
  return false;
}

bool parse_scale_mode_kind(const std::string &text, ScaleModeKind &kind) {
  if (text == "auto") {
    kind = ScaleModeKind::kAuto;
    return true;
  }
  if (text == "scalar" || text == "scalar32f") {
    kind = ScaleModeKind::kScalar32F;
    return true;
  }
  if (text == "vec16ue4m3") {
    kind = ScaleModeKind::kVec16Ue4M3;
    return true;
  }
  return false;
}

float parse_positive_float(const char *text, const char *name) {
  char *end = nullptr;
  const float value = std::strtof(text, &end);
  if (end == text || *end != '\0' || value <= 0.0f) {
    std::fprintf(stderr, "非法 %s: %s\n", name, text);
    std::exit(EXIT_FAILURE);
  }
  return value;
}

RunnerKind infer_runner_kind(const RunSpec &spec) {
  if (is_fp4_data_kind(spec.a_kind) || is_fp4_data_kind(spec.b_kind)) {
    return RunnerKind::kFp4;
  }
  if (is_fp8_data_kind(spec.a_kind) || is_fp8_data_kind(spec.b_kind)) {
    return RunnerKind::kFp8;
  }
  return RunnerKind::kDense;
}

float default_tolerance(const RunSpec &spec) {
  const RunnerKind runner_kind = infer_runner_kind(spec);
  if (runner_kind == RunnerKind::kFp4) {
    return 0.5f;
  }
  if (runner_kind == RunnerKind::kFp8) {
    return 0.08f;
  }
  if (spec.compute_kind == ComputeKind::kTf32) {
    return 1e-2f;
  }
  if (spec.compute_kind == ComputeKind::kFp16) {
    return 1e-2f;
  }
  if (spec.a_kind == DataKind::kBf16 || spec.b_kind == DataKind::kBf16) {
    return 2e-2f;
  }
  if (spec.a_kind == DataKind::kFp16 || spec.b_kind == DataKind::kFp16 ||
      spec.d_kind == DataKind::kFp16) {
    return 5e-3f;
  }
  return 1e-4f;
}

float source_scale(const RunSpec &spec) {
  return infer_runner_kind(spec) == RunnerKind::kFp4 ? 0.5f : 0.25f;
}

std::string spec_label(const RunSpec &spec) {
  return std::string("A=") + data_kind_name(spec.a_kind) + ", B=" +
         data_kind_name(spec.b_kind) + ", C=" + data_kind_name(spec.c_kind) +
         ", D=" + data_kind_name(spec.d_kind) + ", compute=" +
         compute_kind_name(spec.compute_kind);
}

bool validate_spec(RunSpec &spec, std::string &detail, std::string &error) {
  const RunnerKind runner_kind = infer_runner_kind(spec);
  detail.clear();
  error.clear();

  if (!spec.tolerance_overridden) {
    spec.tolerance = default_tolerance(spec);
  }
  if (spec.label.empty()) {
    spec.label = spec_label(spec);
  }

  if (!is_supported_output_kind(spec.c_kind) ||
      !is_supported_output_kind(spec.d_kind)) {
    error = "当前 demo 仅支持 C/D 使用 fp32、fp16 或 bf16";
    return false;
  }

  if (runner_kind == RunnerKind::kDense) {
    if (!is_dense_data_kind(spec.a_kind) || !is_dense_data_kind(spec.b_kind) ||
        !is_dense_data_kind(spec.c_kind) || !is_dense_data_kind(spec.d_kind)) {
      error = "dense 路径仅支持 fp32/fp16/bf16";
      return false;
    }
    if (spec.a_scale_kind != ScaleModeKind::kAuto ||
        spec.b_scale_kind != ScaleModeKind::kAuto) {
      error = "dense 路径不需要 A/B scale mode，请移除 --a-scale/--b-scale";
      return false;
    }
    if (spec.compute_kind == ComputeKind::kFp16) {
      if (spec.a_kind != DataKind::kFp16 || spec.b_kind != DataKind::kFp16) {
        error = "compute=fp16 时，A/B 必须都是 fp16";
        return false;
      }
      if (spec.c_kind != DataKind::kFp16 || spec.d_kind != DataKind::kFp16) {
        error = "compute=fp16 时，当前 demo 仅开放 C/D 为 fp16";
        return false;
      }
      detail = "纯 half accumulate 路径";
      return true;
    }
    if (spec.compute_kind == ComputeKind::kTf32) {
      if (spec.a_kind != DataKind::kFp32 || spec.b_kind != DataKind::kFp32 ||
          spec.c_kind != DataKind::kFp32 || spec.d_kind != DataKind::kFp32) {
        error = "compute=tf32 时，当前 demo 仅开放 A/B/C/D 全为 fp32";
        return false;
      }
      detail = "输入存储为 fp32，Tensor Core 计算路径使用 tf32";
      return true;
    }
    detail = "dense/mixed precision 路径";
    return true;
  }

  if (runner_kind == RunnerKind::kFp8) {
    if (!is_fp8_data_kind(spec.a_kind) || !is_fp8_data_kind(spec.b_kind)) {
      error = "fp8 路径要求 A/B 都是 fp8 类型（fp8e4m3 或 fp8e5m2）";
      return false;
    }
    if (spec.compute_kind != ComputeKind::kFp32) {
      error = "fp8 路径当前仅开放 compute=fp32";
      return false;
    }
    if (spec.a_scale_kind == ScaleModeKind::kAuto) {
      spec.a_scale_kind = ScaleModeKind::kScalar32F;
    }
    if (spec.b_scale_kind == ScaleModeKind::kAuto) {
      spec.b_scale_kind = ScaleModeKind::kScalar32F;
    }
    if (spec.a_scale_kind != ScaleModeKind::kScalar32F ||
        spec.b_scale_kind != ScaleModeKind::kScalar32F) {
      error = "fp8 路径当前仅开放 scalar32f scale mode";
      return false;
    }
    detail = "fp8 路径使用 device scale pointer，scale_mode=scalar32f";
    return true;
  }

  if (!is_fp4_data_kind(spec.a_kind) || !is_fp4_data_kind(spec.b_kind)) {
    error = "fp4 路径要求 A/B 都是 fp4";
    return false;
  }
  if (spec.compute_kind != ComputeKind::kFp32) {
    error = "fp4 路径当前仅开放 compute=fp32";
    return false;
  }
  if (spec.a_scale_kind == ScaleModeKind::kAuto) {
    spec.a_scale_kind = ScaleModeKind::kVec16Ue4M3;
  }
  if (spec.b_scale_kind == ScaleModeKind::kAuto) {
    spec.b_scale_kind = ScaleModeKind::kVec16Ue4M3;
  }
  if (spec.a_scale_kind != ScaleModeKind::kVec16Ue4M3 ||
      spec.b_scale_kind != ScaleModeKind::kVec16Ue4M3) {
    error = "fp4 路径当前仅开放 vec16ue4m3 scale mode";
    return false;
  }
  if (spec.m % 16 != 0 || spec.n % 16 != 0 || spec.k % 16 != 0) {
    error = "fp4 路径要求 M/N/K 为 16 的倍数";
    return false;
  }
  detail = "A 以转置布局打包为 fp4，scale_mode=vec16_ue4m3";
  return true;
}

void print_usage(const char *prog) {
  std::fprintf(stderr,
               "用法:\n"
               "  %s --list\n"
               "  %s all [M N K]\n"
               "  %s <function_name> [M N K]\n"
               "  %s <input_precision> <accum_precision> [M N K]\n"
               "  %s --a <type> --b <type> --c <type> --d <type> --compute <type> [options]\n\n"
               "固定函数示例:\n"
               "  %s f16a_f16b_acc32_gemm 1024 1024 1024\n"
               "  %s f32a_f32b_tf32_gemm\n"
               "  %s all 256 256 256\n\n"
               "旧预设示例:\n"
               "  %s fp16 fp32 1024 1024 1024\n"
               "  %s tf32 fp32\n\n"
               "通用混合精度示例:\n"
               "  %s --a fp16 --b bf16 --c fp32 --d fp16 --compute fp32 --m 1024 --n 1024 --k 1024\n"
               "  %s --a fp8e4m3 --b fp8e5m2 --c bf16 --d bf16 --compute fp32\n"
               "  %s --a fp4 --b fp4 --c bf16 --d bf16 --compute fp32 --a-scale vec16ue4m3 --b-scale vec16ue4m3\n\n"
               "支持 type:\n"
               "  fp32, fp16, bf16, fp8, fp8e4m3, fp8e5m2, fp4\n"
               "支持 compute:\n"
               "  fp16, fp32, tf32\n"
               "支持 scale mode:\n"
               "  auto, scalar32f, vec16ue4m3\n",
               prog, prog, prog, prog, prog, prog, prog, prog, prog, prog,
               prog, prog, prog);
}

RunSpec make_preset(const DataKind input_kind,
                    const ComputeKind compute_kind,
                    const DataKind c_kind,
                    const DataKind d_kind,
                    const ScaleModeKind a_scale_kind,
                    const ScaleModeKind b_scale_kind) {
  RunSpec spec;
  spec.a_kind = input_kind;
  spec.b_kind = input_kind;
  spec.c_kind = c_kind;
  spec.d_kind = d_kind;
  spec.compute_kind = compute_kind;
  spec.a_scale_kind = a_scale_kind;
  spec.b_scale_kind = b_scale_kind;
  return spec;
}

const std::vector<RunSpec> &preset_modes() {
  static const std::vector<RunSpec> presets = {
      make_preset(DataKind::kFp32, ComputeKind::kFp32, DataKind::kFp32,
                  DataKind::kFp32, ScaleModeKind::kAuto,
                  ScaleModeKind::kAuto),
      make_preset(DataKind::kFp32, ComputeKind::kTf32, DataKind::kFp32,
                  DataKind::kFp32, ScaleModeKind::kAuto,
                  ScaleModeKind::kAuto),
      make_preset(DataKind::kFp16, ComputeKind::kFp16, DataKind::kFp16,
                  DataKind::kFp16, ScaleModeKind::kAuto,
                  ScaleModeKind::kAuto),
      make_preset(DataKind::kFp16, ComputeKind::kFp32, DataKind::kFp32,
                  DataKind::kFp32, ScaleModeKind::kAuto,
                  ScaleModeKind::kAuto),
      make_preset(DataKind::kBf16, ComputeKind::kFp32, DataKind::kFp32,
                  DataKind::kFp32, ScaleModeKind::kAuto,
                  ScaleModeKind::kAuto),
      make_preset(DataKind::kFp8E4M3, ComputeKind::kFp32, DataKind::kBf16,
                  DataKind::kBf16, ScaleModeKind::kScalar32F,
                  ScaleModeKind::kScalar32F),
      make_preset(DataKind::kFp4E2M1, ComputeKind::kFp32, DataKind::kBf16,
                  DataKind::kBf16, ScaleModeKind::kVec16Ue4M3,
                  ScaleModeKind::kVec16Ue4M3),
  };
  return presets;
}

BenchmarkResult run_spec(const RunSpec &spec);

bool prepare_fixed_spec(RunSpec &spec,
                        const int m,
                        const int n,
                        const int k,
                        std::string &detail) {
  spec.m = m;
  spec.n = n;
  spec.k = k;
  std::string error;
  if (!validate_spec(spec, detail, error)) {
    std::fprintf(stderr, "非法固定函数配置: %s\n", error.c_str());
    return false;
  }
  return true;
}

#define DEFINE_FIXED_GEMM_FUNCTION(FUNC_NAME, SPEC_EXPR)                        \
  RunSpec make_##FUNC_NAME##_spec(const int m, const int n, const int k,       \
                                  std::string &detail) {                        \
    RunSpec spec = (SPEC_EXPR);                                                 \
    if (!prepare_fixed_spec(spec, m, n, k, detail)) {                           \
      return {};                                                                \
    }                                                                           \
    return spec;                                                                \
  }                                                                             \
  BenchmarkResult FUNC_NAME(const int m, const int n, const int k) {           \
    std::string detail;                                                         \
    const RunSpec spec = make_##FUNC_NAME##_spec(m, n, k, detail);             \
    if (spec.label.empty()) {                                                   \
      return {};                                                                \
    }                                                                           \
    return run_spec(spec);                                                      \
  }

DEFINE_FIXED_GEMM_FUNCTION(
    f32a_f32b_acc32_gemm,
    make_preset(DataKind::kFp32, ComputeKind::kFp32, DataKind::kFp32,
                DataKind::kFp32, ScaleModeKind::kAuto, ScaleModeKind::kAuto))

DEFINE_FIXED_GEMM_FUNCTION(
    f32a_f32b_tf32_gemm,
    make_preset(DataKind::kFp32, ComputeKind::kTf32, DataKind::kFp32,
                DataKind::kFp32, ScaleModeKind::kAuto, ScaleModeKind::kAuto))

DEFINE_FIXED_GEMM_FUNCTION(
    f16a_f16b_acc16_gemm,
    make_preset(DataKind::kFp16, ComputeKind::kFp16, DataKind::kFp16,
                DataKind::kFp16, ScaleModeKind::kAuto, ScaleModeKind::kAuto))

DEFINE_FIXED_GEMM_FUNCTION(
    f16a_f16b_acc32_gemm,
    make_preset(DataKind::kFp16, ComputeKind::kFp32, DataKind::kFp32,
                DataKind::kFp32, ScaleModeKind::kAuto, ScaleModeKind::kAuto))

DEFINE_FIXED_GEMM_FUNCTION(
    bf16a_bf16b_acc32_gemm,
    make_preset(DataKind::kBf16, ComputeKind::kFp32, DataKind::kFp32,
                DataKind::kFp32, ScaleModeKind::kAuto, ScaleModeKind::kAuto))

DEFINE_FIXED_GEMM_FUNCTION(
    fp8e4m3a_fp8e4m3b_acc32_gemm,
    make_preset(DataKind::kFp8E4M3, ComputeKind::kFp32, DataKind::kBf16,
                DataKind::kBf16, ScaleModeKind::kScalar32F,
                ScaleModeKind::kScalar32F))

DEFINE_FIXED_GEMM_FUNCTION(
    fp4a_fp4b_acc32_gemm,
    make_preset(DataKind::kFp4E2M1, ComputeKind::kFp32, DataKind::kBf16,
                DataKind::kBf16, ScaleModeKind::kVec16Ue4M3,
                ScaleModeKind::kVec16Ue4M3))

#undef DEFINE_FIXED_GEMM_FUNCTION

struct FixedEntry {
  const char *function_name;
  const char *description;
  RunSpec (*make_spec)(int m, int n, int k, std::string &detail);
  BenchmarkResult (*run)(int m, int n, int k);
};

const std::vector<FixedEntry> &fixed_entries() {
  static const std::vector<FixedEntry> entries = {
      {"f32a_f32b_acc32_gemm", "A/B=fp32, C/D=fp32, accumulate=fp32",
       make_f32a_f32b_acc32_gemm_spec, f32a_f32b_acc32_gemm},
      {"f32a_f32b_tf32_gemm",
       "A/B=fp32, C/D=fp32, tensor core compute=tf32",
       make_f32a_f32b_tf32_gemm_spec, f32a_f32b_tf32_gemm},
      {"f16a_f16b_acc16_gemm", "A/B/C/D=fp16, accumulate=fp16",
       make_f16a_f16b_acc16_gemm_spec, f16a_f16b_acc16_gemm},
      {"f16a_f16b_acc32_gemm", "A/B=fp16, C/D=fp32, accumulate=fp32",
       make_f16a_f16b_acc32_gemm_spec, f16a_f16b_acc32_gemm},
      {"bf16a_bf16b_acc32_gemm", "A/B=bf16, C/D=fp32, accumulate=fp32",
       make_bf16a_bf16b_acc32_gemm_spec, bf16a_bf16b_acc32_gemm},
      {"fp8e4m3a_fp8e4m3b_acc32_gemm",
       "A/B=fp8e4m3, C/D=bf16, accumulate=fp32",
       make_fp8e4m3a_fp8e4m3b_acc32_gemm_spec,
       fp8e4m3a_fp8e4m3b_acc32_gemm},
      {"fp4a_fp4b_acc32_gemm", "A/B=fp4, C/D=bf16, accumulate=fp32",
       make_fp4a_fp4b_acc32_gemm_spec, fp4a_fp4b_acc32_gemm},
  };
  return entries;
}

const FixedEntry *find_fixed_entry(const std::string &name) {
  for (const FixedEntry &entry : fixed_entries()) {
    if (name == entry.function_name) {
      return &entry;
    }
  }
  return nullptr;
}

void print_mode_list() {
  std::printf("固定函数模式:\n");
  for (const FixedEntry &entry : fixed_entries()) {
    std::printf("  %-30s %s\n", entry.function_name, entry.description);
  }
  std::printf("\n旧预设模式:\n");
  for (RunSpec spec : preset_modes()) {
    std::string detail;
    std::string error;
    validate_spec(spec, detail, error);
    std::printf("  %s\n", spec.label.c_str());
  }
  std::printf("\n通用模式可用字段:\n");
  std::printf("  --a/--b: fp32, fp16, bf16, fp8, fp8e4m3, fp8e5m2, fp4\n");
  std::printf("  --c/--d: fp32, fp16, bf16\n");
  std::printf("  --compute: fp16, fp32, tf32\n");
  std::printf("  --a-scale/--b-scale: auto, scalar32f, vec16ue4m3\n");
}

bool find_preset(const std::string &input_name,
                 const std::string &accum_name,
                 RunSpec &spec) {
  if (input_name == "fp32" && accum_name == "fp32") {
    spec = make_preset(DataKind::kFp32, ComputeKind::kFp32, DataKind::kFp32,
                       DataKind::kFp32, ScaleModeKind::kAuto,
                       ScaleModeKind::kAuto);
    return true;
  }
  if (input_name == "tf32" && accum_name == "fp32") {
    spec = make_preset(DataKind::kFp32, ComputeKind::kTf32, DataKind::kFp32,
                       DataKind::kFp32, ScaleModeKind::kAuto,
                       ScaleModeKind::kAuto);
    return true;
  }
  if (input_name == "fp16" && accum_name == "fp16") {
    spec = make_preset(DataKind::kFp16, ComputeKind::kFp16, DataKind::kFp16,
                       DataKind::kFp16, ScaleModeKind::kAuto,
                       ScaleModeKind::kAuto);
    return true;
  }
  if (input_name == "fp16" && accum_name == "fp32") {
    spec = make_preset(DataKind::kFp16, ComputeKind::kFp32, DataKind::kFp32,
                       DataKind::kFp32, ScaleModeKind::kAuto,
                       ScaleModeKind::kAuto);
    return true;
  }
  if (input_name == "bf16" && accum_name == "fp32") {
    spec = make_preset(DataKind::kBf16, ComputeKind::kFp32, DataKind::kFp32,
                       DataKind::kFp32, ScaleModeKind::kAuto,
                       ScaleModeKind::kAuto);
    return true;
  }
  if ((input_name == "fp8" || input_name == "fp8e4m3") &&
      accum_name == "fp32") {
    spec = make_preset(DataKind::kFp8E4M3, ComputeKind::kFp32,
                       DataKind::kBf16, DataKind::kBf16,
                       ScaleModeKind::kScalar32F,
                       ScaleModeKind::kScalar32F);
    return true;
  }
  if (input_name == "fp4" && accum_name == "fp32") {
    spec = make_preset(DataKind::kFp4E2M1, ComputeKind::kFp32,
                       DataKind::kBf16, DataKind::kBf16,
                       ScaleModeKind::kVec16Ue4M3,
                       ScaleModeKind::kVec16Ue4M3);
    return true;
  }
  return false;
}

void print_result(const RunSpec &spec,
                  const BenchmarkResult &result,
                  const std::string &detail) {
  using namespace cublas_demo;

  std::printf("cuBLASLt GEMM benchmark\n");
  std::printf("mode        = %s\n", spec.label.c_str());
  std::printf("A           = %s\n", data_kind_name(spec.a_kind));
  std::printf("B           = %s\n", data_kind_name(spec.b_kind));
  std::printf("C           = %s\n", data_kind_name(spec.c_kind));
  std::printf("D           = %s\n", data_kind_name(spec.d_kind));
  std::printf("compute     = %s\n", compute_kind_name(spec.compute_kind));
  std::printf("a_scale     = %s\n", scale_mode_name(spec.a_scale_kind));
  std::printf("b_scale     = %s\n", scale_mode_name(spec.b_scale_kind));
  std::printf("M=%d, N=%d, K=%d\n", spec.m, spec.n, spec.k);
  print_device_summary();
  if (!detail.empty()) {
    std::printf("details     = %s\n", detail.c_str());
  }
  std::printf("tolerance   = %.6g\n", spec.tolerance);
  std::printf("max_abs_err = %.6g\n", result.max_abs_err);
  std::printf("avg_time_ms = %.4f\n", result.avg_time_ms);
  std::printf("tflops      = %.4f\n", result.tflops);
  std::printf("status      = %s\n", result.passed ? "PASS" : "FAIL");
}

template <typename AType, typename BType, typename CType, typename DType,
          typename ScaleType>
BenchmarkResult run_dense_typed(const RunSpec &spec) {
  using namespace cublas_demo;

  const float input_scale = source_scale(spec);
  const std::vector<float> h_a_src =
      make_patterned_matrix_col_major(spec.m, spec.k, input_scale);
  const std::vector<float> h_b_src =
      make_patterned_matrix_col_major(spec.k, spec.n, input_scale);
  const std::vector<AType> h_a = quantize_vector<AType>(h_a_src);
  const std::vector<BType> h_b = quantize_vector<BType>(h_b_src);
  const std::vector<float> h_a_ref = dequantize_vector(h_a);
  const std::vector<float> h_b_ref = dequantize_vector(h_b);

  std::vector<CType> h_c(static_cast<size_t>(spec.m) * spec.n,
                         cast_from_float<CType>(0.0f));
  std::vector<DType> h_d(static_cast<size_t>(spec.m) * spec.n,
                         cast_from_float<DType>(0.0f));
  std::vector<float> h_ref(h_d.size(), 0.0f);
  cpu_gemm_col_major(h_a_ref, h_b_ref, h_ref, spec.m, spec.n, spec.k);

  AType *d_a = malloc_and_copy_to_device(h_a);
  BType *d_b = malloc_and_copy_to_device(h_b);
  CType *d_c = malloc_and_copy_to_device(h_c);
  DType *d_d = nullptr;
  CHECK_CUDA(cudaMalloc(&d_d, h_d.size() * sizeof(DType)));
  CHECK_CUDA(cudaMemset(d_d, 0, h_d.size() * sizeof(DType)));

  void *workspace = nullptr;
  CHECK_CUDA(cudaMalloc(&workspace, kWorkspaceBytes));

  cublasLtHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasLtCreate(&handle));

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  const cublasOperation_t transa = CUBLAS_OP_N;
  const cublasOperation_t transb = CUBLAS_OP_N;
  const cublasComputeType_t compute_type =
      to_cublas_compute_type(spec.compute_kind, spec.a_kind, spec.b_kind);
  const cudaDataType scale_type = to_scale_type(spec.compute_kind);
  CHECK_CUBLAS(cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc,
                                              CUBLASLT_MATMUL_DESC_TRANSA,
                                              &transa, sizeof(transa)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc,
                                              CUBLASLT_MATMUL_DESC_TRANSB,
                                              &transb, sizeof(transb)));

  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
      &a_desc, to_cuda_data_type(spec.a_kind), spec.m, spec.k, spec.m));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
      &b_desc, to_cuda_data_type(spec.b_kind), spec.k, spec.n, spec.k));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
      &c_desc, to_cuda_data_type(spec.c_kind), spec.m, spec.n, spec.m));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
      &d_desc, to_cuda_data_type(spec.d_kind), spec.m, spec.n, spec.m));

  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceBytes,
      sizeof(kWorkspaceBytes)));

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;
  BenchmarkResult result;
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, op_desc, a_desc, b_desc,
                                              c_desc, d_desc, preference, 1,
                                              &heuristic, &returned_results));
  if (returned_results == 0) {
    std::fprintf(stderr, "未找到 dense heuristic: %s\n", spec.label.c_str());
  } else {
    const ScaleType alpha = cast_from_float<ScaleType>(1.0f);
    const ScaleType beta = cast_from_float<ScaleType>(0.0f);
    result.avg_time_ms = benchmark_launch_ms(
        [&]() {
          CHECK_CUBLAS(cublasLtMatmul(handle, op_desc, &alpha, d_a, a_desc, d_b,
                                      b_desc, &beta, d_c, c_desc, d_d, d_desc,
                                      &heuristic.algo, workspace,
                                      kWorkspaceBytes, 0));
        },
        kWarmupIters, kRepeatIters);
    result.tflops = compute_tflops(spec.m, spec.n, spec.k, result.avg_time_ms);
    copy_from_device(h_d, d_d);
    const std::vector<float> h_d_float = dequantize_vector(h_d);
    result.max_abs_err = max_abs_diff(h_d_float, h_ref);
    result.passed = result.max_abs_err <= spec.tolerance;
  }

  CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(d_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(c_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(b_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(a_desc));
  CHECK_CUBLAS(cublasLtMatmulDescDestroy(op_desc));
  CHECK_CUBLAS(cublasLtDestroy(handle));
  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaFree(d_d));
  CHECK_CUDA(cudaFree(d_c));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_a));
  return result;
}

template <typename AType, typename BType, typename CType, typename DType>
BenchmarkResult run_fp8_typed(const RunSpec &spec) {
  using namespace cublas_demo;

  const float input_scale = source_scale(spec);
  const std::vector<float> h_a_src =
      make_patterned_matrix_col_major(spec.m, spec.k, input_scale);
  const std::vector<float> h_b_src =
      make_patterned_matrix_col_major(spec.k, spec.n, input_scale);
  const std::vector<AType> h_a = quantize_vector<AType>(h_a_src);
  const std::vector<BType> h_b = quantize_vector<BType>(h_b_src);
  const std::vector<float> h_a_ref = dequantize_vector(h_a);
  const std::vector<float> h_b_ref = dequantize_vector(h_b);

  std::vector<CType> h_c(static_cast<size_t>(spec.m) * spec.n,
                         cast_from_float<CType>(0.0f));
  std::vector<DType> h_d(static_cast<size_t>(spec.m) * spec.n,
                         cast_from_float<DType>(0.0f));
  std::vector<float> h_ref(h_d.size(), 0.0f);
  cpu_gemm_col_major(h_a_ref, h_b_ref, h_ref, spec.m, spec.n, spec.k);

  AType *d_a = malloc_and_copy_to_device(h_a);
  BType *d_b = malloc_and_copy_to_device(h_b);
  CType *d_c = malloc_and_copy_to_device(h_c);
  DType *d_d = nullptr;
  CHECK_CUDA(cudaMalloc(&d_d, h_d.size() * sizeof(DType)));
  CHECK_CUDA(cudaMemset(d_d, 0, h_d.size() * sizeof(DType)));

  const std::vector<float> h_scale(1, 1.0f);
  float *d_a_scale = malloc_and_copy_to_device(h_scale);
  float *d_b_scale = malloc_and_copy_to_device(h_scale);

  void *workspace = nullptr;
  CHECK_CUDA(cudaMalloc(&workspace, kWorkspaceBytes));

  cublasLtHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasLtCreate(&handle));

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  const cublasOperation_t transa = CUBLAS_OP_N;
  const cublasOperation_t transb = CUBLAS_OP_N;
  const cublasComputeType_t compute_type =
      to_cublas_compute_type(spec.compute_kind, spec.a_kind, spec.b_kind);
  CHECK_CUBLAS(
      cublasLtMatmulDescCreate(&op_desc, compute_type, CUDA_R_32F));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc,
                                              CUBLASLT_MATMUL_DESC_TRANSA,
                                              &transa, sizeof(transa)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc,
                                              CUBLASLT_MATMUL_DESC_TRANSB,
                                              &transb, sizeof(transb)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_a_scale,
      sizeof(d_a_scale)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_b_scale,
      sizeof(d_b_scale)));

  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
      &a_desc, to_cuda_data_type(spec.a_kind), spec.m, spec.k, spec.m));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
      &b_desc, to_cuda_data_type(spec.b_kind), spec.k, spec.n, spec.k));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
      &c_desc, to_cuda_data_type(spec.c_kind), spec.m, spec.n, spec.m));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
      &d_desc, to_cuda_data_type(spec.d_kind), spec.m, spec.n, spec.m));

  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceBytes,
      sizeof(kWorkspaceBytes)));

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;
  BenchmarkResult result;
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, op_desc, a_desc, b_desc,
                                              c_desc, d_desc, preference, 1,
                                              &heuristic, &returned_results));
  if (returned_results == 0) {
    std::fprintf(stderr, "未找到 fp8 heuristic: %s\n", spec.label.c_str());
  } else {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    result.avg_time_ms = benchmark_launch_ms(
        [&]() {
          CHECK_CUBLAS(cublasLtMatmul(handle, op_desc, &alpha, d_a, a_desc, d_b,
                                      b_desc, &beta, d_c, c_desc, d_d, d_desc,
                                      &heuristic.algo, workspace,
                                      kWorkspaceBytes, 0));
        },
        kWarmupIters, kRepeatIters);
    result.tflops = compute_tflops(spec.m, spec.n, spec.k, result.avg_time_ms);
    copy_from_device(h_d, d_d);
    const std::vector<float> h_d_float = dequantize_vector(h_d);
    result.max_abs_err = max_abs_diff(h_d_float, h_ref);
    result.passed = result.max_abs_err <= spec.tolerance;
  }

  CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(d_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(c_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(b_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(a_desc));
  CHECK_CUBLAS(cublasLtMatmulDescDestroy(op_desc));
  CHECK_CUBLAS(cublasLtDestroy(handle));
  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaFree(d_b_scale));
  CHECK_CUDA(cudaFree(d_a_scale));
  CHECK_CUDA(cudaFree(d_d));
  CHECK_CUDA(cudaFree(d_c));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_a));
  return result;
}

template <typename CType, typename DType>
BenchmarkResult run_fp4_typed(const RunSpec &spec) {
  using namespace cublas_demo;

  const float input_scale = source_scale(spec);
  const std::vector<float> h_a_storage_src =
      make_patterned_matrix_col_major(spec.k, spec.m, input_scale);
  const std::vector<float> h_b_src =
      make_patterned_matrix_col_major(spec.k, spec.n, input_scale);
  const std::vector<__nv_fp4x2_e2m1> h_a = pack_fp4_vector(h_a_storage_src);
  const std::vector<__nv_fp4x2_e2m1> h_b = pack_fp4_vector(h_b_src);
  const std::vector<float> h_a_storage_ref =
      unpack_fp4_vector(h_a, h_a_storage_src.size());
  std::vector<float> h_a_ref(static_cast<size_t>(spec.m) * spec.k, 0.0f);
  for (int kk = 0; kk < spec.k; ++kk) {
    for (int row = 0; row < spec.m; ++row) {
      h_a_ref[static_cast<size_t>(row) + static_cast<size_t>(kk) * spec.m] =
          h_a_storage_ref[static_cast<size_t>(kk) +
                          static_cast<size_t>(row) * spec.k];
    }
  }
  const std::vector<float> h_b_ref = unpack_fp4_vector(h_b, h_b_src.size());

  std::vector<CType> h_c(static_cast<size_t>(spec.m) * spec.n,
                         cast_from_float<CType>(0.0f));
  std::vector<DType> h_d(static_cast<size_t>(spec.m) * spec.n,
                         cast_from_float<DType>(0.0f));
  std::vector<float> h_ref(h_d.size(), 0.0f);
  cpu_gemm_col_major(h_a_ref, h_b_ref, h_ref, spec.m, spec.n, spec.k);

  const cublasLtMatmulMatrixScale_t a_scale_mode =
      to_cublaslt_scale_mode(spec.a_scale_kind);
  const cublasLtMatmulMatrixScale_t b_scale_mode =
      to_cublaslt_scale_mode(spec.b_scale_kind);
  const size_t a_scale_size = get_scale_tensor_size(spec.k, spec.m, a_scale_mode);
  const size_t b_scale_size = get_scale_tensor_size(spec.k, spec.n, b_scale_mode);
  const std::vector<__nv_fp8_e4m3> h_a_scale(
      a_scale_size, cast_from_float<__nv_fp8_e4m3>(1.0f));
  const std::vector<__nv_fp8_e4m3> h_b_scale(
      b_scale_size, cast_from_float<__nv_fp8_e4m3>(1.0f));

  __nv_fp4x2_e2m1 *d_a = malloc_and_copy_to_device(h_a);
  __nv_fp4x2_e2m1 *d_b = malloc_and_copy_to_device(h_b);
  CType *d_c = malloc_and_copy_to_device(h_c);
  DType *d_d = nullptr;
  CHECK_CUDA(cudaMalloc(&d_d, h_d.size() * sizeof(DType)));
  CHECK_CUDA(cudaMemset(d_d, 0, h_d.size() * sizeof(DType)));
  __nv_fp8_e4m3 *d_a_scale = malloc_and_copy_to_device(h_a_scale);
  __nv_fp8_e4m3 *d_b_scale = malloc_and_copy_to_device(h_b_scale);

  void *workspace = nullptr;
  CHECK_CUDA(cudaMalloc(&workspace, kWorkspaceBytes));

  cublasLtHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasLtCreate(&handle));

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatrixLayout_t d_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  const cublasOperation_t transa = CUBLAS_OP_T;
  const cublasOperation_t transb = CUBLAS_OP_N;
  const cublasComputeType_t compute_type =
      to_cublas_compute_type(spec.compute_kind, spec.a_kind, spec.b_kind);
  CHECK_CUBLAS(
      cublasLtMatmulDescCreate(&op_desc, compute_type, CUDA_R_32F));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc,
                                              CUBLASLT_MATMUL_DESC_TRANSA,
                                              &transa, sizeof(transa)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc,
                                              CUBLASLT_MATMUL_DESC_TRANSB,
                                              &transb, sizeof(transb)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc,
                                              CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                              &a_scale_mode,
                                              sizeof(a_scale_mode)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(op_desc,
                                              CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                              &b_scale_mode,
                                              sizeof(b_scale_mode)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_a_scale,
      sizeof(d_a_scale)));
  CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_b_scale,
      sizeof(d_b_scale)));

  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
      &a_desc, to_cuda_data_type(spec.a_kind), spec.k, spec.m, spec.k));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
      &b_desc, to_cuda_data_type(spec.b_kind), spec.k, spec.n, spec.k));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
      &c_desc, to_cuda_data_type(spec.c_kind), spec.m, spec.n, spec.m));
  CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
      &d_desc, to_cuda_data_type(spec.d_kind), spec.m, spec.n, spec.m));

  CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
  CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceBytes,
      sizeof(kWorkspaceBytes)));

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned_results = 0;
  BenchmarkResult result;
  CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, op_desc, a_desc, b_desc,
                                              c_desc, d_desc, preference, 1,
                                              &heuristic, &returned_results));
  if (returned_results == 0) {
    std::fprintf(stderr, "未找到 fp4 heuristic: %s\n", spec.label.c_str());
  } else {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    result.avg_time_ms = benchmark_launch_ms(
        [&]() {
          CHECK_CUBLAS(cublasLtMatmul(handle, op_desc, &alpha, d_a, a_desc, d_b,
                                      b_desc, &beta, d_c, c_desc, d_d, d_desc,
                                      &heuristic.algo, workspace,
                                      kWorkspaceBytes, 0));
        },
        kWarmupIters, kRepeatIters);
    result.tflops = compute_tflops(spec.m, spec.n, spec.k, result.avg_time_ms);
    copy_from_device(h_d, d_d);
    const std::vector<float> h_d_float = dequantize_vector(h_d);
    result.max_abs_err = max_abs_diff(h_d_float, h_ref);
    result.passed = result.max_abs_err <= spec.tolerance;
  }

  CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(preference));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(d_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(c_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(b_desc));
  CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(a_desc));
  CHECK_CUBLAS(cublasLtMatmulDescDestroy(op_desc));
  CHECK_CUBLAS(cublasLtDestroy(handle));
  CHECK_CUDA(cudaFree(workspace));
  CHECK_CUDA(cudaFree(d_b_scale));
  CHECK_CUDA(cudaFree(d_a_scale));
  CHECK_CUDA(cudaFree(d_d));
  CHECK_CUDA(cudaFree(d_c));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_a));
  return result;
}

template <typename DType>
BenchmarkResult dispatch_dense_d(const RunSpec &spec) {
  switch (spec.c_kind) {
  case DataKind::kFp32:
    switch (spec.b_kind) {
    case DataKind::kFp32:
      switch (spec.a_kind) {
      case DataKind::kFp32:
        return run_dense_typed<float, float, float, DType,
                               float>(spec);
      case DataKind::kFp16:
        return run_dense_typed<__half, float, float, DType,
                               float>(spec);
      case DataKind::kBf16:
        return run_dense_typed<__nv_bfloat16, float, float, DType,
                               float>(spec);
      default:
        break;
      }
      break;
    case DataKind::kFp16:
      switch (spec.a_kind) {
      case DataKind::kFp32:
        return run_dense_typed<float, __half, float, DType,
                               float>(spec);
      case DataKind::kFp16:
        if (spec.compute_kind == ComputeKind::kFp16) {
          return run_dense_typed<__half, __half, float, DType,
                                 __half>(spec);
        }
        return run_dense_typed<__half, __half, float, DType,
                               float>(spec);
      case DataKind::kBf16:
        return run_dense_typed<__nv_bfloat16, __half, float, DType,
                               float>(spec);
      default:
        break;
      }
      break;
    case DataKind::kBf16:
      switch (spec.a_kind) {
      case DataKind::kFp32:
        return run_dense_typed<float, __nv_bfloat16, float, DType,
                               float>(spec);
      case DataKind::kFp16:
        return run_dense_typed<__half, __nv_bfloat16, float, DType,
                               float>(spec);
      case DataKind::kBf16:
        return run_dense_typed<__nv_bfloat16, __nv_bfloat16, float, DType,
                               float>(spec);
      default:
        break;
      }
      break;
    default:
      break;
    }
    break;
  case DataKind::kFp16:
    if (spec.compute_kind == ComputeKind::kFp16 && spec.a_kind == DataKind::kFp16 &&
        spec.b_kind == DataKind::kFp16) {
      return run_dense_typed<__half, __half, __half, DType, __half>(spec);
    }
    switch (spec.b_kind) {
    case DataKind::kFp32:
      switch (spec.a_kind) {
      case DataKind::kFp32:
        return run_dense_typed<float, float, __half, DType, float>(spec);
      case DataKind::kFp16:
        return run_dense_typed<__half, float, __half, DType, float>(spec);
      case DataKind::kBf16:
        return run_dense_typed<__nv_bfloat16, float, __half, DType,
                               float>(spec);
      default:
        break;
      }
      break;
    case DataKind::kFp16:
      switch (spec.a_kind) {
      case DataKind::kFp32:
        return run_dense_typed<float, __half, __half, DType, float>(spec);
      case DataKind::kFp16:
        return run_dense_typed<__half, __half, __half, DType, float>(spec);
      case DataKind::kBf16:
        return run_dense_typed<__nv_bfloat16, __half, __half, DType,
                               float>(spec);
      default:
        break;
      }
      break;
    case DataKind::kBf16:
      switch (spec.a_kind) {
      case DataKind::kFp32:
        return run_dense_typed<float, __nv_bfloat16, __half, DType,
                               float>(spec);
      case DataKind::kFp16:
        return run_dense_typed<__half, __nv_bfloat16, __half, DType,
                               float>(spec);
      case DataKind::kBf16:
        return run_dense_typed<__nv_bfloat16, __nv_bfloat16, __half, DType,
                               float>(spec);
      default:
        break;
      }
      break;
    default:
      break;
    }
    break;
  case DataKind::kBf16:
    switch (spec.b_kind) {
    case DataKind::kFp32:
      switch (spec.a_kind) {
      case DataKind::kFp32:
        return run_dense_typed<float, float, __nv_bfloat16, DType,
                               float>(spec);
      case DataKind::kFp16:
        return run_dense_typed<__half, float, __nv_bfloat16, DType,
                               float>(spec);
      case DataKind::kBf16:
        return run_dense_typed<__nv_bfloat16, float, __nv_bfloat16, DType,
                               float>(spec);
      default:
        break;
      }
      break;
    case DataKind::kFp16:
      switch (spec.a_kind) {
      case DataKind::kFp32:
        return run_dense_typed<float, __half, __nv_bfloat16, DType,
                               float>(spec);
      case DataKind::kFp16:
        return run_dense_typed<__half, __half, __nv_bfloat16, DType,
                               float>(spec);
      case DataKind::kBf16:
        return run_dense_typed<__nv_bfloat16, __half, __nv_bfloat16, DType,
                               float>(spec);
      default:
        break;
      }
      break;
    case DataKind::kBf16:
      switch (spec.a_kind) {
      case DataKind::kFp32:
        return run_dense_typed<float, __nv_bfloat16, __nv_bfloat16, DType,
                               float>(spec);
      case DataKind::kFp16:
        return run_dense_typed<__half, __nv_bfloat16, __nv_bfloat16, DType,
                               float>(spec);
      case DataKind::kBf16:
        return run_dense_typed<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16,
                               DType, float>(spec);
      default:
        break;
      }
      break;
    default:
      break;
    }
    break;
  default:
    break;
  }

  return {};
}

BenchmarkResult dispatch_dense(const RunSpec &spec) {
  switch (spec.d_kind) {
  case DataKind::kFp32:
    return dispatch_dense_d<float>(spec);
  case DataKind::kFp16:
    return dispatch_dense_d<__half>(spec);
  case DataKind::kBf16:
    return dispatch_dense_d<__nv_bfloat16>(spec);
  default:
    return {};
  }
}

template <typename DType>
BenchmarkResult dispatch_fp8_d(const RunSpec &spec) {
  switch (spec.c_kind) {
  case DataKind::kFp32:
    if (spec.a_kind == DataKind::kFp8E4M3 && spec.b_kind == DataKind::kFp8E4M3) {
      return run_fp8_typed<__nv_fp8_e4m3, __nv_fp8_e4m3, float, DType>(spec);
    }
    if (spec.a_kind == DataKind::kFp8E4M3 && spec.b_kind == DataKind::kFp8E5M2) {
      return run_fp8_typed<__nv_fp8_e4m3, __nv_fp8_e5m2, float, DType>(spec);
    }
    if (spec.a_kind == DataKind::kFp8E5M2 && spec.b_kind == DataKind::kFp8E4M3) {
      return run_fp8_typed<__nv_fp8_e5m2, __nv_fp8_e4m3, float, DType>(spec);
    }
    return run_fp8_typed<__nv_fp8_e5m2, __nv_fp8_e5m2, float, DType>(spec);
  case DataKind::kFp16:
    if (spec.a_kind == DataKind::kFp8E4M3 && spec.b_kind == DataKind::kFp8E4M3) {
      return run_fp8_typed<__nv_fp8_e4m3, __nv_fp8_e4m3, __half, DType>(spec);
    }
    if (spec.a_kind == DataKind::kFp8E4M3 && spec.b_kind == DataKind::kFp8E5M2) {
      return run_fp8_typed<__nv_fp8_e4m3, __nv_fp8_e5m2, __half, DType>(spec);
    }
    if (spec.a_kind == DataKind::kFp8E5M2 && spec.b_kind == DataKind::kFp8E4M3) {
      return run_fp8_typed<__nv_fp8_e5m2, __nv_fp8_e4m3, __half, DType>(spec);
    }
    return run_fp8_typed<__nv_fp8_e5m2, __nv_fp8_e5m2, __half, DType>(spec);
  case DataKind::kBf16:
    if (spec.a_kind == DataKind::kFp8E4M3 && spec.b_kind == DataKind::kFp8E4M3) {
      return run_fp8_typed<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16,
                           DType>(spec);
    }
    if (spec.a_kind == DataKind::kFp8E4M3 && spec.b_kind == DataKind::kFp8E5M2) {
      return run_fp8_typed<__nv_fp8_e4m3, __nv_fp8_e5m2, __nv_bfloat16,
                           DType>(spec);
    }
    if (spec.a_kind == DataKind::kFp8E5M2 && spec.b_kind == DataKind::kFp8E4M3) {
      return run_fp8_typed<__nv_fp8_e5m2, __nv_fp8_e4m3, __nv_bfloat16,
                           DType>(spec);
    }
    return run_fp8_typed<__nv_fp8_e5m2, __nv_fp8_e5m2, __nv_bfloat16,
                         DType>(spec);
  default:
    return {};
  }
}

BenchmarkResult dispatch_fp8(const RunSpec &spec) {
  switch (spec.d_kind) {
  case DataKind::kFp32:
    return dispatch_fp8_d<float>(spec);
  case DataKind::kFp16:
    return dispatch_fp8_d<__half>(spec);
  case DataKind::kBf16:
    return dispatch_fp8_d<__nv_bfloat16>(spec);
  default:
    return {};
  }
}

template <typename DType>
BenchmarkResult dispatch_fp4_d(const RunSpec &spec) {
  switch (spec.c_kind) {
  case DataKind::kFp32:
    return run_fp4_typed<float, DType>(spec);
  case DataKind::kFp16:
    return run_fp4_typed<__half, DType>(spec);
  case DataKind::kBf16:
    return run_fp4_typed<__nv_bfloat16, DType>(spec);
  default:
    return {};
  }
}

BenchmarkResult dispatch_fp4(const RunSpec &spec) {
  switch (spec.d_kind) {
  case DataKind::kFp32:
    return dispatch_fp4_d<float>(spec);
  case DataKind::kFp16:
    return dispatch_fp4_d<__half>(spec);
  case DataKind::kBf16:
    return dispatch_fp4_d<__nv_bfloat16>(spec);
  default:
    return {};
  }
}

BenchmarkResult run_spec(const RunSpec &spec) {
  switch (infer_runner_kind(spec)) {
  case RunnerKind::kDense:
    return dispatch_dense(spec);
  case RunnerKind::kFp8:
    return dispatch_fp8(spec);
  case RunnerKind::kFp4:
    return dispatch_fp4(spec);
  }

  return {};
}

int run_all_modes(const int m, const int n, const int k) {
  int final_status = EXIT_SUCCESS;
  for (const FixedEntry &entry : fixed_entries()) {
    std::string detail;
    const RunSpec spec = entry.make_spec(m, n, k, detail);
    if (spec.label.empty()) {
      final_status = EXIT_FAILURE;
      continue;
    }
    const BenchmarkResult result = entry.run(m, n, k);
    print_result(spec, result, detail);
    std::printf("\n");
    if (!result.passed) {
      final_status = EXIT_FAILURE;
    }
  }
  return final_status;
}

bool parse_generic_args(int argc, char **argv, RunSpec &spec) {
  bool seen_a = false;
  bool seen_b = false;
  bool seen_c = false;
  bool seen_d = false;
  bool seen_compute = false;

  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    auto require_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s 缺少取值\n", name);
        std::exit(EXIT_FAILURE);
      }
      ++i;
      return argv[i];
    };

    if (key == "--a") {
      DataKind kind{};
      if (!parse_data_kind(require_value("--a"), kind)) {
        std::fprintf(stderr, "不支持的 --a 类型\n");
        return false;
      }
      spec.a_kind = kind;
      seen_a = true;
      continue;
    }
    if (key == "--b") {
      DataKind kind{};
      if (!parse_data_kind(require_value("--b"), kind)) {
        std::fprintf(stderr, "不支持的 --b 类型\n");
        return false;
      }
      spec.b_kind = kind;
      seen_b = true;
      continue;
    }
    if (key == "--c") {
      DataKind kind{};
      if (!parse_data_kind(require_value("--c"), kind)) {
        std::fprintf(stderr, "不支持的 --c 类型\n");
        return false;
      }
      spec.c_kind = kind;
      seen_c = true;
      continue;
    }
    if (key == "--d") {
      DataKind kind{};
      if (!parse_data_kind(require_value("--d"), kind)) {
        std::fprintf(stderr, "不支持的 --d 类型\n");
        return false;
      }
      spec.d_kind = kind;
      seen_d = true;
      continue;
    }
    if (key == "--compute") {
      ComputeKind kind{};
      if (!parse_compute_kind(require_value("--compute"), kind)) {
        std::fprintf(stderr, "不支持的 --compute 类型\n");
        return false;
      }
      spec.compute_kind = kind;
      seen_compute = true;
      continue;
    }
    if (key == "--a-scale") {
      ScaleModeKind kind{};
      if (!parse_scale_mode_kind(require_value("--a-scale"), kind)) {
        std::fprintf(stderr, "不支持的 --a-scale 类型\n");
        return false;
      }
      spec.a_scale_kind = kind;
      continue;
    }
    if (key == "--b-scale") {
      ScaleModeKind kind{};
      if (!parse_scale_mode_kind(require_value("--b-scale"), kind)) {
        std::fprintf(stderr, "不支持的 --b-scale 类型\n");
        return false;
      }
      spec.b_scale_kind = kind;
      continue;
    }
    if (key == "--m") {
      spec.m = cublas_demo::parse_positive_int(require_value("--m"), "M");
      continue;
    }
    if (key == "--n") {
      spec.n = cublas_demo::parse_positive_int(require_value("--n"), "N");
      continue;
    }
    if (key == "--k") {
      spec.k = cublas_demo::parse_positive_int(require_value("--k"), "K");
      continue;
    }
    if (key == "--tolerance") {
      spec.tolerance = parse_positive_float(require_value("--tolerance"),
                                            "tolerance");
      spec.tolerance_overridden = true;
      continue;
    }
    if (key == "--help" || key == "-h") {
      return false;
    }

    std::fprintf(stderr, "未知参数: %s\n", argv[i]);
    return false;
  }

  return seen_a && seen_b && seen_c && seen_d && seen_compute;
}

} // namespace

int main(int argc, char **argv) {
  using namespace cublas_demo;

  if (argc == 2 && std::string(argv[1]) == "--list") {
    print_mode_list();
    return EXIT_SUCCESS;
  }

  if (argc == 1) {
    ensure_cuda_device();
    return run_all_modes(kDefaultDim, kDefaultDim, kDefaultDim);
  }

  if (argc == 2 && std::string(argv[1]) == "all") {
    ensure_cuda_device();
    return run_all_modes(kDefaultDim, kDefaultDim, kDefaultDim);
  }

  if (argc == 5 && std::string(argv[1]) == "all") {
    const int m = parse_positive_int(argv[2], "M");
    const int n = parse_positive_int(argv[3], "N");
    const int k = parse_positive_int(argv[4], "K");
    ensure_cuda_device();
    return run_all_modes(m, n, k);
  }

  if ((argc == 2 || argc == 5) && argv[1][0] != '-') {
    const FixedEntry *entry = find_fixed_entry(argv[1]);
    if (entry != nullptr) {
      int m = kDefaultDim;
      int n = kDefaultDim;
      int k = kDefaultDim;
      if (argc == 5) {
        m = parse_positive_int(argv[2], "M");
        n = parse_positive_int(argv[3], "N");
        k = parse_positive_int(argv[4], "K");
      }
      std::string detail;
      const RunSpec spec = entry->make_spec(m, n, k, detail);
      if (spec.label.empty()) {
        return EXIT_FAILURE;
      }
      ensure_cuda_device();
      const BenchmarkResult result = entry->run(m, n, k);
      print_result(spec, result, detail);
      return result.passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
  }

  if ((argc == 3 || argc == 6) && argv[1][0] != '-') {
    RunSpec spec;
    if (!find_preset(argv[1], argv[2], spec)) {
      std::fprintf(stderr, "不支持的预设组合: %s + %s\n", argv[1], argv[2]);
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
    if (argc == 6) {
      spec.m = parse_positive_int(argv[3], "M");
      spec.n = parse_positive_int(argv[4], "N");
      spec.k = parse_positive_int(argv[5], "K");
    }
    std::string detail;
    std::string error;
    if (!validate_spec(spec, detail, error)) {
      std::fprintf(stderr, "非法预设: %s\n", error.c_str());
      return EXIT_FAILURE;
    }
    ensure_cuda_device();
    const BenchmarkResult result = run_spec(spec);
    print_result(spec, result, detail);
    return result.passed ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  RunSpec spec;
  if (!parse_generic_args(argc, argv, spec)) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  std::string detail;
  std::string error;
  if (!validate_spec(spec, detail, error)) {
    std::fprintf(stderr, "非法配置: %s\n", error.c_str());
    return EXIT_FAILURE;
  }

  ensure_cuda_device();
  const BenchmarkResult result = run_spec(spec);
  print_result(spec, result, detail);
  return result.passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
