#!/usr/bin/env bash
set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[gitc] Error: current directory is not a git repository."
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

branch="$(git rev-parse --abbrev-ref HEAD)"
custom_subject="${1:-}"
subject=""
max_items="${GITC_MAX_ITEMS:-120}"

echo "[gitc] Running: git add -A"
git add -A

if git diff --cached --quiet; then
  echo "[gitc] No changes staged after git add -A. Nothing to commit or push."
  exit 0
fi

mapfile -t status_lines < <(git diff --cached --name-status)
file_body_lines=()
semantic_lines=()
support_lines=()
fallback_semantic_lines=()
total_items="${#status_lines[@]}"
semantic_limit="${GITC_SEMANTIC_MAX_ITEMS:-12}"

count_add=0
count_update=0
count_remove=0
count_rename=0
count_copy=0
count_other=0
declare -A dir_seen=()
dir_order=()
declare -A semantic_seen=()
declare -A support_seen=()
declare -A fallback_seen=()
declare -A symbol_seen=()
primary_path=""
has_cuda_kernel=0
changed_paths=()
changed_codes=()
standard_headers=()
cuda_headers=()
cublas_headers=()
other_headers=()
macro_names=()
const_names=()
kernel_names=()
function_names=()
class_names=()
has_cuda_gemm=0
has_cublas_demo=0
has_cublaslt_demo=0
has_async_demo=0
has_demo_utils=0
has_run_cuda_script=0

track_dir() {
  local path="$1"
  local dir
  if [[ "$path" == *"/"* ]]; then
    dir="${path%/*}/"
  else
    dir="(repo-root)"
  fi

  if [[ -z "${dir_seen[$dir]+x}" ]]; then
    dir_seen["$dir"]=1
    dir_order+=("$dir")
  fi
}

add_semantic_line() {
  local text="$1"
  [[ -z "$text" ]] && return
  if [[ -z "${semantic_seen[$text]+x}" ]]; then
    semantic_seen["$text"]=1
    semantic_lines+=("- ${text}")
  fi
}

add_support_line() {
  local text="$1"
  [[ -z "$text" ]] && return
  if [[ -z "${support_seen[$text]+x}" ]]; then
    support_seen["$text"]=1
    support_lines+=("- ${text}")
  fi
}

add_fallback_line() {
  local text="$1"
  [[ -z "$text" ]] && return
  if [[ -z "${fallback_seen[$text]+x}" ]]; then
    fallback_seen["$text"]=1
    fallback_semantic_lines+=("- ${text}")
  fi
}

join_preview() {
  local max_count="$1"
  shift || true
  local items=("$@")
  local total="${#items[@]}"
  if ((total == 0)); then
    echo ""
    return
  fi

  local show_count="$total"
  if ((show_count > max_count)); then
    show_count="$max_count"
  fi

  local picked=()
  local i
  for ((i = 0; i < show_count; i++)); do
    picked+=("${items[i]}")
  done

  local text
  text="$(IFS='、'; echo "${picked[*]}")"
  if ((total > max_count)); then
    text="${text} 等 ${total} 项"
  fi
  echo "$text"
}

record_symbol() {
  local group="$1"
  local value="$2"
  local key="${group}:${value}"

  [[ -z "$value" ]] && return
  if [[ -n "${symbol_seen[$key]+x}" ]]; then
    return
  fi
  symbol_seen["$key"]=1

  case "$group" in
    standard_header)
      standard_headers+=("$value")
      ;;
    cuda_header)
      cuda_headers+=("$value")
      ;;
    cublas_header)
      cublas_headers+=("$value")
      ;;
    other_header)
      other_headers+=("$value")
      ;;
    macro)
      macro_names+=("$value")
      ;;
    constant)
      const_names+=("$value")
      ;;
    kernel)
      kernel_names+=("$value")
      ;;
    function)
      function_names+=("$value")
      ;;
    class)
      class_names+=("$value")
      ;;
  esac
}

record_header() {
  local header="$1"

  case "$header" in
    algorithm|array|atomic|chrono|cmath|complex|cstddef|cstdint|cstdio|cstdlib|cstring|deque|exception|filesystem|float.h|functional|iomanip|iostream|limits|list|map|memory|numeric|optional|random|set|span|sstream|string|tuple|type_traits|unordered_map|unordered_set|utility|vector)
      record_symbol "standard_header" "$header"
      ;;
    cublasLt.h|cublas_v2.h|cublas*.h)
      record_symbol "cublas_header" "$header"
      ;;
    cuda*.h|cooperative_groups*.h|mma.h|curand*.h)
      record_symbol "cuda_header" "$header"
      ;;
    *)
      record_symbol "other_header" "$header"
      ;;
  esac
}

detect_path_scope() {
  local first_seg=""
  local second_seg=""
  local path
  local current_first=""
  local current_second=""

  if (( ${#changed_paths[@]} == 0 )); then
    echo "仓库"
    return
  fi

  for path in "${changed_paths[@]}"; do
    IFS='/' read -r current_first current_second _ <<< "$path"

    if [[ -z "$current_first" ]]; then
      echo "仓库"
      return
    fi

    if [[ -z "$first_seg" ]]; then
      first_seg="$current_first"
      second_seg="$current_second"
      continue
    fi

    if [[ "$first_seg" != "$current_first" ]]; then
      echo "仓库"
      return
    fi

    if [[ -n "$second_seg" && "$second_seg" != "$current_second" ]]; then
      second_seg=""
    fi
  done

  if [[ -n "$first_seg" && -n "$second_seg" ]]; then
    echo "$first_seg/$second_seg"
  elif [[ -n "$first_seg" ]]; then
    echo "$first_seg"
  else
    echo "仓库"
  fi
}

register_path_intent() {
  local code="$1"
  local path="$2"
  local base

  base="$(basename "$path")"

  case "$path" in
    learn/cuda/kernels/gemm/*)
      has_cuda_gemm=1
      case "$base" in
        cublas_demo_utils.h)
          has_demo_utils=1
          if [[ "$code" == A ]]; then
            add_semantic_line "抽取 cuBLAS/cuBLASLt GEMM 示例公共工具"
          else
            add_semantic_line "完善 cuBLAS/cuBLASLt GEMM 示例公共工具"
          fi
          ;;
        sgemm_async.cu)
          has_async_demo=1
          if [[ "$code" == A ]]; then
            add_semantic_line "新增异步拷贝版 SGEMM CUDA 示例"
          else
            add_semantic_line "完善异步拷贝版 SGEMM CUDA 示例"
          fi
          ;;
        sgemm_cublas_ex.cu)
          has_cublas_demo=1
          if [[ "$code" == A ]]; then
            add_semantic_line "新增 cuBLASEx SGEMM 示例"
          else
            add_semantic_line "完善 cuBLASEx SGEMM 示例"
          fi
          ;;
        sgemm_cublaslt_fp4.cu)
          has_cublaslt_demo=1
          if [[ "$code" == A ]]; then
            add_semantic_line "新增 cuBLASLt FP4 GEMM 示例"
          else
            add_semantic_line "完善 cuBLASLt FP4 GEMM 示例"
          fi
          ;;
        sgemm_cublaslt_fp8.cu)
          has_cublaslt_demo=1
          if [[ "$code" == A ]]; then
            add_semantic_line "新增 cuBLASLt FP8 GEMM 示例"
          else
            add_semantic_line "完善 cuBLASLt FP8 GEMM 示例"
          fi
          ;;
        sgemm_cublas.cu)
          has_cublas_demo=1
          if [[ "$code" == A ]]; then
            add_semantic_line "新增 cuBLAS SGEMM 示例"
          else
            add_semantic_line "完善 cuBLAS SGEMM 示例"
          fi
          ;;
        sgemm.cu)
          add_semantic_line "调整基础 SGEMM CUDA 示例"
          ;;
      esac
      ;;
    learn/cuda/run_cuda.sh)
      has_run_cuda_script=1
      if [[ "$code" == A ]]; then
        add_semantic_line "新增 CUDA demo 编译与运行脚本"
      else
        add_semantic_line "增强 CUDA demo 编译与运行脚本"
      fi
      ;;
    .codex/skills/gitc/scripts/gitc_push.sh)
      add_semantic_line "调整 gitc 提交信息生成规则"
      ;;
    .codex/skills/gitc/SKILL.md)
      add_semantic_line "明确 gitc 提交信息应优先描述功能变化"
      ;;
    .codex/skills/gitc/agents/openai.yaml)
      add_semantic_line "更新 gitc 默认提示词，强调功能摘要"
      ;;
  esac
}

emit_collected_semantics() {
  if (( ${#cublas_headers[@]} > 0 )); then
    add_support_line "补充 cuBLAS/cuBLASLt 依赖 ($(join_preview 4 "${cublas_headers[@]}"))"
  fi

  if (( ${#cuda_headers[@]} > 0 )); then
    add_support_line "补充 CUDA 混合精度与运行时头文件 ($(join_preview 4 "${cuda_headers[@]}"))"
  fi

  if (( ${#standard_headers[@]} > 0 )); then
    add_support_line "补充算法/数值/容器等基础依赖 ($(join_preview 4 "${standard_headers[@]}"))"
  fi

  if (( ${#other_headers[@]} > 0 )); then
    add_support_line "补充其他编译依赖 ($(join_preview 4 "${other_headers[@]}"))"
  fi

  if (( ${#macro_names[@]} > 0 )); then
    add_support_line "定义辅助宏 ($(join_preview 4 "${macro_names[@]}"))"
  fi

  if (( ${#const_names[@]} > 0 )); then
    add_support_line "定义常量 ($(join_preview 4 "${const_names[@]}"))"
  fi

  if (( ${#kernel_names[@]} > 0 )); then
    if (( ${#semantic_lines[@]} == 0 )); then
      add_semantic_line "实现 CUDA 内核 ($(join_preview 4 "${kernel_names[@]}"))"
    else
      add_support_line "涉及 CUDA 内核 ($(join_preview 4 "${kernel_names[@]}"))"
    fi
  fi

  if (( ${#function_names[@]} > 0 )); then
    add_support_line "补充函数/工具 ($(join_preview 4 "${function_names[@]}"))"
  fi

  if (( ${#class_names[@]} > 0 )); then
    add_support_line "补充类型定义 ($(join_preview 4 "${class_names[@]}"))"
  fi
}

detect_subject_scope() {
  local path
  local skill_name=""
  local current_skill=""
  local saw_skill=0
  local saw_non_skill=0

  if (( ${#changed_paths[@]} == 0 )); then
    echo "gitc skill"
    return
  fi

  for path in "${changed_paths[@]}"; do
    if [[ "$path" =~ ^\.codex/skills/([^/]+)/ ]]; then
      saw_skill=1
      current_skill="${BASH_REMATCH[1]}"
      if [[ -z "$skill_name" ]]; then
        skill_name="$current_skill"
      elif [[ "$skill_name" != "$current_skill" ]]; then
        echo "仓库"
        return
      fi
    else
      saw_non_skill=1
    fi
  done

  if ((saw_skill == 1 && saw_non_skill == 1)); then
    echo "仓库"
  elif [[ -n "$skill_name" ]]; then
    echo "${skill_name} skill"
  else
    detect_path_scope
  fi
}

detect_subject_action() {
  local non_add_count
  local all_doc_like=1
  local path
  local diff_text

  non_add_count=$((count_update + count_remove + count_rename + count_copy + count_other))
  diff_text="$(git diff --cached --no-color -U0)"

  if ((count_add > 0 && count_add >= non_add_count)); then
    echo "新增"
    return
  fi

  if printf '%s\n' "$diff_text" | grep -Eiq '(fix|bug|修复|错误|异常|兼容|回归|crash|panic)'; then
    echo "修复"
    return
  fi

  if ((count_remove > 0 && count_add == 0 && count_update == 0 && count_other == 0 && count_rename == 0 && count_copy == 0)); then
    echo "清理"
    return
  fi

  if ((count_rename > 0 && count_add == 0 && count_remove == 0 && count_copy == 0)); then
    echo "重构"
    return
  fi

  for path in "${changed_paths[@]}"; do
    case "$path" in
      *.md|*.rst|*.txt|LICENSE|LICENSE.txt|NOTICE|NOTICE.txt)
        ;;
      *)
        all_doc_like=0
        break
        ;;
    esac
  done

  if ((all_doc_like == 1)); then
    echo "更新"
    return
  fi

  if ((count_remove > 0 && (count_update + count_other) > 0)); then
    echo "修复"
    return
  fi

  echo "更新"
}

summarize_file_patch() {
  local code="$1"
  local path="$2"
  local summary_path="$path"
  local patch
  local added
  local removed
  local -a points=()

  patch="$(git diff --cached --no-color -U0 -- "$path" || true)"
  [[ -z "$patch" ]] && return

  added="$(printf '%s\n' "$patch" | sed -nE '/^\+[^+]/p')"
  removed="$(printf '%s\n' "$patch" | sed -nE '/^\-[^-]/p')"

  if [[ "$code" == A ]]; then
    points+=("新增文件")
  elif [[ "$code" == D ]]; then
    points+=("删除文件")
  fi

  if [[ "$path" == *.sh ]]; then
    mapfile -t fn_names < <(printf '%s\n' "$added" | sed -nE 's/^\+[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)\(\)[[:space:]]*\{.*/\1/p' | sort -u)
    if (( ${#fn_names[@]} > 0 )); then
      points+=("新增函数 $(join_preview 4 "${fn_names[@]}")")
    fi

    mapfile -t add_vars < <(printf '%s\n' "$added" | sed -nE 's/^\+[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' | sort -u)
    mapfile -t del_vars < <(printf '%s\n' "$removed" | sed -nE 's/^\-[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=.*/\1/p' | sort -u)

    if (( ${#add_vars[@]} > 0 || ${#del_vars[@]} > 0 )); then
      declare -A added_var_set=()
      declare -A removed_var_set=()
      declare -a tuned_vars=()
      declare -a new_vars=()

      for var_name in "${add_vars[@]}"; do
        added_var_set["$var_name"]=1
      done
      for var_name in "${del_vars[@]}"; do
        removed_var_set["$var_name"]=1
      done

      for var_name in "${add_vars[@]}"; do
        if [[ -n "${removed_var_set[$var_name]+x}" ]]; then
          tuned_vars+=("$var_name")
        else
          new_vars+=("$var_name")
        fi
      done

      if (( ${#tuned_vars[@]} > 0 )); then
        points+=("调整参数 $(join_preview 5 "${tuned_vars[@]}")")
      fi
      if (( ${#new_vars[@]} > 0 )); then
        points+=("新增参数 $(join_preview 5 "${new_vars[@]}")")
      fi
    fi

    if printf '%s\n' "$added" | grep -q 'semantic_lines\|add_semantic_line\|build_heading'; then
      points+=("增强提交信息语义摘要逻辑")
    fi
    if printf '%s\n' "$added" | grep -q 'stat_line\|涉及目录\|文件统计'; then
      points+=("补充文件统计与目录摘要")
    fi
    if printf '%s\n' "$added" | grep -q '#include\|#define\|__global__'; then
      points+=("新增代码模式识别（头文件/宏/CUDA 内核）")
    fi
    if printf '%s\n' "$added" | grep -q '同步工作区更新' && printf '%s\n' "$removed" | grep -q 'sync workspace updates'; then
      points+=("默认提交标题改为中文")
    fi
  fi

  if [[ "$path" == *.md ]]; then
    if printf '%s\n' "$added" | grep -q '语义摘要'; then
      points+=("文档规则改为语义摘要优先")
    fi
    if printf '%s\n' "$added" | grep -q '省略项提示' && printf '%s\n' "$removed" | grep -q 'remaining'; then
      points+=("文档补充省略项说明要求")
    fi
    if printf '%s\n' "$added" | grep -q '同步工作区更新' && printf '%s\n' "$removed" | grep -q 'sync workspace updates'; then
      points+=("文档默认标题示例改为中文")
    fi

    mapfile -t md_lines < <(printf '%s\n' "$added" | sed -nE '
      s/^\+//;
      s/^[[:space:]]+//;
      /^[[:space:]]*$/d;
      /^#/d;
      /^```/d;
      s/`//g;
      p
    ' | head -n 2)
    if (( ${#md_lines[@]} > 0 )); then
      md_preview="$(join_preview 2 "${md_lines[@]}")"
      points+=("文档新增要点 ${md_preview}")
    fi
  fi

  if [[ "$path" != *.md && "$path" != *.sh ]]; then
    mapfile -t func_names < <(printf '%s\n' "$added" | sed -nE '
      s/^\+[[:space:]]*def[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)\(.*/\1/p;
      s/^\+[[:space:]]*class[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*/\1/p;
      s/^\+[[:space:]]*__global__[[:space:]]+void[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)\(.*/\1/p;
      s/^\+[[:space:]]*[A-Za-z_][A-Za-z0-9_<>[:space:]\*]*[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*\(.*/\1/p
    ' | sort -u | head -n 5)
    if (( ${#func_names[@]} > 0 )); then
      points+=("涉及函数/类 $(join_preview 5 "${func_names[@]}")")
    fi
  fi

  if (( ${#points[@]} > 0 )); then
    local point_text
    point_text="$(join_preview 3 "${points[@]}")"
    add_fallback_line "更新 ${summary_path}：${point_text}"
  fi
}

build_heading() {
  local non_add_count
  local action
  local stem

  non_add_count=$((count_update + count_remove + count_rename + count_copy + count_other))
  action="更新"
  if ((count_add > 0 && non_add_count == 0)); then
    action="添加"
  fi

  if ((has_cuda_kernel == 1)); then
    if ((has_cuda_gemm == 1)); then
      if ((count_add > 0)); then
        echo "补充 CUDA GEMM 示例与公共工具"
      else
        echo "完善 CUDA GEMM 示例与公共工具"
      fi
      return
    fi

    if [[ -n "$primary_path" && "$primary_path" == *.cu ]]; then
      stem="$(basename "$primary_path")"
      stem="${stem%.*}"
      stem="$(echo "$stem" | tr '[:lower:]' '[:upper:]')"
      if [[ -n "$stem" ]]; then
        echo "${action} ${stem} CUDA 内核实现"
        return
      fi
    fi
    echo "${action} CUDA 内核实现"
    return
  fi

  if ((has_cuda_gemm == 1)); then
    if ((count_add > 0)); then
      echo "补充 CUDA GEMM 示例与运行支持"
    else
      echo "完善 CUDA GEMM 示例与运行支持"
    fi
    return
  fi

  if ((count_add > 0 && non_add_count == 0)); then
    echo "新增内容摘要"
  elif ((count_update + count_other > 0 && count_add == 0 && count_remove == 0 && count_rename == 0 && count_copy == 0)); then
    echo "代码更新摘要"
  else
    echo "变更摘要"
  fi
}

for ((i = 0; i < total_items; i++)); do
  line="${status_lines[i]}"
  [[ -z "$line" ]] && continue

  code="${line%%$'\t'*}"
  rest="${line#*$'\t'}"

  case "$code" in
    A)
      [[ -z "$primary_path" ]] && primary_path="$rest"
      changed_paths+=("$rest")
      changed_codes+=("A")
      register_path_intent "A" "$rest"
      ((count_add += 1))
      track_dir "$rest"
      if ((i < max_items)); then
        file_body_lines+=("- 新增 ${rest}")
      fi
      ;;
    M)
      [[ -z "$primary_path" ]] && primary_path="$rest"
      changed_paths+=("$rest")
      changed_codes+=("M")
      register_path_intent "M" "$rest"
      ((count_update += 1))
      track_dir "$rest"
      if ((i < max_items)); then
        file_body_lines+=("- 更新 ${rest}")
      fi
      ;;
    D)
      [[ -z "$primary_path" ]] && primary_path="$rest"
      changed_paths+=("$rest")
      changed_codes+=("D")
      register_path_intent "D" "$rest"
      ((count_remove += 1))
      track_dir "$rest"
      if ((i < max_items)); then
        file_body_lines+=("- 删除 ${rest}")
      fi
      ;;
    R*|C*)
      src="${rest%%$'\t'*}"
      dst="${rest#*$'\t'}"
      [[ -z "$primary_path" ]] && primary_path="$dst"
      changed_paths+=("$dst")
      changed_codes+=("$code")
      register_path_intent "$code" "$dst"
      if [[ "$code" == R* ]]; then
        ((count_rename += 1))
        if ((i < max_items)); then
          file_body_lines+=("- 重命名 ${src} -> ${dst}")
        fi
      else
        ((count_copy += 1))
        if ((i < max_items)); then
          file_body_lines+=("- 复制 ${src} -> ${dst}")
        fi
      fi
      track_dir "$dst"
      ;;
    *)
      [[ -z "$primary_path" ]] && primary_path="$rest"
      changed_paths+=("$rest")
      changed_codes+=("$code")
      register_path_intent "$code" "$rest"
      ((count_other += 1))
      track_dir "$rest"
      if ((i < max_items)); then
        file_body_lines+=("- 更新 ${rest}")
      fi
      ;;
  esac
done

for ((i = 0; i < ${#changed_paths[@]}; i++)); do
  summarize_file_patch "${changed_codes[i]}" "${changed_paths[i]}"
done

if [[ -n "$custom_subject" ]]; then
  subject="$custom_subject"
else
  subject="$(detect_subject_scope) $(detect_subject_action)"
fi

while IFS= read -r diff_line; do
  if [[ "$diff_line" == "+++ b/"* ]]; then
    continue
  fi
  if [[ "$diff_line" != +* || "$diff_line" == "+++"* ]]; then
    continue
  fi

  added_line="${diff_line#?}"
  added_line="$(echo "$added_line" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
  [[ -z "$added_line" ]] && continue

  if [[ "$added_line" == \#include* ]]; then
    header="$(echo "$added_line" | sed -nE 's/^#include[[:space:]]*[<"]([^">]+)[">].*$/\1/p')"
    if [[ -n "$header" ]]; then
      record_header "$header"
      continue
    fi
  fi

  if [[ "$added_line" =~ ^#define[[:space:]]+([A-Za-z_][A-Za-z0-9_]*) ]]; then
    macro_name="${BASH_REMATCH[1]}"
    record_symbol "macro" "$macro_name"
    continue
  fi

  if [[ "$added_line" =~ ^(constexpr|const|static[[:space:]]+const).*[[:space:]]([A-Z][A-Z0-9_]+)[[:space:]]*= ]]; then
    const_name="${BASH_REMATCH[2]}"
    record_symbol "constant" "$const_name"
    continue
  fi

  if [[ "$added_line" =~ __global__[[:space:]]+void[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*\( ]]; then
    kernel_name="${BASH_REMATCH[1]}"
    has_cuda_kernel=1
    record_symbol "kernel" "$kernel_name"
    continue
  fi

  if [[ "$added_line" =~ ^def[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*\( ]]; then
    func_name="${BASH_REMATCH[1]}"
    record_symbol "function" "$func_name"
    continue
  fi

  if [[ "$added_line" =~ ^class[[:space:]]+([A-Za-z_][A-Za-z0-9_]*) ]]; then
    class_name="${BASH_REMATCH[1]}"
    record_symbol "class" "$class_name"
    continue
  fi
done < <(git diff --cached --no-color -U0)

emit_collected_semantics

stat_line="- 文件统计：新增 ${count_add}，更新 $((count_update + count_other))，删除 ${count_remove}，重命名 ${count_rename}，复制 ${count_copy}"

dir_line=""
if (( ${#dir_order[@]} > 0 )); then
  preview_limit=8
  preview_dirs=()
  for ((i = 0; i < ${#dir_order[@]} && i < preview_limit; i++)); do
    preview_dirs+=("${dir_order[i]}")
  done
  dir_line="- 涉及目录：$(IFS='、'; echo "${preview_dirs[*]}")"
fi

if ((total_items > max_items)); then
  remaining=$((total_items - max_items))
  file_body_lines+=("- 其余 ${remaining} 个文件已省略逐条列表，请参考文件统计与目录摘要")
fi

combined_semantic_lines=("${semantic_lines[@]}")
if (( ${#support_lines[@]} > 0 )); then
  combined_semantic_lines+=("${support_lines[@]}")
fi

commit_msg_file=""
cleanup() {
  if [[ -n "$commit_msg_file" && -f "$commit_msg_file" ]]; then
    rm -f "$commit_msg_file"
  fi
}
trap cleanup EXIT

commit_msg_file="$(mktemp)"
{
  echo "$subject"
  echo
  if (( ${#combined_semantic_lines[@]} > 0 )); then
    echo "$(build_heading)"
    echo
    semantic_total="${#combined_semantic_lines[@]}"
    semantic_show_count="$semantic_total"
    if ((semantic_show_count > semantic_limit)); then
      semantic_show_count="$semantic_limit"
    fi
    for ((i = 0; i < semantic_show_count; i++)); do
      echo "${combined_semantic_lines[i]}"
    done
    if ((semantic_total > semantic_limit)); then
      echo "- 其余 $((semantic_total - semantic_limit)) 条语义摘要已省略"
    fi
    echo "$stat_line"
    if [[ -n "$dir_line" ]]; then
      echo "$dir_line"
    fi
  elif (( ${#fallback_semantic_lines[@]} > 0 )); then
    echo "变更内容摘要"
    echo
    fallback_total="${#fallback_semantic_lines[@]}"
    fallback_show_count="$fallback_total"
    if ((fallback_show_count > semantic_limit)); then
      fallback_show_count="$semantic_limit"
    fi
    for ((i = 0; i < fallback_show_count; i++)); do
      echo "${fallback_semantic_lines[i]}"
    done
    if ((fallback_total > semantic_limit)); then
      echo "- 其余 $((fallback_total - semantic_limit)) 条摘要已省略"
    fi
    echo "$stat_line"
    if [[ -n "$dir_line" ]]; then
      echo "$dir_line"
    fi
  else
    if (( ${#file_body_lines[@]} == 0 )); then
      echo "- 更新项目文件"
    else
      printf '%s\n' "${file_body_lines[@]}"
    fi
    echo "$stat_line"
    if [[ -n "$dir_line" ]]; then
      echo "$dir_line"
    fi
  fi
} > "$commit_msg_file"

echo "[gitc] Running: git commit"
git commit -F "$commit_msg_file"

echo "[gitc] Running: git push"
if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
  git push
else
  if git remote get-url origin >/dev/null 2>&1; then
    git push --set-upstream origin "$branch"
  else
    echo "[gitc] Error: upstream is not configured and remote 'origin' is missing."
    echo "[gitc] Commit created locally; push was not completed."
    exit 1
  fi
fi

echo "[gitc] Done: add -> commit -> push on branch '${branch}'."
