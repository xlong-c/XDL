#!/usr/bin/env bash
set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[gitc] Error: current directory is not a git repository."
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

branch="$(git rev-parse --abbrev-ref HEAD)"
subject="${1:-chore: 同步工作区更新}"
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
declare -A fallback_seen=()
primary_path=""
has_cuda_kernel=0
changed_paths=()
changed_codes=()

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
      add_semantic_line "添加头文件 (${header})"
      continue
    fi
  fi

  if [[ "$added_line" =~ ^#define[[:space:]]+([A-Za-z_][A-Za-z0-9_]*) ]]; then
    macro_name="${BASH_REMATCH[1]}"
    add_semantic_line "定义宏 (${macro_name})"
    continue
  fi

  if [[ "$added_line" =~ ^(constexpr|const|static[[:space:]]+const).*[[:space:]]([A-Z][A-Z0-9_]+)[[:space:]]*= ]]; then
    const_name="${BASH_REMATCH[2]}"
    add_semantic_line "定义常量 (${const_name})"
    continue
  fi

  if [[ "$added_line" =~ __global__[[:space:]]+void[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*\( ]]; then
    kernel_name="${BASH_REMATCH[1]}"
    has_cuda_kernel=1
    if [[ "$kernel_name" == *naive* ]]; then
      add_semantic_line "实现 naive CUDA 内核 (${kernel_name})"
    else
      add_semantic_line "实现 CUDA 内核 (${kernel_name})"
    fi
    continue
  fi

  if [[ "$added_line" =~ ^def[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*\( ]]; then
    func_name="${BASH_REMATCH[1]}"
    add_semantic_line "新增函数 (${func_name})"
    continue
  fi

  if [[ "$added_line" =~ ^class[[:space:]]+([A-Za-z_][A-Za-z0-9_]*) ]]; then
    class_name="${BASH_REMATCH[1]}"
    add_semantic_line "新增类 (${class_name})"
    continue
  fi
done < <(git diff --cached --no-color -U0)

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
  if (( ${#semantic_lines[@]} > 0 )); then
    echo "$(build_heading)"
    echo
    semantic_total="${#semantic_lines[@]}"
    semantic_show_count="$semantic_total"
    if ((semantic_show_count > semantic_limit)); then
      semantic_show_count="$semantic_limit"
    fi
    for ((i = 0; i < semantic_show_count; i++)); do
      echo "${semantic_lines[i]}"
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
