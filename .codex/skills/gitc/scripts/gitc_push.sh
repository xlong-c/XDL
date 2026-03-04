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
primary_path=""
has_cuda_kernel=0

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
      ((count_add += 1))
      track_dir "$rest"
      if ((i < max_items)); then
        file_body_lines+=("- 新增 ${rest}")
      fi
      ;;
    M)
      [[ -z "$primary_path" ]] && primary_path="$rest"
      ((count_update += 1))
      track_dir "$rest"
      if ((i < max_items)); then
        file_body_lines+=("- 更新 ${rest}")
      fi
      ;;
    D)
      [[ -z "$primary_path" ]] && primary_path="$rest"
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
      ((count_other += 1))
      track_dir "$rest"
      if ((i < max_items)); then
        file_body_lines+=("- 更新 ${rest}")
      fi
      ;;
  esac
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
