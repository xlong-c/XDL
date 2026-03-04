#!/usr/bin/env bash
set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[gitc] Error: current directory is not a git repository."
  exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

branch="$(git rev-parse --abbrev-ref HEAD)"
subject="${1:-chore: sync workspace updates}"
max_items=8

echo "[gitc] Running: git add -A"
git add -A

if git diff --cached --quiet; then
  echo "[gitc] No changes staged after git add -A. Nothing to commit or push."
  exit 0
fi

mapfile -t status_lines < <(git diff --cached --name-status)
body_lines=()
total_items="${#status_lines[@]}"

for ((i = 0; i < total_items && i < max_items; i++)); do
  line="${status_lines[i]}"
  [[ -z "$line" ]] && continue

  code="${line%%$'\t'*}"
  rest="${line#*$'\t'}"

  case "$code" in
    A)
      body_lines+=("- add ${rest}")
      ;;
    M)
      body_lines+=("- update ${rest}")
      ;;
    D)
      body_lines+=("- remove ${rest}")
      ;;
    R*|C*)
      src="${rest%%$'\t'*}"
      dst="${rest#*$'\t'}"
      if [[ "$code" == R* ]]; then
        body_lines+=("- rename ${src} -> ${dst}")
      else
        body_lines+=("- copy ${src} -> ${dst}")
      fi
      ;;
    *)
      body_lines+=("- update ${rest}")
      ;;
  esac
done

if ((total_items > max_items)); then
  body_lines+=("- update remaining $((total_items - max_items)) files")
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
  if (( ${#body_lines[@]} == 0 )); then
    echo "- update project files"
  else
    printf '%s\n' "${body_lines[@]}"
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
