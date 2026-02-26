#!/usr/bin/env bash

set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "错误：当前目录不是 Git 仓库。"
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "错误：未找到远程仓库 origin，请先配置 GitHub 远程地址。"
  exit 1
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ -z "$branch" ]]; then
  echo "错误：无法识别当前分支。"
  exit 1
fi

commit_msg="${1:-}"
if [[ -z "$commit_msg" ]]; then
  read -r -p "请输入 commit 信息: " commit_msg
fi

if [[ -z "$commit_msg" ]]; then
  echo "错误：commit 信息不能为空。"
  exit 1
fi

git add -A

if git diff --cached --quiet; then
  echo "没有检测到可提交的改动，跳过 commit。"
else
  git commit -m "$commit_msg"
fi

git push origin "$branch"
echo "上传完成：origin/$branch"
