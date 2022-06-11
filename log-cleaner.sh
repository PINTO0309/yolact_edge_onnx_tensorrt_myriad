#!/bin/bash
TARGETS=(
  "*.whl"
  "*.tar.gz"
  "*.deb"
  "*.zip"
  "*.xml"
  "*.bin"
  "*.mapping"
  "*.engine"
  "*.profile"
)

target=$(printf " %s" "${TARGETS[@]}")
target=${target:1}

sudo rm -rf .git/refs/original
git filter-branch --index-filter "git rm -r --cached --ignore-unmatch ${target}" -- --all

