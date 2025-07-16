#!/bin/bash

cd "$(dirname "$0")"  # Navigate to script's directory

git pull origin main      # ✅ Pull changes first to avoid conflict
git add .
git commit -m "${1:-Auto update}"
git push origin main