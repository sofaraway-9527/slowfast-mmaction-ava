#!/bin/bash

IN_DATA_DIR="./videos"
OUT_DATA_DIR="./video_crop"

# 检查 ffmpeg 是否安装
if ! command -v ffmpeg &> /dev/null; then
  echo "ffmpeg could not be found. Please install ffmpeg to proceed."
  exit 1
fi

# 检查并创建输出目录（如果不存在）
if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it."
  mkdir -p "${OUT_DATA_DIR}"
fi

# 遍历输入目录中的每个视频文件
find "${IN_DATA_DIR}" -type f | while read -r video; do
  out_name="${OUT_DATA_DIR}/$(basename "${video}")"
  if [[ ! -f "${out_name}" ]]; then
    echo "Processing ${video}..."
    ffmpeg -ss 0 -t 46 -i "${video}" -c copy "${out_name}"
  else
    echo "${out_name} already exists. Skipping."
  fi
done

echo "Processing complete."
