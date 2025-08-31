#!/bin/bash

# 项目目录名
PROJECT_DIR="SAGE"
# 环境名称
ENV_NAME="gptq"

# 检查 Conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到 Conda，请先安装 Miniconda 或 Anaconda"
    exit 1
fi


# 进入项目目录
cd "$PROJECT_DIR" || exit 1

# 检查环境是否存在，不存在则创建
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "创建 Conda 环境..."
    conda env create -f environment.yml
    if [ $? -ne 0 ]; then
        echo "错误: 创建环境失败"
        exit 1
    fi
else
    echo "环境已存在，更新环境..."
    conda env update -f environment.yml --prune
fi

# 运行程序
echo "启动程序..."
conda run -n "$ENV_NAME" python run_sage.py
    