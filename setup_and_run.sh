#!/bin/bash

# 环境名称
ENV_NAME="gptq"

echo "SAGE project setup and run script"
echo "========================"

# 检查 Conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "Error: Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# 进入脚本所在目录
cd "$(dirname "$0")" || exit 1

# 初始化conda
eval "$(conda shell.bash hook)"

# 检查并创建环境
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating Conda environment..."
    conda env create -f environment.yml
    if [ $? -ne 0 ]; then
        echo "Error: Environment creation failed"
        exit 1
    fi
else
    echo "Environment exists. Installing packages with exact versions from yml..."
    conda run -n "$ENV_NAME" pip install \
        torch==2.4.1 \
        transformers==4.53.2 \
        pandas==2.2.2 \
        numpy==1.26.4 \
        scikit-learn==1.4.2 \
        tqdm==4.66.4 \
        peft==0.11.1 \
        python-levenshtein==0.27.1 \
        accelerate==1.8.1 \
        datasets==4.0.0 \
        tokenizers==0.21.2 \
        packaging==25.0 \
        huggingface-hub==0.33.4 \
        regex==2024.4.28 \
        requests==2.32.4 \
        pyyaml==6.0.2 \
        safetensors==0.5.3
fi

# 运行程序
echo "Starting program..."
conda run -n "$ENV_NAME" python run_sage.py

echo "Done"
    