import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from torch.optim import AdamW
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
import numpy as np
from motf_datasets import FeatureTextDataset
from Selector import NonParametricMISelector, MiLogitsBiasProcessor
from generation import NoLeadingCommaLogitsProcessor, AllowedTokensLogitsProcessor
from utils import clean_csv_in_place
import re
import Levenshtein
from loss import FocalLoss
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
from sage import SAGE


def main():
    target_column=""
    train_pth="train.csv"
    test_pth='test.csv'
    device=torch.device("cuda:0")
    
    generator = SAGE(model_name="gpt2", device=device, use_lora=False) #Qwen/Qwen3-0.6B-Base meta-llama/Llama-3.2-1B
    mi_threshold=0.0001 
    a=time.time()
    generator.fit(train_pth, max_sample_num=None,\
                    batch_size=32, epochs=100, lr=1e-4, max_length=50, shuffle=True,\
                    early_stopping_rounds=5, mi_threshold=mi_threshold, mi_n_bins=5,\
                    constrain_string_values=True, val_ratio=0.001, num_workers=8, gradient_accumulation_steps=1, drop=[])

    # generator.model.load_state_dict(torch.load("best_generator_model.pt")) 

    result = generator.imputation(test_pth, target_column, \
                       max_new_tokens_per_value=20, temperature=1.0, \
                       mi_threshold=0.0001, apply_final_constraints=True,\
                       save_path = '../output.csv')

    # Print training time and results
    training_time = time.time() - a
    print(f"Training time: {training_time:.2f} seconds")
    
    # Print evaluation results
    if result['task_type'] == 'regression':
        print(f"MSE: {result.get('mse', 'N/A')}")
        print(f"MAE: {result.get('mae', 'N/A')}")
        print(f"RMSE: {result.get('rmse', 'N/A')}")
    else:
        print(f"Accuracy: {result.get('accuracy', 'N/A')}")
        print(f"Error Rate: {result.get('error_rate', 'N/A')}")

if __name__=='__main__':
    main()