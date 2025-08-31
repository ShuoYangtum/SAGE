import random
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from torch.optim import AdamW
from tqdm.notebook import tqdm
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
    target_column="" # 这里是他们要补的那个列名
    train_pth="xxx.csv"
    test_pth='yyy.csv'
    device=torch.device("cuda:1")
    
    generator = SAGE(model_name="gpt2", device=device) #Qwen/Qwen3-0.6B-Base meta-llama/Llama-3.2-1B
    mi_threshold=0.0001 
    a=time.time()
    generator.fit(train_pth, max_sample_num=None,\
                    batch_size=32, epochs=100, lr=1e-4, max_length=50, shuffle=True,\
                    early_stopping_rounds=5, mi_threshold=mi_threshold, mi_n_bins=5,\
                    constrain_string_values=True, val_ratio=0.001, num_workers=8, gradient_accumulation_steps=1, drop=[])

    generator.model.load_state_dict(torch.load("best_generator_model.pt")) 

    imputation(self, test_pth, target_column, \
                       max_new_tokens_per_value=20, temperature=1.0, \
                       mi_threshold=0.0001, apply_final_constraints=True,\
                       save_path = '../output.csv')
    
if __name__=='__main__':
    main()