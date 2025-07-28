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

class SAGE:
    def __init__(self, model_name="gpt2", device=None, use_lora=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["k_proj","v_proj"],
                lora_dropout=0.1,
                bias="all",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(base_model, lora_config)
        else:
            self.model=base_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.selector = NonParametricMISelector()
        self.cross_entropy_loss = FocalLoss(gamma=2.0, ignore_index=-100)
        self.numerical_features = set() 
        self.categorical_features = set() 
        self.valid_value_token_ids = {} 
        self.max_tokens_for_value = {}
        self.discretizers = None
        self.constraints = None
        self.feature_columns = None
        self.numerical_min_diffs = {} 

    def _discretize_dataframe(self, df, n_bins=10, strategy='quantile'):
        df_discretized = df.copy()
        discretizers = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                non_nan_values = df[col].dropna().values.reshape(-1, 1)
                if len(non_nan_values) > 0:
                    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy, subsample=None)
                    discretizer.fit(non_nan_values)
                    
                    df_discretized[col] = discretizer.transform(df[col].values.reshape(-1, 1)).astype(int).flatten()
                    discretizers[col] = discretizer
            else:
                df_discretized[col] = df_discretized[col].astype('category')
        return df_discretized, discretizers

    def _calculate_all_mutual_info(self, df_original, df_discretized, feature_columns):
        mi_table = {}
        for p_feat_name in tqdm(feature_columns, desc="Calculating MI for (CurrentFeature, PastFeature_Value) pairs"):
            past_values_discrete = df_discretized[p_feat_name].dropna().unique()
            for p_val_discrete in past_values_discrete:
                mask = (df_discretized[p_feat_name] == p_val_discrete)
                for c_feat_name in feature_columns:
                    if c_feat_name == p_feat_name:
                        continue 

                    temp_df = df_discretized[[c_feat_name, p_feat_name]].dropna()
                    
                    temp_mask = (temp_df[p_feat_name] == p_val_discrete)
                    if not temp_df.empty and temp_mask.any():
                        X_binary = temp_mask.astype(int).values.reshape(-1, 1)
                        y_target = temp_df[c_feat_name].values
                        try:
                            mi_score = mutual_info_classif(X_binary, y_target, discrete_features=[True])[0]
                            key = (c_feat_name, p_feat_name, p_val_discrete)
                            mi_table[key] = mi_score
                        except ValueError as e:
                            key = (c_feat_name, p_feat_name, p_val_discrete)
                            mi_table[key] = 0.0

        print("\n--- Mutual Information Scores Distribution (Feature-Value Pairs) ---")
        all_mi_scores = list(mi_table.values())
        if all_mi_scores:
            print(f"Min MI: {np.min(all_mi_scores):.4f}")
            print(f"Max MI: {np.max(all_mi_scores):.4f}")
            print(f"Mean MI: {np.mean(all_mi_scores):.4f}")
            print(f"Median MI: {np.median(all_mi_scores):.4f}")
            print(f"25th percentile MI: {np.percentile(all_mi_scores, 25):.4f}")
            print(f"75th percentile MI: {np.percentile(all_mi_scores, 75):.4f}")
            print(f"Number of MI scores > 0: {np.sum(np.array(all_mi_scores) > 0)}")
            print(f"Total MI scores calculated: {len(all_mi_scores)}")
        else:
            print("No MI scores calculated (possibly very small dataset or all NaN).")
        print("--------------------------------------------")
        return mi_table

    def fit(self, data, epochs=10, batch_size=8, lr=1e-3, max_length=256,
            val_ratio=0.05, early_stopping_rounds=5, mi_n_bins=10,
            mi_strategy='quantile', mi_threshold=0.01, constrain_string_values=False,
            num_workers=4, gradient_accumulation_steps=4, max_sample_num=None, shuffle=False, drop=None):
        df = pd.read_csv(data)
        df = df.fillna(-100)
        #df=df.dropna()
        if drop:
            for col in drop:
                df = df.drop(col, axis=1)
        if max_sample_num:
            df=df[:max_sample_num]
        self.df=df
        self.feature_columns = [col for col in df.columns]
        # Pass the original df to analyze_constraints to get min diffs
        self.constraints = self.analyze_constraints(df)
        self.set_feature_columns(df.columns)
        self.constrain_string_values=constrain_string_values
        if constrain_string_values:
            self.constrain_string_values=constrain_string_values
            self.numerical_features.clear()
            self.categorical_features.clear()
            self.valid_value_token_ids.clear()
            self.max_tokens_for_value.clear() 
            print("Analyzing feature types and collecting valid token IDs...")
            for col in tqdm(self.feature_columns, desc="Processing features for tokenization"):

                allowed_chars_for_number = set("0123456789.,-")

                if pd.api.types.is_numeric_dtype(df[col]):
                    self.numerical_features.add(col)

                    non_null_str_vals = df[col].dropna().astype(str)

                    max_len_str = non_null_str_vals.apply(len).max() if not non_null_str_vals.empty else 5
                    self.max_tokens_for_value[col] = max(2, int(max_len_str * 1.5))

                    allowed_ids = set()
                    for char in allowed_chars_for_number:
                        token_ids = self.tokenizer.encode(char, add_special_tokens=False)
                        allowed_ids.update(token_ids)

                    self.valid_value_token_ids[col] = allowed_ids
                else: 
                    self.categorical_features.add(col)

                    unique_values = df[col].dropna().astype(str).unique()
                    allowed_ids_for_feat = set()
                    max_tokens_for_current_feat = 1 

                    for val in unique_values:
                        token_ids = self.tokenizer.encode(val, add_special_tokens=False)
                        if token_ids: 
                            allowed_ids_for_feat.update(token_ids)
                            max_tokens_for_current_feat = max(max_tokens_for_current_feat, len(token_ids))
                        else:
                            print(f"Warning: Value '{val}' for feature '{col}' could not be tokenized or resulted in empty token IDs. Skipping for constraint.")
                    if not allowed_ids_for_feat:
                        print(f"Warning: No valid token IDs found for feature '{col}'. It will not have generation constraints during sampling.")
                    self.valid_value_token_ids[col] = allowed_ids_for_feat
                  
                    self.max_tokens_for_value[col] = max(max_tokens_for_current_feat, 2)
            print("Feature analysis complete.")
       
        df_discretized, self.discretizers = self._discretize_dataframe(df, n_bins=mi_n_bins, strategy=mi_strategy)
     
        mi_table = self._calculate_all_mutual_info(df, df_discretized, self.feature_columns)
       
        self.selector.set_mi_data(mi_table, self.feature_columns, self.discretizers)

        train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=42)
        train_dataset = FeatureTextDataset(train_df, self.tokenizer, max_length=max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        optimizer = AdamW(self.model.parameters(), lr=lr)
        best_val_loss = float('inf')
        no_improve_epochs = 0
        for epoch in range(epochs):
            self.model.train()
            total_train_loss_generator = 0
            if shuffle:
                train_dataset = FeatureTextDataset(train_df, self.tokenizer, max_length=max_length)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} (Generator Training)")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                #outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                
                # Get logits (shape: [batch_size, seq_len, vocab_size])
                logits = outputs.logits
                
                # Shift logits and labels for decoder-style models (e.g. GPT, T5)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()  # align with next token prediction
                
                # Optionally: flatten inputs for loss calculation
                generator_loss_batch = self.cross_entropy_loss(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                #generator_loss_batch = outputs.loss
                total_train_loss_generator += generator_loss_batch.item()
                generator_loss_batch.backward()
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            if (len(train_loader) % gradient_accumulation_steps != 0):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            avg_train_loss_generator = total_train_loss_generator / len(train_loader)
         
            self.model.eval()
            total_val_generator_loss = 0
            valid_val_samples = 0
            with torch.no_grad():
                for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"Epoch {epoch+1} (Validation)"):
                    pairs = []
                    for col in self.feature_columns:
                        pairs.append((col, str(row[col])))
                    random.shuffle(pairs)
                    current_history_fv = [] 
                    for i, (current_feat, current_val) in enumerate(pairs):
                        if i > 0:
                            
                            mi_scores = self.selector(current_feat, current_history_fv)
                            selected_past_fv_texts = []
                            if mi_scores.numel() > 0:
                                relevant_indices = (mi_scores >= mi_threshold).nonzero(as_tuple=True)[0]
                                for idx in relevant_indices:
                                    selected_feat, selected_val = current_history_fv[idx.item()]
                                    selected_past_fv_texts.append(f"{selected_feat} is {selected_val}")
                            generator_prompt_base = f"{current_feat} is "
                            if selected_past_fv_texts:
                                full_generator_prompt_for_lm = ", ".join(selected_past_fv_texts) + ", " + generator_prompt_base
                            else:
                                full_generator_prompt_for_lm = generator_prompt_base
                            input_text_ids = self.tokenizer.encode(full_generator_prompt_for_lm, add_special_tokens=False)
                            true_val_str = str(current_val)
                            true_val_ids = self.tokenizer.encode(true_val_str, add_special_tokens=False)
                            labels_lm = [-100] * len(input_text_ids) + true_val_ids
                            input_ids_lm = input_text_ids + true_val_ids
                            input_ids_lm = input_ids_lm[:max_length]
                            labels_lm = labels_lm[:max_length]
                            attention_mask_lm = [1] * len(input_ids_lm)
                            pad_len_lm = max_length - len(input_ids_lm)
                            input_ids_lm += [self.tokenizer.pad_token_id] * pad_len_lm
                            labels_lm += [-100] * pad_len_lm
                            attention_mask_lm += [0] * pad_len_lm
                            input_ids_lm = torch.tensor(input_ids_lm).unsqueeze(0).to(self.device)
                            attention_mask_lm = torch.tensor(attention_mask_lm).unsqueeze(0).to(self.device)
                            labels_lm = torch.tensor(labels_lm).unsqueeze(0).to(self.device)
                            val_outputs = self.model(input_ids_lm, attention_mask=attention_mask_lm, labels=labels_lm)
                            total_val_generator_loss += val_outputs.loss.item()
                            valid_val_samples += 1
                        current_history_fv.append((current_feat, current_val))
                avg_val_loss = total_val_generator_loss / (valid_val_samples if valid_val_samples > 0 else 1)
            print(f"Epoch {epoch + 1}: Generator_train_loss = {avg_train_loss_generator:.4f}, Val_gen_loss = {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve_epochs = 0
                torch.save(self.model.state_dict(), "best_generator_model.pt")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= early_stopping_rounds:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        self.model.load_state_dict(torch.load("best_generator_model.pt"))

    def _cosine_annealing(self, current_step, total_steps, start_value, end_value):
        if current_step > 5:
            return end_value
        if total_steps == 0: 
            return start_value

        progress = max(0.0, min(1.0, current_step / total_steps))
    
        annealed_value = end_value + 0.5 * (start_value - end_value) * (1 + np.cos(np.pi * progress))
        return annealed_value


    def sample(self, sample_num=1, feature_order=None, p=0.5, max_new_tokens_per_value=20, temperature=1.0, mi_threshold=0.01,
               copy_factor=3, copy_prob_start=1.0, copy_prob_end=0.0, apply_final_constraints=True):
        self.model.eval()
        self.selector.eval()
        constrain_string_values=self.constrain_string_values 
        samples = []
        generated_full_samples_set = set()
        current_incomplete_samples = []
        for _ in range(sample_num//copy_factor):
            if feature_order is None:
                initial_order = self.feature_columns.copy()
                random.shuffle(initial_order)
            else:
                initial_order = feature_order
            num_predefined = int(p * len(initial_order))
            predefined_feats = initial_order[:num_predefined]
            random_row = self.df.sample(1).iloc[0]
            predefined_value_map = {feat: str(random_row[feat]) for feat in predefined_feats}
            predefined_pairs = [(feat, str(random_row[feat])) for feat in predefined_feats]
            current_incomplete_samples.append({
                'feature_value_map': predefined_value_map,
                'past_feature_value_pairs': predefined_pairs,
                'current_feature_order': initial_order
            })
        num_features = len(initial_order)
        while len(generated_full_samples_set) < sample_num and current_incomplete_samples:
            temp_samples_to_process = current_incomplete_samples
            current_incomplete_samples = []
            if not temp_samples_to_process:
                new_initial_order = self.feature_columns.copy()
                random.shuffle(new_initial_order)
                current_incomplete_samples.append({
                    'feature_value_map': {},
                    'past_feature_value_pairs': [],
                    'current_feature_order': new_initial_order})
                continue
            for sample_dict in temp_samples_to_process:
                feature_value_map = sample_dict['feature_value_map']
                past_feature_value_pairs = sample_dict['past_feature_value_pairs']
                current_feature_order = sample_dict['current_feature_order']
                feat_idx = len(past_feature_value_pairs)
                if feat_idx >= num_features:
                    ordered_tuple = tuple(sorted(feature_value_map.items()))
                    if ordered_tuple not in generated_full_samples_set:
                        generated_full_samples_set.add(ordered_tuple)
                        if len(generated_full_samples_set) >= sample_num: break
                    continue
                current_feat = current_feature_order[feat_idx]
                # --- 构建 Prompt ---
                filtered_prefix_parts = []
                if past_feature_value_pairs:
                    mi_scores = self.selector(current_feat, past_feature_value_pairs)
                    if mi_scores.numel() > 0:
                        relevant_indices = (mi_scores >= mi_threshold).nonzero(as_tuple=True)[0]
                        for idx in relevant_indices:
                            selected_feat, selected_val = past_feature_value_pairs[idx.item()]
                            filtered_prefix_parts.append(f"{selected_feat} is {selected_val}")
                current_generator_prompt_base = f"{current_feat} is"
                if filtered_prefix_parts:
                    current_prompt_for_generator = ", ".join(filtered_prefix_parts) + ", " + current_generator_prompt_base
                else:
                    current_prompt_for_generator = current_generator_prompt_base
                input_ids = self.tokenizer(current_prompt_for_generator, return_tensors='pt').input_ids.to(self.device)
                logits_processor_list = LogitsProcessorList()
                effective_max_new_tokens = self.max_tokens_for_value.get(current_feat, max_new_tokens_per_value)
 
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=effective_max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=30,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    logits_processor=logits_processor_list if logits_processor_list else None )
                full_generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                new_segment = full_generated_text[len(current_prompt_for_generator):]
                generated_value = new_segment.split(",")[0].strip()

                new_feature_value_map = feature_value_map.copy()
                new_feature_value_map[current_feat] = generated_value
                new_past_feature_value_pairs = past_feature_value_pairs + [(current_feat, generated_value)]
                current_incomplete_samples.append({
                    'feature_value_map': new_feature_value_map,
                    'past_feature_value_pairs': new_past_feature_value_pairs,
                    'current_feature_order': current_feature_order})
                if copy_factor<=1:
                    if num_features > 1:
                        annealed_copy_prob = self._cosine_annealing(
                            current_step=feat_idx,
                            total_steps=num_features - 1,
                            start_value=copy_prob_start,
                            end_value=copy_prob_end)
                    else:
                        annealed_copy_prob = copy_prob_start
                    if random.random() < annealed_copy_prob:
                        for _ in range(copy_factor - 1):
                            current_incomplete_samples.append({
                                'feature_value_map': new_feature_value_map.copy(),
                                'past_feature_value_pairs': new_past_feature_value_pairs.copy(),
                                'current_feature_order': current_feature_order.copy()})
                    if len(generated_full_samples_set) >= sample_num: break
            if len(generated_full_samples_set) >= sample_num: break
            if not current_incomplete_samples and len(generated_full_samples_set) < sample_num:
                if feature_order is None:
                    initial_order = self.feature_columns.copy()
                    random.shuffle(initial_order)
                else:
                    initial_order = feature_order
                num_predefined = int(p * len(initial_order))
                predefined_feats = initial_order[:num_predefined]
                random_row = self.df.sample(1).iloc[0]
                predefined_value_map = {feat: str(random_row[feat]) for feat in predefined_feats}
                predefined_pairs = [(feat, str(random_row[feat])) for feat in predefined_feats]
                current_incomplete_samples.append({
                    'feature_value_map': predefined_value_map,
                    'past_feature_value_pairs': predefined_pairs,
                    'current_feature_order': initial_order
                })
        for s_tuple in generated_full_samples_set:
            temp_map = dict(s_tuple)
            filtered_map = temp_map 
            restored = {feat: filtered_map.get(feat, "") for feat in self.feature_columns}
            samples.append(restored)
        if len(samples) > sample_num:
            samples = random.sample(samples, sample_num)
        tmp_df=pd.DataFrame(samples)
        dict_list = tmp_df.to_dict(orient='records')
        output=[generator.apply_constraints(dict_list[i]) for i in range(sample_num)]
        return pd.DataFrame(output)

    
    def analyze_constraints(self, df, numeric_threshold=30):
        constraints = {}
        self.numerical_min_diffs.clear() # Clear previous data
        for col in df.columns:
            unique_values = df[col].dropna().unique()
            if pd.api.types.is_numeric_dtype(df[col]):
                constraints[col] = {
                    "type": "numeric",
                    "min": float(df[col].min()),
                    "max": float(df[col].max())
                }
                # Calculate minimum difference for numerical columns
                if len(unique_values) > 1:
                    sorted_unique = np.sort(unique_values)
                    # Calculate differences between consecutive sorted unique values
                    diffs = np.diff(sorted_unique)
                    # Filter out zero differences (if any, though unique() should handle it)
                    non_zero_diffs = diffs[diffs > 1e-9] # Use a small tolerance for floating point
                    if len(non_zero_diffs) > 0:
                        self.numerical_min_diffs[col] = np.round(np.min(non_zero_diffs), 10)
                    else:
                        self.numerical_min_diffs[col] = None # No meaningful difference (e.g., all same value)
                else:
                    self.numerical_min_diffs[col] = None # Only one unique value or empty
            else:
                if len(unique_values) <= numeric_threshold:
                    constraints[col] = {
                        "type": "categorical",
                        "values": set(str(v) for v in unique_values)
                    }
                else:
                    '''
                    constraints[col] = {
                        "type": "categorical"  # text
                    }
                    '''
                    constraints[col] = {
                        "type": "categorical",
                        "values": set(str(v) for v in unique_values)
                    }
        return constraints

    def set_feature_columns(self, columns):
        self.feature_columns = list(columns)

    def _levenshtein_distance(self, s1, s2):
        # Using Levenshtein from the library directly as imported
        return Levenshtein.distance(s1, s2)

    def apply_constraints(self, generated_data):
        corrected_data = {}
        for feature, gen_value_str in generated_data.items():
            constraint = self.constraints.get(feature)
            if not constraint:
                corrected_data[feature] = gen_value_str # No constraint, keep as is
                continue
            try:
                if constraint["type"] == "numeric":
                    # Try to convert to float
                    gen_value_str=gen_value_str.strip(' ')
                    if gen_value_str and gen_value_str[-1] in ['-', '.']:
                        gen_value_str = gen_value_str[:-1]
                    gen_value_str = re.sub(r'[^\d.-]', '', gen_value_str)
                    gen_value = float(gen_value_str)
                    
                    min_val = constraint.get("min")
                    max_val = constraint.get("max")
                    min_diff = generator.numerical_min_diffs.get(feature)
 
                    if min_val is not None:
                        gen_value = max(gen_value, min_val)
                    if max_val is not None:
                        gen_value = min(gen_value, max_val)
                    
                    if min_val is not None and max_val is not None and min_diff is not None and min_diff > 0:

                        steps = round((gen_value - min_val) / min_diff)
                        nearest_grid_value = min_val + steps * min_diff
                    

                        nearest_grid_value = min(max(nearest_grid_value, min_val), max_val)
                        gen_value = nearest_grid_value

                    if gen_value.is_integer():
                        gen_value = int(gen_value)
                    gen_value=np.round(gen_value, 10)
                    corrected_data[feature] = str(gen_value) # Keep as float/int
                elif constraint["type"] == "categorical":
                    # For categorical features, find the closest valid value using Levenshtein distance
                    valid_values = constraint["values"]
                    if gen_value_str in valid_values:
                        corrected_data[feature] = gen_value_str
                    else:
                        closest_value = None
                        min_distance = float('inf')
                        
                        # Handle potential empty string generation
                        if not gen_value_str:
                            # If an empty string is generated for a categorical feature,
                            # it might be better to assign a common value or NaN,
                            # or just skip. For now, we'll try to find the closest.
                            # If valid_values is empty, this loop won't run.
                            if valid_values:
                                corrected_data[feature] = random.choice(list(valid_values))
                            else:
                                corrected_data[feature] = "" # No valid values to choose from
                            continue

                        for valid_val in valid_values:
                            dist = self._levenshtein_distance(gen_value_str, valid_val)
                            if dist < min_distance:
                                min_distance = dist
                                closest_value = valid_val
                            # If distance is 0, we found an exact match, no need to check further
                            if min_distance == 0:
                                break
                        corrected_data[feature] = closest_value if closest_value is not None else gen_value_str
                
                elif constraint["type"] == "text":
                    corrected_data[feature] = gen_value_str.strip()
            except ValueError:
                # If conversion to numeric fails, or other parsing errors
                if constraint["type"] == "numeric":
                    min_val = constraint.get("min")
                    max_val = constraint.get("max")
                    min_diff = self.numerical_min_diffs.get(feature)
                
                    if min_val is not None and min_diff is not None and min_diff > 0:
         
                        if min_val < max_val:
                            steps = round((min_val - min_val) / min_diff)  
                            aligned_min_val = min_val + steps * min_diff
               
                            aligned_min_val = min(max(aligned_min_val, min_val), max_val)
                        else:
                            aligned_min_val = min_val
                
                       
                        if min_diff.is_integer() and isinstance(aligned_min_val, float) and aligned_min_val.is_integer():
                            aligned_min_val = int(aligned_min_val)
                        else:
                            s_min_diff = str(min_diff)
                            if '.' in s_min_diff:
                                decimal_places = len(s_min_diff.split('.')[1])
                                aligned_min_val = round(aligned_min_val, decimal_places)
                
                        corrected_data[feature] = str(aligned_min_val)
                    else:
                        corrected_data[feature] = np.nan
                else:
                    corrected_data[feature] = gen_value_str
        return corrected_data
    def sample_attn(self, sample_num=1, feature_order=None, p=0.5, max_new_tokens_per_value=20, temperature=1.0, mi_threshold=0.01, apply_final_constraints=True):
        self.model.eval()
        self.selector.eval()
        
        final_samples = []

        with torch.no_grad():
            for _ in tqdm(range(sample_num), desc="Generating Samples with Attention Mask"):

                if feature_order:
                    current_feature_order = feature_order.copy()
                else:
                    current_feature_order = self.feature_columns.copy()
                    random.shuffle(current_feature_order)

                num_predefined = int(p * len(current_feature_order))
                predefined_feats = current_feature_order[:num_predefined]

                random_row = self.df.sample(1).iloc[0]

                feature_value_map = {feat: str(random_row[feat]) for feat in predefined_feats}

                past_feature_value_pairs = [(feat, str(random_row[feat])) for feat in predefined_feats]

                for i in range(num_predefined, len(current_feature_order)):
                    current_feat = current_feature_order[i]

                    prompt_input_ids = []
                    prompt_attention_mask = []

                    if past_feature_value_pairs:
                        mi_scores = self.selector(current_feat, past_feature_value_pairs)
                        
                        for idx, (past_feat, past_val) in enumerate(past_feature_value_pairs):

                            text_part = f"{past_feat} is "
                            text_part2= f"{past_val}"
                            token_ids_part = self.tokenizer.encode(text_part, add_special_tokens=False)
                            token_ids_part2 = self.tokenizer.encode(text_part2, add_special_tokens=False)
                            prompt_input_ids.extend(token_ids_part)
                            prompt_input_ids.extend(token_ids_part2)

                            is_relevant = mi_scores[idx] >= mi_threshold
                            if is_relevant:
                                prompt_attention_mask.extend([1] * len(token_ids_part))
                                prompt_attention_mask.extend([1] * len(token_ids_part2))
                            else:
                                prompt_attention_mask.extend([1] * len(token_ids_part))
                                prompt_attention_mask.extend([0] * len(token_ids_part2))

                            if idx < len(past_feature_value_pairs) - 1:
                                separator_tokens = self.tokenizer.encode(", ", add_special_tokens=False)
                                prompt_input_ids.extend(separator_tokens)
                                prompt_attention_mask.extend([1] * len(separator_tokens)) 

                    if past_feature_value_pairs:
                        leading_separator = self.tokenizer.encode(", ", add_special_tokens=False)
                        prompt_input_ids.extend(leading_separator)
                        prompt_attention_mask.extend([1] * len(leading_separator))
                        
                    current_prompt_base = f"{current_feat} is"
                    token_ids_base = self.tokenizer.encode(current_prompt_base, add_special_tokens=False)
                    prompt_input_ids.extend(token_ids_base)
                    prompt_attention_mask.extend([1] * len(token_ids_base))

                    input_ids_tensor = torch.tensor([prompt_input_ids]).to(self.device)
                    attention_mask_tensor = torch.tensor([prompt_attention_mask]).to(self.device)


                    logits_processor_list = LogitsProcessorList()
                    logits_processor_list.append(NoLeadingCommaLogitsProcessor(self.tokenizer))
                    
                    effective_max_new_tokens = self.max_tokens_for_value.get(current_feat, max_new_tokens_per_value)

                    output_ids = self.model.generate(
                        input_ids=input_ids_tensor,
                        attention_mask=attention_mask_tensor,
                        max_new_tokens=effective_max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_k=50,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        logits_processor=logits_processor_list
                    )

 
                    generated_ids = output_ids[0][input_ids_tensor.shape[1]:]
                    generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    generated_value = generated_text.split(",")[0].strip()

                    if not generated_value:
                        generated_value = str(self.df[current_feat].dropna().sample(1).iloc[0])

                    feature_value_map[current_feat] = generated_value
                    past_feature_value_pairs.append((current_feat, generated_value))

                if apply_final_constraints:
                    corrected_sample = self.apply_constraints(feature_value_map)
                    final_samples.append(corrected_sample)
                else:
                    final_samples.append(feature_value_map)

        if not final_samples:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(final_samples)
        for col in self.feature_columns:
            if col not in result_df.columns:
                result_df[col] = pd.NA
        
        return result_df[self.feature_columns]

    def sample_logits(self, sample_num=1, feature_order=None, p=0.5, max_new_tokens_per_value=20, temperature=1.0, mi_bias_scale=0.1, mi_floor_bias=-1.0, apply_final_constraints=True):

        self.model.eval()
        self.selector.eval() 
        
        final_samples = []
    
        with torch.no_grad():
            for _ in tqdm(range(sample_num), desc="Generating Samples with MI Weighted Logits"):
                try:
                   
                    if feature_order:
                        current_feature_order = feature_order.copy()
                    else:
                        current_feature_order = self.feature_columns.copy()
                        random.shuffle(current_feature_order)
        
                    num_predefined = int(p * len(current_feature_order))
                    predefined_feats = current_feature_order[:num_predefined]
                    
                    random_row = self.df.sample(1).iloc[0]
                    
                    feature_value_map = {feat: str(random_row[feat]) for feat in predefined_feats}
                    past_feature_value_pairs = [(feat, str(random_row[feat])) for feat in predefined_feats]
        
 
                    for i in range(num_predefined, len(current_feature_order)):
                        current_feat = current_feature_order[i]
                        
                        prompt_input_ids = []

                        if past_feature_value_pairs:
                            for idx, (past_feat, past_val) in enumerate(past_feature_value_pairs):
                                text_part = f"{past_feat} is "
                                text_part2 = f"{past_val}"
                                token_ids_part = self.tokenizer.encode(text_part, add_special_tokens=False)
                                token_ids_part2 = self.tokenizer.encode(text_part2, add_special_tokens=False)
                                prompt_input_ids.extend(token_ids_part)
                                prompt_input_ids.extend(token_ids_part2)
                                
                                if idx < len(past_feature_value_pairs) - 1:
                                    separator_tokens = self.tokenizer.encode(", ", add_special_tokens=False)
                                    prompt_input_ids.extend(separator_tokens)
                            
                            leading_separator = self.tokenizer.encode(", ", add_special_tokens=False)
                            prompt_input_ids.extend(leading_separator)
                        
                        current_prompt_base = f"{current_feat} is"
                        token_ids_base = self.tokenizer.encode(current_prompt_base, add_special_tokens=False)
                        prompt_input_ids.extend(token_ids_base)
        
                        input_ids_tensor = torch.tensor([prompt_input_ids]).to(self.device)
                        
                        logits_processor_list = LogitsProcessorList()
                        logits_processor_list.append(NoLeadingCommaLogitsProcessor(self.tokenizer)) 
                        
                        if past_feature_value_pairs: 
                            logits_processor_list.append(
                                MiLogitsBiasProcessor(
                                    tokenizer=self.tokenizer,
                                    current_feat=current_feat,
                                    past_feature_value_pairs=past_feature_value_pairs,
                                    mi_calculator=self.selector,
                                    mi_bias_scale=mi_bias_scale,
                                    mi_floor_bias=mi_floor_bias
                                )
                            )
                        
                        effective_max_new_tokens = self.max_tokens_for_value.get(current_feat, max_new_tokens_per_value)
        
                        output_ids = self.model.generate(
                            input_ids=input_ids_tensor,

                            attention_mask=torch.ones_like(input_ids_tensor), 
                            max_new_tokens=effective_max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_k=50,
                            top_p=0.95,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            logits_processor=logits_processor_list
                        )
        

                        generated_ids = output_ids[0][input_ids_tensor.shape[1]:]
                        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                        generated_value = generated_text.split(",")[0].strip()
                        
                        if not generated_value:
                            generated_value = str(self.df[current_feat].dropna().sample(1).iloc[0])
        
                        feature_value_map[current_feat] = generated_value
                        past_feature_value_pairs.append((current_feat, generated_value))
                    
                    if apply_final_constraints:
                        corrected_sample = self.apply_constraints(feature_value_map)
                        final_samples.append(corrected_sample)
                    else:
                        final_samples.append(feature_value_map)
                except:
                    pass
        
        if not final_samples:
            return pd.DataFrame()
            
        result_df = pd.DataFrame(final_samples)
        for col in self.feature_columns:
            if col not in result_df.columns:
                result_df[col] = pd.NA
                
        return result_df[self.feature_columns]