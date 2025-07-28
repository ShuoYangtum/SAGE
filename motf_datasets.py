import torch
from torch.utils.data import Dataset
import random

class FeatureTextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.input_ids_list = []
        self.attention_masks = []
        self.labels_list = []

        for _, row in df.iterrows():
            pairs = []
            for col in df.columns:
                feat = col
                val = str(row[col])
                pairs.append((feat, val))

            random.shuffle(pairs) 

            input_ids = []
            labels = []

            for i, (feat, val) in enumerate(pairs):
                prefix = f"{feat} is "
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                value_ids = tokenizer.encode(val, add_special_tokens=False)

                input_ids.extend(prefix_ids)
                labels.extend([-100] * len(prefix_ids))

                input_ids.extend(value_ids)
                labels.extend(value_ids)

                sep_ids = tokenizer.encode(", ", add_special_tokens=False)
                input_ids.extend(sep_ids)
                labels.extend(sep_ids)

            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            attention_mask = [1] * len(input_ids)

            pad_len = max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            attention_mask += [0] * pad_len

            self.input_ids_list.append(torch.tensor(input_ids))
            self.labels_list.append(torch.tensor(labels))
            self.attention_masks.append(torch.tensor(attention_mask))

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids_list[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels_list[idx],
        }