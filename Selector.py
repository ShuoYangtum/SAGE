import torch
import torch.nn as nn
from transformers import LogitsProcessor, LogitsProcessorList
import numpy as np

class Selection_Head(nn.Module):
    def __init__(self, feature_num=2, hidden_size=768):
        super().__init__()
        self.shead=nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid() 
        )
    def forward(self, hidden_states):
        return self.shead(hidden_states)



class Selector(nn.Module):
    def __init__(self, hidden_size=768, feature_num=2, device=torch.device("cuda")):
        super().__init__()
        self.device = device
        self.feature_num = feature_num

        self.feature_embeddings = nn.Embedding(feature_num, hidden_size).to(device) 
        self.scorer = Selection_Head(feature_num=feature_num, hidden_size=hidden_size).to(device)

    def forward(self, encoder_hidden_states, feature_indices):

        B, F, H = encoder_hidden_states.shape

        if feature_indices.dim() == 1:
            feature_indices = feature_indices.unsqueeze(0).expand(B, -1)  # -> [B, F]

        feature_embs = self.feature_embeddings(feature_indices.to(self.device))  # [B, F, H]
        cat = torch.cat([encoder_hidden_states, feature_embs], dim=-1)  # [B, F, 2H]

        scores = self.scorer(cat).squeeze(-1)  # [B, F]
        return scores

class NonParametricMISelector(nn.Module):
    def __init__(self):
        super().__init__()

        self.mi_table = {} 
        self.feature_columns = []
        self.discretizers = None 
        self.feature_mean_mi = {}

    def set_mi_data(self, mi_table, feature_columns, discretizers):

        self.mi_table = mi_table
        self.feature_columns = feature_columns
        self.discretizers = discretizers 
        self.feature_mean_mi = {}
        for (current_feature, _, _), score in mi_table.items():
            self.feature_mean_mi.setdefault(current_feature, []).append(float(score))
        self.feature_mean_mi = {
            feat: (float(np.mean(scores)) if len(scores) > 0 else 0.0)
            for feat, scores in self.feature_mean_mi.items()
        }
        print(f"Selector MI table loaded with {len(mi_table)} entries.")

    def get_feature_mean_mi(self, feature_name: str) -> float:
        return float(self.feature_mean_mi.get(feature_name, 0.0))

    def forward(self, current_feature: str, past_feature_value_pairs: list[tuple[str, str]]):
        if not past_feature_value_pairs:
            return torch.empty(0, dtype=torch.float32)

        mi_scores = []
        for p_feat, p_val_str in past_feature_value_pairs:
            
            original_p_val = p_val_str
            if p_feat in self.discretizers:
                try:
                    original_p_val = float(p_val_str)
                    discretized_val = self.discretizers[p_feat].transform([[original_p_val]])[0][0]
                except ValueError:
                    discretized_val = p_val_str 
            else:
                discretized_val = p_val_str

            key = (current_feature, p_feat, discretized_val)
            score = self.mi_table.get(key, 0.0) 

            mi_scores.append(score)
        
        return torch.tensor(mi_scores, dtype=torch.float32)


class MiLogitsBiasProcessor(LogitsProcessor):
    def __init__(self, tokenizer, current_feat, past_feature_value_pairs, mi_calculator, mi_lambda=1.0, scale_clip_min=0.5, scale_clip_max=1.5):
        self.tokenizer = tokenizer
        self.current_feat = current_feat
        self.past_feature_value_pairs = past_feature_value_pairs
        self.mi_calculator = mi_calculator
        self.mi_lambda = mi_lambda
        self.scale_clip_min = scale_clip_min
        self.scale_clip_max = scale_clip_max


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        assert input_ids.shape[0] == 1, "MiLogitsBiasProcessor assumes batch_size = 1"
        if not self.past_feature_value_pairs:
            return scores

        mi_scores = self.mi_calculator(self.current_feat, self.past_feature_value_pairs)
        if mi_scores.numel() == 0:
            return scores

        mu_sample = float(mi_scores.mean().item())
        mu_train = float(self.mi_calculator.get_feature_mean_mi(self.current_feat))
        if mu_train <= 1e-12:
            return scores

        delta = (mu_sample / mu_train) - 1.0
        scale = 1.0 + (self.mi_lambda * delta)
        scale = max(self.scale_clip_min, min(self.scale_clip_max, scale))
        return scores * scale