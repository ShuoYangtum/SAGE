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

    def set_mi_data(self, mi_table, feature_columns, discretizers):

        self.mi_table = mi_table
        self.feature_columns = feature_columns
        self.discretizers = discretizers 
        print(f"Selector MI table loaded with {len(mi_table)} entries.")

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
    def __init__(self, tokenizer, current_feat, past_feature_value_pairs, mi_calculator, mi_bias_scale=0.1, mi_floor_bias=-1.0):
        self.tokenizer = tokenizer
        self.current_feat = current_feat
        self.past_feature_value_pairs = past_feature_value_pairs
        self.mi_calculator = mi_calculator
        self.mi_bias_scale = mi_bias_scale
        self.mi_floor_bias = mi_floor_bias

        self.mi_scores = []
        if past_feature_value_pairs:
            self.mi_scores = self.mi_calculator(current_feat, past_feature_value_pairs)
            mi_scores_np = np.array(self.mi_scores)
            if mi_scores_np.size > 0:
                mi_scores_np = (mi_scores_np - mi_scores_np.min()) / (mi_scores_np.max() - mi_scores_np.min() + 1e-6)
                self.mi_scores = mi_scores_np.tolist()
            else:
                self.mi_scores = []


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        assert input_ids.shape[0] == 1, "MiLogitsBiasProcessor assumes batch_size = 1"
        
        bias = torch.zeros_like(scores)

        if self.mi_scores:
    
            avg_mi_score = sum(self.mi_scores) / len(self.mi_scores)

            global_bias = avg_mi_score * self.mi_bias_scale

            global_bias = max(global_bias, self.mi_floor_bias)
            
            bias += global_bias 

        return scores + bias