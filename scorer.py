import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class Selector(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_size=768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
       
        self.text_encoder = AutoModel.from_pretrained(model_name)

        self.text_encoder_hidden_size = self.text_encoder.config.hidden_size

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), # current_feature_embedding + f_v_pair_embedding
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid() 
        )
        self.hidden_size = hidden_size

    def forward(self, current_feature_text: str, past_feature_value_texts: list[str], device='cpu'):
        current_feature_input = self.tokenizer(current_feature_text, return_tensors='pt', truncation=True, max_length=64).to(device)
        
        current_feature_embedding = self.text_encoder(**current_feature_input).last_hidden_state[:, 0, :] # (1, hidden_size)

        if not past_feature_value_texts:
            return torch.empty(0).to(device) 

        probabilities = []
        for f_v_text in past_feature_value_texts:
            f_v_input = self.tokenizer(f_v_text, return_tensors='pt', truncation=True, max_length=128).to(device)
            f_v_embedding = self.text_encoder(**f_v_input).last_hidden_state[:, 0, :] # (1, hidden_size)

            combined_embedding = torch.cat((current_feature_embedding, f_v_embedding), dim=-1) # (1, 2 * hidden_size)

            prob = self.classification_head(combined_embedding) # (1, 1)
            probabilities.append(prob)
        
        if probabilities:
            return torch.cat(probabilities).squeeze(-1) # (num_past_features,)
        else:
            return torch.empty(0).to(device)
