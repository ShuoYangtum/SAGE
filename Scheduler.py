import torch
import torch.nn as nn

class Classification_Head(nn.Module):
    def __init__(self, feature_num=2, hidden_size=768):
        super().__init__()
        self.head=nn.Linear(hidden_size, feature_num)
    def forward(self, hidden_states):
        hidden_states=hidden_states.last_hidden_state[:, -1, :]
        return self.head(hidden_states)


class Scheduler(nn.Module):
    def __init__(self, generator, feature_num=2, hidden_size=768, device=torch.device("cuda")):
        """
        Args:
            generator: the base model
            model_name: the model card of the base model
            feature_num: number of features
        """
        super().__init__()
        self.device=device
        self.generator=generator.to(device)
        self.head=Classification_Head(feature_num=feature_num, hidden_size=hidden_size).to(device)
    
    def forward(self, input_ids, attention_mask):
        hidden_states=self.generator(input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
        return torch.softmax(self.head(hidden_states), -1)