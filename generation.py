import torch
from transformers import LogitsProcessor

class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = torch.tensor(list(allowed_token_ids), dtype=torch.long)
        if len(self.allowed_token_ids) == 0:
            raise ValueError("allowed_token_ids cannot be empty.")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        mask = torch.zeros_like(scores, dtype=torch.bool)
        allowed_tokens_on_device = self.allowed_token_ids.to(scores.device)
        mask.scatter_(1, allowed_tokens_on_device.unsqueeze(0), True)

        scores[~mask] = -float('inf')
        return scores
        
class NoLeadingCommaLogitsProcessor(LogitsProcessor):
    """
    LogitsProcessor that prevents the generation of a leading comma token.
    This is useful when generating feature values, where a comma would typically
    be a separator *after* the value.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        comma_space_token_ids = self.tokenizer.encode(" ,", add_special_tokens=False)

        comma_token_ids = self.tokenizer.encode(",", add_special_tokens=False)

        self.forbidden_token_ids = set()
        if comma_space_token_ids:
            self.forbidden_token_ids.update(comma_space_token_ids)
        if comma_token_ids:
            self.forbidden_token_ids.update(comma_token_ids)
            
        if not self.forbidden_token_ids:
            print(f"Warning: No valid token IDs found for comma. NoLeadingCommaLogitsProcessor will not be effective.")


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        for token_id in self.forbidden_token_ids:
            if token_id < scores.shape[1]: 
                scores[:, token_id] = -float('inf')
        
        return scores