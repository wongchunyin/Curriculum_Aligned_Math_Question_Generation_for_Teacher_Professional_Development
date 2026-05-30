from transformers import DataCollatorForLanguageModeling
import torch

# 3. Manually pad sequences if needed
def pad_sequence(seq, max_len, pad_token):
    return seq + [pad_token] * (max_len - len(seq))

# 4. Create a custom collator if standard one fails
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, examples):
        # Find max length in batch
        max_len = max(len(ex['input_ids']) for ex in examples)
        
        # Pad all sequences
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
        
        for ex in examples:
            batch['input_ids'].append(
                pad_sequence(ex['input_ids'], max_len, self.tokenizer.pad_token_id)
            )
            batch['attention_mask'].append(
                pad_sequence(ex['attention_mask'], max_len, 0)
            )
            batch['labels'].append(
                pad_sequence(ex['labels'], max_len, -100)  # -100 ignores padding in loss
            )
        
        return {k: torch.tensor(v) for k,v in batch.items()}