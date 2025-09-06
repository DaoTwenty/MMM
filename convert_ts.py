# === Configuration ===
model_name_or_path = "/Users/paultriana/creative_labs/models/MISTRAL_123000"
output_export_path = "/Users/paultriana/creative_labs/models/MISTRAL_123000_TORCHSCRIPT/model.pt"
device = "cpu"  # or "cuda"

import torch
from transformers import AutoModelForCausalLM
from torch.export import export

# Model parameters
vocab_size = 16000
seq_len = 3
batch_size = 1
device = "cpu"  # or "cuda" if available

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torchscript=True)
model.eval()
model.to(device)

# Wrapper to avoid kwargs
class MistralWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False
        ).logits

wrapped_model = MistralWrapper(model)

# Generate example inputs
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long).to(device)
attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long).to(device)
position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).to(device)

# Export the model
exported_model = export(
    wrapped_model,
    args=(input_ids, attention_mask, position_ids)
)

# Save the exported model
torch.export.save(exported_model, output_export_path)
print(f"Model successfully exported to {output_export_path}")
