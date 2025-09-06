import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import StaticCache

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "/Users/paultriana/creative_labs/models/MISTRAL_123000"
EXPORT_PATH = "/Users/paultriana/creative_labs/models/MISTRAL_123000_OPT_ONNX/model.onnx"
SEQ_LEN = 8       # dummy input length for export
MAX_CACHE_LEN = 16  # dummy max cache length for export
BATCH = 1

# -----------------------------
# Load model
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()

# -----------------------------
# Wrapper to handle StaticCache
# -----------------------------
class MistralForONNX(torch.nn.Module):
    def __init__(self, model, max_cache_len=16):
        super().__init__()
        self.model = model
        self.num_layers = model.config.num_hidden_layers
        self.max_cache_len = max_cache_len

    def forward(self, input_ids, attention_mask=None, *past_key_values_flat):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        dtype = self.model.dtype

        # create StaticCache
        past = StaticCache(
            config=self.model.config,
            max_batch_size=batch_size,
            max_cache_len=self.max_cache_len,
            device=device,
            dtype=dtype
        )

        # run the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past
        )

        # flatten key_cache and value_cache manually
        flat_present = []
        for k_tensor, v_tensor in zip(outputs.past_key_values.key_cache, outputs.past_key_values.value_cache):
            flat_present.extend([k_tensor, v_tensor])

        return (outputs.logits, *flat_present)

onnx_model = MistralForONNX(model)

# -----------------------------
# Dummy inputs for export
# -----------------------------
dummy_input_ids = torch.randint(0, model.config.vocab_size, (BATCH, SEQ_LEN), dtype=torch.long)
dummy_attention_mask = torch.ones_like(dummy_input_ids, dtype=torch.long)

# For ONNX, past_key_values_flat is empty; wrapper handles cache internally
inputs = (dummy_input_ids, dummy_attention_mask)

# -----------------------------
# Export to ONNX
# -----------------------------
input_names = ["input_ids", "attention_mask"]
output_names = ["logits"]
dynamic_axes = {
    "input_ids": {1: "seq_len"},
    "attention_mask": {1: "seq_len"},
}

# Add outputs for each layer’s present key/value
for i in range(model.config.num_hidden_layers):
    output_names.append(f"present.{i}.key")
    output_names.append(f"present.{i}.value")
    dynamic_axes[f"present.{i}.key"] = {2: "past_seq_len"}
    dynamic_axes[f"present.{i}.value"] = {2: "past_seq_len"}

torch.onnx.export(
    onnx_model,
    inputs,
    EXPORT_PATH,
    input_names=input_names,
    output_names=output_names,
    opset_version=18,
    dynamic_axes=dynamic_axes,
    do_constant_folding=True,
)

print(f"✅ Exported Mistral with cache support to {EXPORT_PATH}")
