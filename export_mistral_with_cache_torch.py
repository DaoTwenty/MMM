import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import StaticCache

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "/Users/paultriana/creative_labs/models/MISTRAL_123000"
EXPORT_PATH = "/Users/paultriana/creative_labs/models/MISTRAL_123000_OPT_ONNX/model.onnx"
SEQ_LEN = 8         # dummy input length
MAX_CACHE_LEN = 16  # dummy cache length
BATCH = 1

# -----------------------------
# Load model
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()

# -----------------------------
# Wrapper for StaticCache
# -----------------------------
class MistralForExport(torch.nn.Module):
    def __init__(self, model, max_cache_len=MAX_CACHE_LEN):
        super().__init__()
        self.model = model
        self.num_layers = model.config.num_hidden_layers
        self.max_cache_len = max_cache_len

    def forward(self, input_ids, attention_mask=None, *past_key_values_flat):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        dtype = self.model.dtype

        # Create StaticCache (internal preallocated key/value tensors)
        past = StaticCache(
            config=self.model.config,
            max_batch_size=batch_size,
            max_cache_len=self.max_cache_len,
            device=device,
            dtype=dtype
        )

        # Run the model with caching
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past
        )

        # Flatten key_cache / value_cache manually
        flat_present = []
        for k_tensor, v_tensor in zip(outputs.past_key_values.key_cache, outputs.past_key_values.value_cache):
            flat_present.extend([k_tensor, v_tensor])

        return (outputs.logits, *flat_present)

# your wrapper model
export_model = MistralForExport(model)

# dummy inputs
dummy_input_ids = torch.randint(0, model.config.vocab_size, (BATCH, SEQ_LEN), dtype=torch.long)
dummy_attention_mask = torch.ones_like(dummy_input_ids, dtype=torch.long)
inputs = (dummy_input_ids, dummy_attention_mask)

# export
exported_model = torch.export.export(
    export_model,          # <--- model / callable (f)
    args=inputs,           # <--- example inputs
    dynamic_shapes={       # optional dynamic axes
        "input_ids": {1: "seq_len"},
        "attention_mask": {1: "seq_len"},
        **{f"present.{i}.key": {2: "past_seq_len"} for i in range(model.config.num_hidden_layers)},
        **{f"present.{i}.value": {2: "past_seq_len"} for i in range(model.config.num_hidden_layers)},
    },
    strict=True
)

print(f"✅ Exported Mistral with cache to {EXPORT_PATH} using torch.export")
