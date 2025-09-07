import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import StaticCache

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_cache_len", type=int, default=16)
    args = parser.parse_args()

    # -----------------------------
    # Config
    # -----------------------------
    MODEL_NAME = args.model_name
    EXPORT_PATH = MODEL_NAME + "/gpt2_cache.onnx"
    SEQ_LEN = args.seq_len
    BATCH = args.batch_size
    MAX_CACHE_LEN = args.max_cache_len

    # -----------------------------
    # Load GPT-2
    # -----------------------------
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()

    # -----------------------------
    # Wrapper to handle StaticCache
    # -----------------------------
    class GPT2WithCache(torch.nn.Module):
        def __init__(self, model, max_cache_len=16):
            super().__init__()
            self.model = model
            self.num_layers = model.config.num_hidden_layers
            self.max_cache_len = max_cache_len

        def forward(self, input_ids, attention_mask=None):
            batch_size = input_ids.shape[0]
            device = input_ids.device
            dtype = self.model.dtype

            # Create StaticCache for this run
            past = StaticCache(
                config=self.model.config,
                max_batch_size=batch_size,
                max_cache_len=self.max_cache_len,
                device=device,
                dtype=dtype
            )

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past
            )

            # Flatten cache to tensors
            flat_present = []
            for k_tensor, v_tensor in zip(outputs.past_key_values.key_cache,
                                          outputs.past_key_values.value_cache):
                flat_present.extend([k_tensor, v_tensor])

            return (outputs.logits, *flat_present)

    onnx_model = GPT2WithCache(model, max_cache_len=MAX_CACHE_LEN)

    # -----------------------------
    # Dummy inputs
    # -----------------------------
    dummy_input_ids = torch.randint(0, model.config.vocab_size, (BATCH, SEQ_LEN), dtype=torch.long)
    dummy_attention_mask = torch.ones_like(dummy_input_ids, dtype=torch.long)

    # -----------------------------
    # Export
    # -----------------------------
    output_names = ["logits"] + [f"past_{i}" for i in range(2 * model.config.num_hidden_layers)]
    torch.onnx.export(
        onnx_model,
        args=(dummy_input_ids, dummy_attention_mask),
        f=EXPORT_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=18,
    )

    print(f"✅ Exported GPT-2 (with cache) to {EXPORT_PATH}")
