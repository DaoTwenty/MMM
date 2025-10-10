import torch
from transformers import AutoModelForCausalLM

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    # -----------------------------
    # Config
    # -----------------------------
    MODEL_NAME = args.model_name
    EXPORT_PATH = MODEL_NAME + "/gpt2_nocache.onnx"
    SEQ_LEN = args.seq_len
    BATCH = args.batch_size

    # -----------------------------
    # Load GPT-2
    # -----------------------------
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()

    # -----------------------------
    # Wrapper (only logits, no cache)
    # -----------------------------
    class GPT2NoCache(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask=None):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )
            return outputs.logits  # just logits

    onnx_model = GPT2NoCache(model)

    # -----------------------------
    # Dummy inputs
    # -----------------------------
    dummy_input_ids = torch.randint(0, model.config.vocab_size, (BATCH, SEQ_LEN), dtype=torch.long)
    dummy_attention_mask = torch.ones_like(dummy_input_ids, dtype=torch.long)

    # -----------------------------
    # Export
    # -----------------------------
    torch.onnx.export(
        onnx_model,
        args=(dummy_input_ids, dummy_attention_mask),
        f=EXPORT_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=18,
    )

    print(f"✅ Exported GPT-2 (cacheless) to {EXPORT_PATH}")
