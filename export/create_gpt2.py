import torch
from transformers import GPT2Config, GPT2LMHeadModel

# -----------------------------
# Hyperparameters
# -----------------------------
MAX_POSITION_EMBEDDINGS = 8192
EMBEDDING_SIZE = 512
FEEDFORWARD_SIZE = EMBEDDING_SIZE * 3
NUM_LAYERS = 8
NUM_ATTENTION_HEADS = 8
VOCAB_SIZE = 16000

# attn implementation: "eager", "sdpa", or "flash_attention_2" (HF >= 4.40)
attn_implem = None
attn_implem_name = attn_implem if attn_implem else "default_attn"
dtype = "float32"   # or "bfloat16"/"float16" if you want


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    # -----------------------------
    # Create config
    # -----------------------------
    gpt2_config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=MAX_POSITION_EMBEDDINGS,
        n_embd=EMBEDDING_SIZE,
        n_layer=NUM_LAYERS,
        n_head=NUM_ATTENTION_HEADS,
        n_inner=FEEDFORWARD_SIZE,
        attn_implementation=attn_implem,
        torch_dtype=dtype,
    )

    print(gpt2_config)

    # -----------------------------
    # Create model with random weights
    # -----------------------------
    model = GPT2LMHeadModel(gpt2_config)

    # -----------------------------
    # Save model + config
    # -----------------------------
    save_dir = args.out + f"/gpt2_{attn_implem_name}_{dtype}"
    model.save_pretrained(save_dir)
    print(f"✅ Saved GPT2 model to {save_dir}/")

    # Check files: pytorch_model.bin + config.json should be there