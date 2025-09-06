import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import numpy as np

# -----------------------------
# Arguments
# -----------------------------
if __name__ == "__main__":
    # -----------------------------
    # Arguments
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/Users/paultriana/creative_labs/models/MISTRAL_123000")
    parser.add_argument("--seq_len", type=int, default=512)   # long initial context
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_new_tokens", type=int, default=50)  # how many tokens to "generate"
    parser.add_argument("--vocab_size", type=int, default=16000)
    args = parser.parse_args()

    # -----------------------------
    # Load model and tokenizer
    # -----------------------------
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.eval()

    batch_size = args.batch_size
    seq_len = args.seq_len
    vocab_size = args.vocab_size

    # -----------------------------
    # Random initial input
    # -----------------------------
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    # -----------------------------
    # 1) Without caching
    # -----------------------------
    latencies_no_cache = []
    cur_input = input_ids.clone()
    cur_attention = attention_mask.clone()

    with torch.no_grad():
        for _ in range(args.num_new_tokens):
            start = time.time()
            outputs = model(input_ids=cur_input, attention_mask=cur_attention, use_cache=False)
            end = time.time()
            latencies_no_cache.append(end - start)

            # append 1 new random token for next step
            new_token = torch.randint(0, vocab_size, (batch_size, 1))
            cur_input = torch.cat([cur_input, new_token], dim=-1)
            cur_attention = torch.cat([cur_attention, torch.ones_like(new_token)], dim=-1)

    print("=== Without caching ===")
    print(f"Mean: {np.mean(latencies_no_cache)*1000:.2f} ms")
    print(f"Min: {np.min(latencies_no_cache)*1000:.2f} ms")
    print(f"Max: {np.max(latencies_no_cache)*1000:.2f} ms")
    print(f"Std: {np.std(latencies_no_cache)*1000:.2f} ms")

    # -----------------------------
    # 2) With caching
    # -----------------------------
    latencies_cache = []

    with torch.no_grad():
        # first forward pass on full sequence (gets cache)
        start = time.time()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        end = time.time()
        latencies_cache.append(end - start)

        past = outputs.past_key_values  # proper Cache object

        # subsequent passes: only 1 new token at a time
        for _ in range(args.num_new_tokens - 1):
            new_token = torch.randint(0, vocab_size, (batch_size, 1))
            attention_mask = torch.ones_like(new_token)

            start = time.time()
            outputs = model(input_ids=new_token, attention_mask=attention_mask, use_cache=True, past_key_values=past)
            end = time.time()
            latencies_cache.append(end - start)

            past = outputs.past_key_values  # update cache

    print("\n=== With caching (autoregressive) ===")
    print(f"Mean: {np.mean(latencies_cache)*1000:.2f} ms")
    print(f"Min: {np.min(latencies_cache)*1000:.2f} ms")
    print(f"Max: {np.max(latencies_cache)*1000:.2f} ms")
    print(f"Std: {np.std(latencies_cache)*1000:.2f} ms")