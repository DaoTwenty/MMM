import onnxruntime as ort
import numpy as np
import random

# === CONFIG ===
MODEL_PATH = "../models/GPT2/gpt2_default_attn_float32/trans/model_opt.onnx"  # <-- replace with your path
STEPS = 5                       # number of forward passes
VOCAB_SIZE = 16000              # adjust based on your model
START_TOKENS = [1, 42]          # initial prompt

BATCH_SIZE = 1
NUM_HEADS = 8
HEAD_SIZE = 64
MAX_POS = 8192

# === LOAD MODEL ===
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

print("=== Model Inputs ===")
for inp in session.get_inputs():
    print(f"{inp.name}: shape={inp.shape}, type={inp.type}")
print("=== Model Outputs ===")
for out in session.get_outputs():
    print(f"{out.name}: shape={out.shape}, type={out.type}")

# Collect input/output names for easier handling
input_names = [i.name for i in session.get_inputs()]
output_names = [o.name for o in session.get_outputs()]

# Initialize state
tokens = list(START_TOKENS)
past_key_values = None  # None = first pass (empty past)

for step in range(STEPS):
    print(f"\n=== STEP {step+1} ===")
    seq_len = len(tokens)

    # Prepare input_ids for this step (usually just last token after step 1)
    input_ids = np.array(tokens[-1:], dtype=np.int32).reshape(BATCH_SIZE, 1) if step > 0 \
               else np.array(tokens, dtype=np.int32).reshape(BATCH_SIZE, seq_len)

    # Compute past_len
    past_len = 0
    if past_key_values:
        # any past tensor will have shape [2, batch, num_heads, past_len, head_size]
        past_len = past_key_values[0].shape[3]

    # Build attention_mask: [batch, past_len + seq_len]
    attention_mask = None
    if "attention_mask" in input_names:
        total_len = past_len + input_ids.shape[1]
        attention_mask = np.ones((BATCH_SIZE, total_len), dtype=np.int32)

    # Build position_ids: [batch, seq_len]
    position_ids = None
    if "position_ids" in input_names:
        pos = np.arange(past_len, past_len + input_ids.shape[1], dtype=np.int32)
        pos = np.clip(pos, 0, MAX_POS - 1)
        position_ids = pos.reshape(BATCH_SIZE, input_ids.shape[1])

    # Build input dict
    inputs = {}
    for name in input_names:
        if name == "input_ids":
            inputs[name] = input_ids
        elif name == "position_ids" and position_ids is not None:
            inputs[name] = position_ids
        elif name == "attention_mask" and attention_mask is not None:
            inputs[name] = attention_mask
        elif "past" in name.lower():
            # fill with previous past values or empty tensors
            if past_key_values is not None:
                idx = len([k for k in inputs if "past" in k])
                inputs[name] = past_key_values[idx]
            else:
                zero_shape = [2, BATCH_SIZE, NUM_HEADS, 0, HEAD_SIZE]
                inputs[name] = np.zeros(zero_shape, dtype=np.float32)

    # Debug input summary
    for k, v in inputs.items():
        shape_str = "x".join(str(x) for x in v.shape)
        print(f"{k}: shape={shape_str}, dtype={v.dtype}, first_elements={v.flatten()[:10]}")

    # Run model
    outputs = session.run(output_names, inputs)

    # outputs[0] = logits, outputs[1..] = present key/values
    logits = outputs[0]
    present = outputs[1:]

    print(f"logits shape: {logits.shape}, first_logits={logits.flatten()[:5]}")

    # Update past_key_values for next step
    past_key_values = present

    # Choose next token (random sampling just for debugging)
    next_token = random.randint(0, VOCAB_SIZE - 1)
    tokens.append(next_token)
    print(f"next token chosen = {next_token}")