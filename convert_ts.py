import torch
from transformers import AutoModelForCausalLM

class CausalLMWrapper(torch.nn.Module):
    def __init__(self, model, use_cache=False):
        super().__init__()
        self.model = model
        self.use_cache = use_cache

    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=self.use_cache,
        )
        '''
        if self.use_cache:
            # Flatten cache for TorchScript
            flat_past = []
            for k, v in outputs.past_key_values:
                flat_past.extend([k, v])
            return (outputs.logits, *flat_past)
        else:
            return outputs.logits
        '''
        return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torchscript=True)
    model.to(device).eval()
    model.config.return_dict = True

    wrapper = CausalLMWrapper(model, use_cache=args.cache).to(device).eval()

    # Dummy inputs
    batch_size = args.batch_size
    seq_len = args.seq_len
    vocab_size = model.config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long).to(device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long).to(device)

    if args.cache:
        # Build empty past KV cache
        num_heads = model.config.n_head
        head_dim = model.config.n_embd // num_heads
        past = []
        for _ in range(model.config.n_layer):
            k = torch.zeros(batch_size, num_heads, 0, head_dim).to(device)
            v = torch.zeros(batch_size, num_heads, 0, head_dim).to(device)
            past.append((k, v))
        dummy_inputs = (input_ids, attention_mask, tuple(past))
    else:
        dummy_inputs = (input_ids, attention_mask)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=None, use_cache=args.cache, return_dict=False)

    print(f"Type of outputs: {type(outputs)}")
    print(f"Length of outputs: {len(outputs)}")

    for i, out in enumerate(outputs):
        if out is None:
            print(f"outputs[{i}] = None")
        elif isinstance(out, tuple):
            print(f"outputs[{i}] = tuple of length {len(out)}")
            for j, t in enumerate(out):
                if isinstance(t, tuple):
                    print(f"  outputs[{i}][{j}] = tuple of length {len(t)}")
                    for k, tt in enumerate(t):
                        if isinstance(tt, torch.Tensor):
                            print(f"    outputs[{i}][{j}][{k}]: shape={tt.shape}, dtype={tt.dtype}")
                        else:
                            print(f"    outputs[{i}][{j}][{k}] = {type(tt)}")
                elif isinstance(t, torch.Tensor):
                    print(f"  outputs[{i}][{j}]: shape={t.shape}, dtype={t.dtype}")
                else:
                    print(f"  outputs[{i}][{j}] = {type(t)}")
        elif isinstance(out, torch.Tensor):
            print(f"outputs[{i}]: shape={out.shape}, dtype={out.dtype}")
        else:
            print(f"outputs[{i}] = {type(out)}")

    print(len(model(input_ids=input_ids, attention_mask=attention_mask)))
    # Trace the wrapped model
    traced_model = torch.jit.trace(wrapper, dummy_inputs, strict=False)

    out_path = args.model_name + f"/traced_model_{'cache' if args.cache else 'no_cache'}.ts"
    traced_model.save(out_path)
    print(f"✅ Saved TorchScript model to {out_path}")

    '''
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    if args.cache:
        traced_script_module = torch.jit.trace(model, [input_ids, outputs[1]])
    else:
        traced_script_module = torch.jit.trace(model, [input_ids])
    out_path = args.model_name + f"/traced_model_{'cache' if args.cache else 'no_cache'}.ts"
    traced_model.save(out_path)
    print(f"✅ Saved traced model to {out_path}")
    '''
