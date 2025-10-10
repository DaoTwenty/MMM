from transformers import AutoTokenizer
from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTModelForCausalLM

def export_gpt2_to_onnx(model_name: str, export_dir: str, use_past: bool = True, optimize: int = 0):
    # Choose task based on whether you want to use cache (past key/values)
    task = "text-generation-with-past" if use_past else "text-generation"
    
    optimize_str = None
    if optimize == 1:
        optimize_str = "O1"
    if optimize == 2:
        optimize_str = "O2"
    if optimize == 3:
        optimize_str = "O3"

    # Export via Optimum exporters
    main_export(
        model_name,
        output=export_dir,
        format="onnx",
        task=task,
        optimize=optimize_str,  # add optimizations like O1–O4 if desired
        model_kwargs={},  # override model.forward defaults if needed
        custom_onnx_configs=None  # Provide per-submodel configs if needed
    )
    
    # Load the exported ONNX model for inference
    ort_model = ORTModelForCausalLM.from_pretrained(export_dir)

    print(f"✅ Exported GPT-2 to {export_dir} with use_past = {use_past} and optimize = {optimize_str if optimize_str else 'None'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()

    for cache in [True, False]:
        for opt in range(0, 4):
            export_dir = f"{args.model_name}/optimum_{'with_cache' if cache else 'no_cache'}_optimize_{opt}"
            export_gpt2_to_onnx(args.model_name, export_dir, use_past=cache)
