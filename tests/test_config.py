import json
import pytest
from pathlib import Path
from mmm import GenerationConfig, PromptConfig, ModelConfig

MAX_NEW_TOKENS = 128
DEFAULT_GEN_CONFIG = {
    "do_sample": True,
    "max_new_tokens": MAX_NEW_TOKENS,
    "attempts":4,
    "pad_token_id": 0,
    "repetition_penalty": 1.0,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0
}
RES = Path(__file__).parent / "resources"
MODEL = Path(__file__).parent.parent / "models" / "model.onnx"
GEN = RES / "generation.json"
INFILL = RES / "infill_prompt.json"
SAMPLE = RES / "sample_prompt.json"

def test_generation_config_from_and_to():
    cfg = GenerationConfig(do_sample=True, max_new_tokens=42)
    d = DEFAULT_GEN_CONFIG
    cfg2 = GenerationConfig.from_dict(d)
    cfg3 = GenerationConfig.from_json_str(json.dumps(d))
    assert cfg2.max_new_tokens == MAX_NEW_TOKENS

    cfg4 = GenerationConfig.from_json_file(str(GEN))

def test_prompt_config_modes(tmp_path):
    bars = {"bars": {0: [(0, 4, [])], 1: [(4, 8, [])]}}
    cfg = PromptConfig(bars, context_length=8)
    assert cfg.context_length == 8
    print(cfg.empty)
    assert cfg.empty() == False
    assert isinstance(cfg.bars(), dict)
    to_str = str(cfg)

    json_str = cfg.to_json()
    cfg2 = PromptConfig.from_json_str(json_str)
    assert cfg2.context_length == 8

    json_dict = cfg.to_dict()
    cfg3 = PromptConfig.from_dict(json_dict)
    assert cfg3.context_length == 8

    file_path = tmp_path / "prompt.json"
    cfg.save_json(str(file_path))
    cfg4 = PromptConfig.from_json_file(str(file_path))
    assert cfg4.context_length == 8

@pytest.mark.parametrize(
    "load_path",
    [
        RES / "model_config.json",
        None
    ]
)
def test_model_config_roundtrip(tmp_path, load_path):

    save_path = tmp_path / "model_config.json"

    # Load existing file if available
    if load_path is not None:
        if not load_path.exists():
            pytest.skip(f"Missing resource file: {load_path}")
        print(str(load_path))
        cfg = ModelConfig.from_file(str(load_path))
    else:
        cfg = ModelConfig(model=str(MODEL), cached=True, vocab_size=16000)

    # Always save to temp path
    cfg.save_json(str(save_path))
    assert save_path.exists()

    # Reload from temp save path to verify save integrity
    cfg2 = ModelConfig.from_file(str(save_path))

    # Assertions
    assert cfg2.model == cfg.model
    assert cfg2.vocab_size == cfg.vocab_size
    assert cfg2.cached == cfg.cached

    # Common serialization checks
    d = cfg2.to_dict()
    j = cfg2.to_json(indent=2)
    assert isinstance(d, dict)
    assert isinstance(j, str)
    cfg3 = ModelConfig.from_dict(d)
    cfg4 = ModelConfig.from_json(j)
    assert "vocab_size" in d and "vocab_size" in j
