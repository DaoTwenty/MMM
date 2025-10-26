import pytest
from pathlib import Path
from mmm import (
    ModelConfig, Model, Tokenizer, PromptConfig,
    SamplingEngine, GenerationConfig, Score, generate,
    LogLevel, set_log_level
)

RES = Path(__file__).parent / "resources"
TOKENIZER = RES / "tokenizer.json"
MODEL = RES.parent.parent / "models" / "model.onnx"
MODEL_CACHE = RES.parent.parent / "models" / "model_cache.onnx"
TEST_MIDI_SINGLE = RES / "midis" / "POP909_010.mid"
TEST_MIDI_MULTI = RES / "midis" / "test_in.mid"
LOG = LogLevel.TRACE
set_log_level(LOG)

#MODEL_ARGS = [(MODEL, False),(MODEL_CACHE, True)]
MODEL_ARGS = [(MODEL, False)]

def test_sampling_engine_basic():
    tok = Tokenizer(str(TOKENIZER))
    cfg = GenerationConfig(max_new_tokens=10)
    engine = SamplingEngine(cfg, tok, seed=123)

    assert engine.config.max_new_tokens == 10
    assert isinstance(engine.vocab_size, int)
    assert isinstance(engine.eos_token_id, int)

    engine.seed = -1  # should randomize
    assert isinstance(engine.seed, int)

@pytest.mark.parametrize(
    "model_path,cached",
    MODEL_ARGS
)
def test_generate_infill_multitrack(tmp_path, model_path, cached):
    # create minimal fake components
    model_cfg = ModelConfig(model=str(model_path), vocab_size=16000, cached=cached)
    model = Model(model_cfg)
    tokenizer = Tokenizer(str(TOKENIZER))
    bars = {"bars": {0: [(2, 3, [])]}}
    prompt_cfg = PromptConfig(bars, context_length=4)
    gen_cfg = GenerationConfig(max_new_tokens=128)
    engine = SamplingEngine(gen_cfg, tokenizer, seed=-1)
    score_in = Score(str(TEST_MIDI_SINGLE))

    # run generation
    result = generate(model, tokenizer, prompt_cfg, engine, score_in)
    assert isinstance(result, Score)

    # saving should succeed
    out_path = tmp_path / "output_infill.mid"
    result.save(str(out_path))
    assert out_path.exists()

@pytest.mark.parametrize(
    "model_path,cached",
    MODEL_ARGS
)
def test_generate_infill_multitrack(tmp_path, model_path, cached):
    # create minimal fake components
    model_cfg = ModelConfig(model=str(model_path), vocab_size=16000, cached=cached)
    model = Model(model_cfg)
    tokenizer = Tokenizer(str(TOKENIZER))
    bars = {"bars": {
        0: [(2, 3, [])],
        1: [(2, 3, [])]
    }}
    prompt_cfg = PromptConfig(bars, context_length=4)
    gen_cfg = GenerationConfig(max_new_tokens=256)
    engine = SamplingEngine(gen_cfg, tokenizer, seed=-1)
    score_in = Score(str(TEST_MIDI_MULTI))

    # run generation
    result = generate(model, tokenizer, prompt_cfg, engine, score_in)
    assert isinstance(result, Score)

    # saving should succeed
    out_path = tmp_path / "output_infill.mid"
    result.save(str(out_path))
    assert out_path.exists()

@pytest.mark.parametrize(
    "model_path,cached",
    MODEL_ARGS
)
def test_generate_sample(tmp_path, model_path, cached):
    # create minimal fake components
    model_cfg = ModelConfig(model=str(model_path), vocab_size=16000, cached=cached)
    model = Model(model_cfg)
    tokenizer = Tokenizer(str(TOKENIZER))
    gen_cfg = GenerationConfig(max_new_tokens=128)
    engine = SamplingEngine(gen_cfg, tokenizer, seed=-1)
    score_in = Score(str(TEST_MIDI_MULTI))

    tracks = { "tracks" : [(18, []), (17, [])]}
    prompt_cfg = PromptConfig(tracks, context_length=4)
    result = generate(model, tokenizer, prompt_cfg, engine, score_in)

    # saving should succeed
    out_path = tmp_path / "output_sample.mid"
    result.save(str(out_path))
    assert out_path.exists()
