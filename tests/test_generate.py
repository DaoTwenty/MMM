import pytest
from pathlib import Path
from mmm import (
    ModelConfig, Model, Tokenizer, PromptConfig,
    SamplingEngine, GenerationConfig, Score, generate
)

RES = Path(__file__).parent / "resources"
TOKENIZER = RES / "tokenizer.json"
MODEL = RES.parent.parent / "models" / "model.onnx"
TEST_MIDI_SINGLE = RES / "midis" / "POP909_010.mid"
TEST_MIDI_MULTI = RES / "midis" / "test_in.mid"

def test_sampling_engine_basic():
    tok = Tokenizer(str(TOKENIZER))
    cfg = GenerationConfig(max_new_tokens=10)
    engine = SamplingEngine(cfg, tok, seed=123, verbose=True)

    assert engine.config.max_new_tokens == 10
    assert isinstance(engine.vocab_size, int)
    assert isinstance(engine.eos_token_id, int)

    engine.seed = -1  # should randomize
    assert isinstance(engine.seed, int)

def test_generate_infill_multitrack(tmp_path):
    # create minimal fake components
    model_cfg = ModelConfig(model=str(MODEL), vocab_size=16000)
    model = Model(model_cfg)
    tokenizer = Tokenizer(str(TOKENIZER))
    bars = {"bars": {0: [(2, 3, [])]}}
    prompt_cfg = PromptConfig(bars, context_length=4)
    gen_cfg = GenerationConfig(max_new_tokens=128)
    engine = SamplingEngine(gen_cfg, tokenizer, seed=-1, verbose=False)
    score_in = Score(str(TEST_MIDI_SINGLE))

    # run generation
    result = generate(model, tokenizer, prompt_cfg, engine, score_in, verbose=True)
    assert isinstance(result, Score)

    # saving should succeed
    out_path = tmp_path / "output_infill.mid"
    result.save(str(out_path))
    assert out_path.exists()

def test_generate_infill_multitrack(tmp_path):
    # create minimal fake components
    model_cfg = ModelConfig(model=str(MODEL), vocab_size=16000)
    model = Model(model_cfg)
    tokenizer = Tokenizer(str(TOKENIZER))
    bars = {"bars": {0: [(2, 3, [])]}}
    prompt_cfg = PromptConfig(bars, context_length=4)
    gen_cfg = GenerationConfig(max_new_tokens=128)
    engine = SamplingEngine(gen_cfg, tokenizer, seed=-1, verbose=False)
    score_in = Score(str(TEST_MIDI_MULTI))

    # run generation
    result = generate(model, tokenizer, prompt_cfg, engine, score_in, verbose=True)
    assert isinstance(result, Score)

    # saving should succeed
    out_path = tmp_path / "output_infill.mid"
    result.save(str(out_path))
    assert out_path.exists()

def test_generate_sample(tmp_path):
    # create minimal fake components
    model_cfg = ModelConfig(model=str(MODEL), vocab_size=16000)
    model = Model(model_cfg)
    tokenizer = Tokenizer(str(TOKENIZER))
    gen_cfg = GenerationConfig(max_new_tokens=128)
    engine = SamplingEngine(gen_cfg, tokenizer, seed=-1, verbose=False)
    score_in = Score(str(TEST_MIDI_MULTI))

    tracks = { "tracks" : [(18, [])]}
    prompt_cfg = PromptConfig(tracks, context_length=4)
    result = generate(model, tokenizer, prompt_cfg, engine, score_in, verbose=True)

    # saving should succeed
    out_path = tmp_path / "output_sample.mid"
    result.save(str(out_path))
    assert out_path.exists()
