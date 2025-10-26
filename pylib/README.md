# MMM Python Module Documentation

Python bindings for the MMM music modeling library. Provides models, tokenizers, prompt configurations, and sampling utilities.

## Usage

Example:

```python
from mmm import Model, Tokenizer, PromptConfig, SamplingEngine, GenerationConfig, Score, generate

# Load model and tokenizer
model_cfg = ModelConfig(path="model.onnx", vocab_size=512)
model = Model(model_cfg)
tokenizer = Tokenizer("tokenizer.json")

# Create prompt
prompt_cfg = PromptConfig({"bars": {0: [(0, 4, ["attribute_control_1", "attribute_control_2"])], 1: [(0, 4, ["attribute_control_3"])]}}, context_length=8)

# Create sampling engine
gen_cfg = GenerationConfig(max_new_tokens=50)
engine = SamplingEngine(gen_cfg, tokenizer)

# Create initial score (empty)
score_obj = Score("score_to_infill.mid")

# Run generation
generated_score = generate(
    model, 
    tokenizer, 
    prompt_cfg, 
    engine, 
    score_obj
)

# Save output
generated_score.save("infilled_score.mid")
```

## Functions

### Generation endpoint

```python
score = generate(model, tokenizer, prompt_config, sampling_engine, score)
```

Run a music generation using the provided model, tokenizer, prompt configuration, sampling engine, and optional score.

#### Parameters:

- `model (mmm.Model)` – The model to use for generation.

- `tokenizer (mmm.Tokenizer)` – Tokenizer instance.

- `prompt_config (mmm.PromptConfig)` – Prompt configuration for generation.

- `sampling_engine (mmm.SamplingEngine)` – Engine controlling sampling parameters.

- `score (mmm.Score)` – Optional input score.

#### Returns:
Generated `Score` object.

### Setting logging level

```python
set_log_level(LogLevel.DEBUG)
```

Set the global log level (see `LogLevel` class).

#### Parameters:

- `log_level (mmm.LogLevel)`

### Setting logging method

```python
set_log_medium(
    LogLevel.DEBUG,
    LogMedium.FILE,
    "debug.out"
)
```

Set the log method for each log level (see `LogMedium` class).

#### Parameters:

- `log_level (mmm.LogLevel)`

- `log_medium (mmm.LogMeduim)`

- `filename (string)`

## Classes

### GenerationConfig

Controls generation parameters for sampling.

#### Constructors:

```python
cfg = mmm.GenerationConfig()  # default
cfg = mmm.GenerationConfig(
    do_sample=True,
    max_new_tokens=100,
    attempts=3,
    pad_token_id=0,
    repetition_penalty=1.0,
    temperature=1.0,
    top_p=1.0,
    top_k=0
)
```

#### Static Methods:

- `from_json_str(json_str)` – Create from JSON string.

- `from_dict(dict_obj)` – Create from Python dictionary.

- `from_json_file(path)` – Load from JSON file.

#### Attributes:

- `do_sample: bool`

- `max_new_tokens: int`

- `attempts: int`

- `pad_token_id: int`

- `repetition_penalty: float`

- `temperature: float`

- `top_k: int`

- `top_p: float`

#### Example:

```python
cfg = mmm.GenerationConfig(do_sample=True, max_new_tokens=50)
```

### SamplingEngine

Wraps the sampling logic using a tokenizer and generation config.

#### Constructor:

```python
engine = mmm.SamplingEngine(config, tokenizer, seed=42)
```

#### Properties:

- `config` – Get/set `GenerationConfig`.

- `seed` – Get/set RNG seed (Setting -1 picks random seed).

- `vocab_size` (readonly) – Vocabulary size.

- `eos_token_id` (readonly) – End-of-sequence token ID.

#### Example:

```python
print(engine.vocab_size, engine.eos_token_id)
engine.seed = 123
```

### Tokenizer

Loads and optionally saves a tokenizer.

#### Constructor:

```python
tokenizer = mmm.Tokenizer("tokenizer.json")
```

#### Methods:

- `save(path)` – Save tokenizer to file.

#### Example:

```python
tokenizer.save("tokenizer_copy.json")
```

### Score

Represents a musical score (MIDI).

#### Constructor:

```python
score = mmm.Score("file.mid")   # load from MIDI file
```

#### Methods:

- `save(path)` – Save the score to MIDI.

#### Example:

```python
score.save("output.mid")
```

### PromptConfig

Holds the prompt configuration for generation.

#### Constructors:

```python
cfg = mmm.PromptConfig()  # default
cfg = mmm.PromptConfig(mode_dict, context_length=4)
```

#### Properties:

- `context_length: int` – Get/set context length in bars.

- `bar_infilling()` – Returns True if in `BarInfilling` mode.

- `track_sampling()` – Returns True if in `TrackSampling` mode.

- `empty()` – Returns True if mode is empty.

- `bars()` – Returns bars (only in `BarInfilling` mode).

#### JSON / Dict Methods:

- `from_json_str(json_str)`

- `from_json_file(path)`

- `from_dict(dict_obj)`

- `to_dict()` – Returns Python dictionary.

- `to_json()` – Returns JSON string.

- `save_json(path)` – Save to JSON file.

#### Example:

```python
bar_mode = {"bars": {0: [(0, 4, ["piano", "drums"])], 1: [(4, 8, ["bass"])]}}
prompt_cfg = mmm.PromptConfig(bar_mode, context_length=8)
prompt_cfg.save_json("prompt_config.json")
json_str = prompt_cfg.to_json()
```

### ModelConfig

Configuration object for constructing models.

#### Constructors:

```python
cfg = mmm.ModelConfig()
cfg = mmm.ModelConfig(
    path="model.onnx",
    cached=True,
    coreml=False, # Only for ONNX backend engine
    vocab_size=512
)
```

#### Methods:

- `from_file(path)` – Load from JSON file.

- `from_json(json_str)` – Load from JSON string.

- `from_dict(dict_obj)` – Load from Python dict.

- `to_dict()` – Convert to Python dict.

- `to_json(indent=2)` – Convert to JSON string.

- `save_json(path, indent=2)` – Save to JSON file.

#### Example:

```python
model_cfg = mmm.ModelConfig(path="model.onnx", cached=True, vocab_size=512)
print(model_cfg.to_json())
```

### Model

Opaque model object. Constructed from ModelConfig.

#### Constructor:

```python
model = mmm.Model(model_cfg)
```

#### Example:

```python
model = mmm.Model(model_cfg)
```

### LogLevel (ENUM)

Global logging levels.

#### Values

- `FATAL`

- `ERROR`

- `WARN`

- `INFO`

- `DEBUG`

- `TRACE`

### LogMedium (ENUM)

Logging method (per log level).

#### Values

- `NONE`

- `CONSOLE`

- `FILE`

- `BOTH`