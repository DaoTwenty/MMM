# ----- Test MMM import ------

LIB_ATTRIBUTES = [
    "GenerationConfig",
    "SamplingEngine",
    "PromptConfig",
    "Tokenizer",
    "ModelConfig",
    "Model",
    "Score",
    "generate"
]

def test_import():
    import mmm
    for attribute in LIB_ATTRIBUTES:
        assert hasattr(mmm, attribute)
