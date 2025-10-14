from pathlib import Path
from mmm import Tokenizer

RES = Path(__file__).parent / "resources"
TOKENIZER = RES / "tokenizer.json"

def test_tokenizer_save(tmp_path):
    tok = Tokenizer(str(TOKENIZER))