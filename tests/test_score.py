from mmm import Score
import pytest
from pathlib import Path

RES = Path(__file__).parent / "resources"
MIDI = RES / "midis"

MIDI_FILES = [
    MIDI / "6338816_Etude No. 4.mid",
    MIDI / "6354774_Macabre Waltz.mid",
    MIDI / "Aicha.mid",
    MIDI / "All The Small Things.mid",
    MIDI / "DAFT PUNK.Around the world.mid",
    MIDI / "Funkytown.mid",
    MIDI / "Girls Just Want to Have Fun.mid",
    MIDI / "I Gotta Feeling.mid",
    MIDI / "In Too Deep.mid",
    MIDI / "Les Yeux Revolvers.mid",
    MIDI / "Maestro_10.mid",
    MIDI / "Maestro_9.mid",
    MIDI / "Mr. Blue Sky.mid",
    MIDI / "Never-Gonna-Let-You-Go.mid",
    MIDI / "POP909_010.mid",
    MIDI / "POP909_022.mid",
    MIDI / "POP909_191.mid",
    MIDI / "Queen - Bohemian Rhapsody.mid",
    MIDI / "Rick-Astley-Never-Gonna-Give-You-Up.mid",
    MIDI / "Shut Up.mid",
    MIDI / "The Beatles - You Never Give Me Your Money.mid",
    MIDI / "What a Fool Believes.mid",
    MIDI / "mmmtest3.mid",
    MIDI / "youre only lonely L.mid",
]

@pytest.mark.parametrize(
    "midi_file", MIDI_FILES
)
def test_score_save(tmp_path, midi_file):
    score = Score(str(midi_file))
    midi_out = tmp_path / "out.mid"
    score.save(str(midi_out))
    assert midi_out.exists()