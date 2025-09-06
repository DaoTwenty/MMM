"""Tests for MMM inference."""

import random
import time
import traceback
import logging
from pathlib import Path
import json
import csv
from tqdm import tqdm
import hashlib
import concurrent.futures
import uuid
import os

import numpy as np
import torch.cuda as cuda
import torch
from miditok import MMM
from miditok.pytorch_data import DataCollator
from symusic import Score, Track, Note, Tempo, TimeSignature
from transformers import AutoModelForCausalLM, GenerationConfig
import symusic

from mmm import InferenceConfig, generate

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(prog = f"Test") 
    parser.add_argument("--model",
                        type=str,
                        required=True, help="Model checkpoint")
    
    args = parser.parse_args()

    midi="/Users/paultriana/creative_labs/old_MIDI-GPT/python_scripts_for_testing/mtest.mid"
    midi_out="/Users/paultriana/creative_labs/MMM/test_out.mid"
    config=args.model + "/generation_config.json"
    model=args.model
    tokenizer="runs/tokenizer.json"

    # Parse arguments
    args = parser.parse_args()

    tokenizer = MMM(params=tokenizer)

    # MODEL
    model = AutoModelForCausalLM.from_pretrained(model)

    gen_config_file = Path(config)
    file = open(gen_config_file, 'r')
    data = file.read()
    config_dict = json.loads(data)
    file.close()
    gen_config = GenerationConfig(**config_dict)

    score = symusic.Score(midi)
    tokens = tokenizer.encode(score, concatenate_track_sequences=False)
    num_tracks = len(tokens)

    track_idx = random.randint(0, num_tracks - 1)

    bars_ticks = tokens[track_idx]._ticks_bars
    num_bars = len(bars_ticks)

    bar_idx_infill_start = num_bars-1
    bar_idx_infill_end = num_bars

    bar_tick_end = bars_ticks[
        num_bars-1
    ]

    times = np.array([event.time for event in tokens[track_idx].events])
    token_idx_end = np.nonzero(times >= bar_tick_end)[0]

    inference_config = InferenceConfig(
        {
            track_idx: [
                (
                    bar_idx_infill_start,
                    bar_idx_infill_end,
                    [],
                )
            ],
        },
        [],
    )

    res = generate(
                model,
                tokenizer,
                inference_config,
                midi,
                {"generation_config": gen_config},
            )

    res.dump_midi(midi_out)

