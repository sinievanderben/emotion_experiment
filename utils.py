"""Shared utilities for emotion vector analysis."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_text_file(path: str, strip: bool = True) -> List[str]:
    """Load lines from a text file."""
    with open(path, "r") as f:
        lines = [line.strip() if strip else line for line in f if line.strip()]
    return lines


def load_circumplex(csv_path: Path) -> Dict[str, Tuple[float, float]]:
    """Load valence/arousal ratings from NRC VAD CSV."""
    def normalise(s: str) -> str:
        return s.strip().lower().replace("_", " ")

    circumplex = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = normalise(row["emotion"])
            circumplex[key] = (float(row["valence"]), float(row["arousal"]))
    return circumplex


def lookup_vad(emotion: str, circumplex: Dict) -> Tuple[float, float] | None:
    """Look up valence/arousal for an emotion."""
    key = emotion.strip().lower().replace("_", " ")
    return circumplex.get(key)


def load_vad_lexicon(csv_path: Path) -> Dict[str, Tuple[float, float]]:
    """Load VAD lexicon for scoring text."""
    vad = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            vad[row["emotion"].strip().lower()] = (
                float(row["valence"]), float(row["arousal"])
            )
    return vad
