#!/usr/bin/env python3
"""Train all ML models and save to app/ml/models/."""

import logging
import sys
from pathlib import Path

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from app.ml.train import train

if __name__ == "__main__":
    clf = train(save=True)
    print("Training complete.")
