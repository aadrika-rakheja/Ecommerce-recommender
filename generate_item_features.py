"""Utility script to generate and test item features."""

import logging
import sys

# Add project to path
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.features.item_features import create_and_save_item_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

if __name__ == "__main__":
    print("Generating item features...")
    item_features = create_and_save_item_features()
    print(f"[OK] Generated features for {len(item_features)} items")
    print(f"[OK] Shape: {item_features.shape}")
    print(f"[OK] Columns: {item_features.columns.tolist()}")
