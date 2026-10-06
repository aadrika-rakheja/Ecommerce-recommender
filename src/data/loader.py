"""Data loading and preprocessing utilities."""

import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def load_events(
    filepath: str | None = None,
    processed: bool = True,
) -> pd.DataFrame:
    """
    Load event data from CSV or Parquet.

    Args:
        filepath: Path to data file. If None, uses default location.
        processed: If True, load processed data; otherwise load raw.

    Returns:
        DataFrame with event data.
    """
    if filepath is None:
        root = get_project_root()
        if processed:
            filepath = root / "data" / "processed" / "events_clean.parquet"
        else:
            filepath = root / "data" / "raw" / "events.csv"

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info(f"Loading events from {filepath}")

    if filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif filepath.suffix == ".csv":
        df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")

    logger.info(f"Loaded {len(df)} events")
    return df


def load_interactions(filepath: str | None = None) -> pd.DataFrame:
    """
    Load aggregated user-item interactions.

    Args:
        filepath: Path to interactions file. If None, uses default location.

    Returns:
        DataFrame with aggregated interactions.
    """
    if filepath is None:
        root = get_project_root()
        filepath = root / "data" / "processed" / "interactions.parquet"

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info(f"Loading interactions from {filepath}")
    df = pd.read_parquet(filepath)
    logger.info(f"Loaded {len(df)} user-item interactions")
    return df


def load_item_features(filepath: str | None = None) -> pd.DataFrame:
    """
    Load item feature vectors.

    Args:
        filepath: Path to item features file. If None, uses default location.

    Returns:
        DataFrame with item features.
    """
    if filepath is None:
        root = get_project_root()
        filepath = root / "data" / "processed" / "item_features.parquet"

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Item features not found: {filepath}. Run feature engineering first.")

    logger.info(f"Loading item features from {filepath}")
    df = pd.read_parquet(filepath)
    logger.info(f"Loaded features for {len(df)} items")
    return df


def load_category_tree(filepath: str | None = None) -> pd.DataFrame:
    """
    Load category hierarchy.

    Args:
        filepath: Path to category tree. If None, uses default location.

    Returns:
        DataFrame with category hierarchy.
    """
    if filepath is None:
        root = get_project_root()
        filepath = root / "data" / "raw" / "category_tree.csv"

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Category tree not found: {filepath}")

    logger.info(f"Loading category tree from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} categories")
    return df


def load_item_properties(filepath: str | None = None) -> pd.DataFrame:
    """
    Load item properties from combined parts.

    Args:
        filepath: Path prefix for item properties. If None, uses default location.

    Returns:
        Concatenated DataFrame with all item properties.
    """
    if filepath is None:
        root = get_project_root()
        filepaths = [
            root / "data" / "raw" / "item_properties_part1.csv",
            root / "data" / "raw" / "item_properties_part2.csv",
        ]
    else:
        # Assume user provided prefix, e.g., "data/raw/item_properties"
        root = get_project_root()
        filepaths = [
            root / f"{filepath}_part1.csv",
            root / f"{filepath}_part2.csv",
        ]

    dfs = []
    for fp in filepaths:
        if not fp.exists():
            raise FileNotFoundError(f"Item properties file not found: {fp}")
        logger.info(f"Loading item properties from {fp}")
        dfs.append(pd.read_csv(fp))

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} item property records")
    return df


def save_processed_data(
    df: pd.DataFrame,
    filename: str,
    filepath: str | None = None,
) -> Path:
    """
    Save processed data to parquet format.

    Args:
        df: DataFrame to save.
        filename: Name of output file (without extension).
        filepath: Output directory. If None, uses processed data directory.

    Returns:
        Path to saved file.
    """
    if filepath is None:
        root = get_project_root()
        filepath = root / "data" / "processed"
    else:
        filepath = Path(filepath)

    filepath.mkdir(parents=True, exist_ok=True)
    output_file = filepath / f"{filename}.parquet"

    logger.info(f"Saving {len(df)} records to {output_file}")
    df.to_parquet(output_file, index=False)
    logger.info(f"Saved to {output_file}")
    return output_file


__all__ = [
    "get_project_root",
    "load_events",
    "load_interactions",
    "load_item_features",
    "load_category_tree",
    "load_item_properties",
    "save_processed_data",
]
