"""Item feature engineering."""

import logging
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data.loader import (
    get_project_root,
    load_category_tree,
    save_processed_data,
)

logger = logging.getLogger(__name__)


class ItemFeatureEngineer:
    """Extract and engineer features for items."""

    def __init__(self, min_property_freq: int = 5):
        """
        Initialize feature engineer.

        Args:
            min_property_freq: Minimum frequency of property values to include.
        """
        self.min_property_freq = min_property_freq
        self.category_tree = None
        self.property_encoder = {}
        self.all_features = None
        self.n_property_value_features = 0
        self.n_property_value_features_retained = 0

    def load_category_hierarchy(self) -> None:
        """Load and process category hierarchy."""
        logger.info("Loading category hierarchy")
        self.category_tree = load_category_tree()
        logger.info(f"Loaded {len(self.category_tree)} categories")

    def _get_category_parents(self, catid: int, max_depth: int = 5) -> Set[int]:
        """
        Get all parent categories for a given category.

        Args:
            catid: Category ID.
            max_depth: Maximum depth to traverse.

        Returns:
            Set of parent category IDs.
        """
        if self.category_tree is None:
            return set()

        parents = set()
        current = catid
        depth = 0

        while depth < max_depth:
            parent_rows = self.category_tree[self.category_tree["categoryid"] == current]
            if parent_rows.empty:
                break

            parent_id = parent_rows.iloc[0]["parentid"]
            if pd.isna(parent_id):
                break

            parent_id = int(parent_id)
            parents.add(parent_id)
            current = parent_id
            depth += 1

        return parents

    def extract_item_features(self, max_categories: int = 1000) -> pd.DataFrame:
        """
        Extract features from item properties.

        Uses category ID as primary feature and selected property values.

        Args:
            max_categories: Maximum categories to include in features.

        Returns:
            DataFrame with item features (sparse representation compatible).
        """
        logger.info("Extracting item features from properties")

        # Only item/category summary data is persisted; raw property/value
        # features are assembled as a sparse matrix when fitting the model.
        root = get_project_root()
        properties = pd.concat(
            [
                pd.read_csv(path, usecols=["itemid", "property", "value"])
                for path in (
                    root / "data" / "raw" / "item_properties_part1.csv",
                    root / "data" / "raw" / "item_properties_part2.csv",
                )
            ],
            ignore_index=True,
        )
        logger.info("Loaded %s item property records", len(properties))

        # Extract category IDs
        category_props = properties[properties["property"] == "categoryid"].copy()
        category_props["value"] = pd.to_numeric(
            category_props["value"], errors="coerce"
        )
        category_props = category_props.dropna(subset=["value"])

        # Get unique items and their categories
        item_categories = (
            category_props.groupby("itemid")["value"]
            .first()
            .reset_index()
            .rename(columns={"value": "categoryid"})
        )

        item_ids = properties[["itemid"]].drop_duplicates()
        item_features = item_ids.merge(item_categories, on="itemid", how="left")
        item_features["categoryid"] = item_features["categoryid"].astype("Int64")
        logger.info("Found categories for %s items", item_categories.shape[0])

        # Filter to top categories
        top_categories = (
            item_features["categoryid"]
            .dropna()
            .value_counts()
            .head(max_categories)
            .index.tolist()
        )
        logger.info(f"Using {len(top_categories)} most common categories")

        # Keep the compact item summary backward-compatible.
        item_features["category_in_top"] = item_features["categoryid"].isin(
            top_categories
        )

        # Add the legacy summary column; the matrix uses actual parent IDs.
        if self.category_tree is not None:
            category_parent_presence = {
                category_id: bool(self._get_category_parents(category_id))
                for category_id in item_features["categoryid"].dropna().unique()
            }
            item_features["has_parent"] = (
                item_features["categoryid"]
                .map(category_parent_presence)
                .fillna(False)
                .astype("int8")
            )
        else:
            item_features["has_parent"] = 0

        # Extract property frequency features
        # Count distinct property names per item for the compact summary only.
        property_counts = (
            properties.groupby("itemid")["property"]
            .nunique()
            .reset_index()
            .rename(columns={"property": "num_properties"})
        )
        item_features = item_features.merge(property_counts, on="itemid", how="left")
        item_features["num_properties"] = item_features["num_properties"].fillna(0)
        item_features = item_features.sort_values("itemid").reset_index(drop=True)

        # Log statistics
        logger.info("Created features for %s items", len(item_features))
        logger.info(f"Feature columns: {item_features.columns.tolist()}")

        return item_features

    def create_item_feature_matrix(
        self, item_features: pd.DataFrame
    ) -> csr_matrix:
        """
        Create sparse feature matrix from item features.

        Args:
            item_features: DataFrame with item features.

        Returns:
            Sparse matrix (items x features).
        """
        logger.info("Creating sparse item feature matrix")

        item_features = item_features.drop_duplicates("itemid").reset_index(drop=True)
        item_ids = item_features["itemid"].to_numpy()
        item_id_to_row = {item_id: row for row, item_id in enumerate(item_ids)}
        n_items = len(item_ids)

        root = get_project_root()
        property_paths = (
            root / "data" / "raw" / "item_properties_part1.csv",
            root / "data" / "raw" / "item_properties_part2.csv",
        )
        chunksize = 250_000
        feature_frequencies: Dict[Tuple[str, str], int] = {}

        # First pass counts property/value observations without retaining the
        # full raw dataset in memory.
        for path in property_paths:
            for chunk in pd.read_csv(
                path,
                usecols=["itemid", "property", "value"],
                dtype={"itemid": "int32", "property": "string", "value": "string"},
                keep_default_na=False,
                chunksize=chunksize,
            ):
                property_rows = chunk[chunk["property"] != "categoryid"]
                if property_rows.empty:
                    continue
                counts = property_rows.groupby(
                    ["property", "value"], sort=False, observed=True
                ).size()
                for (property_name, value), count in counts.items():
                    key = (str(property_name), str(value))
                    feature_frequencies[key] = (
                        feature_frequencies.get(key, 0) + int(count)
                    )

        self.n_property_value_features = len(feature_frequencies)
        retained_features = sorted(
            feature
            for feature, frequency in feature_frequencies.items()
            if frequency >= self.min_property_freq
        )
        self.n_property_value_features_retained = len(retained_features)
        self.property_encoder = {
            feature: idx for idx, feature in enumerate(retained_features)
        }
        logger.info("Number of items: %s", n_items)
        logger.info(
            "Unique property/value features: %s; retained after frequency filtering: %s (min frequency=%s)",
            self.n_property_value_features,
            self.n_property_value_features_retained,
            self.min_property_freq,
        )

        property_vocab = pd.DataFrame(
            (
                (property_name, value, feature_idx)
                for (property_name, value), feature_idx in self.property_encoder.items()
            ),
            columns=["property", "value", "feature_idx"],
        )

        # Property features occupy the first columns.
        row_chunks = []
        col_chunks = []
        data_chunks = []
        item_id_filter = set(item_id_to_row)
        for path in property_paths:
            for chunk in pd.read_csv(
                path,
                usecols=["itemid", "property", "value"],
                dtype={"itemid": "int32", "property": "string", "value": "string"},
                keep_default_na=False,
                chunksize=chunksize,
            ):
                property_rows = chunk[
                    (chunk["property"] != "categoryid")
                    & chunk["itemid"].isin(item_id_filter)
                ].drop_duplicates(["itemid", "property", "value"])
                if property_rows.empty or property_vocab.empty:
                    continue

                encoded = property_rows.merge(
                    property_vocab,
                    on=["property", "value"],
                    how="inner",
                    sort=False,
                )
                if encoded.empty:
                    continue
                encoded = encoded.drop_duplicates(["itemid", "feature_idx"])
                row_chunks.append(
                    encoded["itemid"].map(item_id_to_row).to_numpy(dtype=np.int32)
                )
                col_chunks.append(encoded["feature_idx"].to_numpy(dtype=np.int32))
                data_chunks.append(np.ones(len(encoded), dtype=np.float32))

        category_values = pd.to_numeric(
            item_features["categoryid"], errors="coerce"
        )
        categories = sorted(int(value) for value in category_values.dropna().unique())

        if self.category_tree is None:
            self.load_category_hierarchy()
        category_parents = {
            category: self._get_category_parents(category)
            for category in categories
        }
        parent_categories = sorted(
            {parent for parents in category_parents.values() for parent in parents}
        )

        category_offset = len(self.property_encoder)
        category_to_col = {
            category: category_offset + idx
            for idx, category in enumerate(categories)
        }
        parent_offset = category_offset + len(category_to_col)
        parent_to_col = {
            parent: parent_offset + idx
            for idx, parent in enumerate(parent_categories)
        }

        valid_categories = category_values.notna().to_numpy()
        if valid_categories.any():
            category_rows = np.flatnonzero(valid_categories).astype(np.int32)
            category_columns = np.fromiter(
                (
                    category_to_col[int(value)]
                    for value in category_values[valid_categories]
                ),
                dtype=np.int32,
                count=int(valid_categories.sum()),
            )
            row_chunks.append(category_rows)
            col_chunks.append(category_columns)
            data_chunks.append(np.ones(len(category_rows), dtype=np.float32))

            parent_rows = []
            parent_columns = []
            for row, category in zip(category_rows, category_values[valid_categories]):
                for parent in category_parents[int(category)]:
                    parent_rows.append(row)
                    parent_columns.append(parent_to_col[parent])
            if parent_rows:
                row_chunks.append(np.asarray(parent_rows, dtype=np.int32))
                col_chunks.append(np.asarray(parent_columns, dtype=np.int32))
                data_chunks.append(np.full(len(parent_rows), 0.5, dtype=np.float32))

        n_features = parent_offset + len(parent_to_col)
        if row_chunks:
            rows = np.concatenate(row_chunks)
            columns = np.concatenate(col_chunks)
            values = np.concatenate(data_chunks)
            feature_matrix = csr_matrix(
                (values, (rows, columns)),
                shape=(n_items, n_features),
                dtype=np.float32,
            )
            feature_matrix.sum_duplicates()
            feature_matrix.data[:] = 1.0
            feature_matrix.data[feature_matrix.indices >= parent_offset] = 0.5
        else:
            feature_matrix = csr_matrix((n_items, n_features), dtype=np.float32)

        total_cells = n_items * n_features
        sparsity = (
            1.0 - feature_matrix.nnz / total_cells if total_cells else 0.0
        )
        logger.info(
            "Feature matrix shape: %s (items x features)", feature_matrix.shape
        )
        logger.info("Feature matrix non-zero values: %s", feature_matrix.nnz)
        logger.info("Feature matrix sparsity: %.2f%%", sparsity * 100)

        return feature_matrix

    def fit_and_save(self) -> pd.DataFrame:
        """
        Extract features and save to disk.

        Returns:
            DataFrame with item features.
        """
        self.load_category_hierarchy()
        item_features = self.extract_item_features()

        # Save to parquet
        save_processed_data(item_features, "item_features")

        logger.info("Item features saved successfully")
        return item_features


def create_and_save_item_features() -> pd.DataFrame:
    """
    Main entry point to create and save item features.

    Returns:
        DataFrame with item features.
    """
    logger.info("Starting item feature engineering")
    engineer = ItemFeatureEngineer()
    item_features = engineer.fit_and_save()
    return item_features


__all__ = ["ItemFeatureEngineer", "create_and_save_item_features"]
