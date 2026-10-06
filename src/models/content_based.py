"""Content-based recommendation model."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from src.data.loader import load_item_features, load_interactions
from src.features.item_features import ItemFeatureEngineer

logger = logging.getLogger(__name__)


class ContentBasedRecommender:
    """
    Content-based recommendation model using item similarity.

    Recommends items similar to a given item based on features.
    Uses nearest neighbors for memory efficiency.
    """

    def __init__(self, n_neighbors: int = 100):
        """
        Initialize content-based recommender.

        Args:
            n_neighbors: Number of neighbors to find for each item.
        """
        self.n_neighbors = n_neighbors
        self.item_features = None
        self.feature_matrix = None
        self.knn_model = None
        self.item_id_to_idx = {}
        self.idx_to_item_id = {}
        self.feature_engineer = None

    def fit(self, item_features: pd.DataFrame | None = None) -> None:
        """
        Fit the model with item features.

        Args:
            item_features: DataFrame with item features. If None, loads from disk.
        """
        logger.info("Fitting content-based model")

        if item_features is None:
            logger.info("Loading item features from disk")
            item_features = load_item_features()
        else:
            self.item_features = item_features

        item_features = (
            item_features.drop_duplicates("itemid")
            .sort_values("itemid")
            .reset_index(drop=True)
        )
        self.item_features = item_features
        n_items = len(item_features)

        # Matrix rows follow this same item-ID order.
        item_ids = item_features["itemid"].tolist()
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        self.idx_to_item_id = {
            idx: item_id for item_id, idx in self.item_id_to_idx.items()
        }

        logger.info("Mapping %s items", len(item_ids))

        # Create feature matrix
        engineer = ItemFeatureEngineer()
        self.feature_engineer = engineer
        self.feature_matrix = engineer.create_item_feature_matrix(item_features)
        if self.feature_matrix.shape[0] != n_items:
            raise ValueError(
                "Feature matrix row count does not match the item ID mapping "
                f"({self.feature_matrix.shape[0]} != {n_items})"
            )

        # Build KNN model for efficient nearest neighbor search
        logger.info("Building KNN model for similarity search")
        # Use cosine metric for similarity
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors + 1, n_items),  # +1 to include self
            metric="cosine",
            algorithm="brute",  # Brute force is fine for sparse data
            n_jobs=-1,  # Use all cores
        )
        self.knn_model.fit(self.feature_matrix)

        logger.info(f"Model fitted with {n_items} items")

    def recommend(
        self,
        item_id: int,
        k: int = 10,
        exclude_items: set | None = None,
    ) -> List[Tuple[int, float]]:
        """
        Recommend similar items.

        Args:
            item_id: ID of query item.
            k: Number of recommendations to return.
            exclude_items: Set of item IDs to exclude from recommendations.

        Returns:
            List of (item_id, similarity_score) tuples.
        """
        if self.knn_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if exclude_items is None:
            exclude_items = set()

        # Add the query item to exclusion list
        exclude_items = exclude_items | {item_id}

        # Get item index
        if item_id not in self.item_id_to_idx:
            logger.warning(f"Item {item_id} not in training data")
            return []

        item_idx = self.item_id_to_idx[item_id]

        # Query KNN - get more neighbors to account for exclusions
        distances, indices = self.knn_model.kneighbors(
            self.feature_matrix[item_idx],
            n_neighbors=min(
                max(k * 2, k + 1),
                len(self.item_id_to_idx),
            ),
        )

        # Convert distances to similarities (cosine distance to similarity)
        distances = distances[0]
        indices = indices[0]

        # Cosine distance to similarity: similarity = 1 - distance
        similarities = 1.0 - distances

        # Create recommendations, filtering exclusions
        recommendations = []
        for idx, similarity_score in zip(indices, similarities):
            candidate_item_id = self.idx_to_item_id[idx]
            if candidate_item_id not in exclude_items and similarity_score > 0:
                recommendations.append((candidate_item_id, float(similarity_score)))

        # Return top k
        recommendations = recommendations[:k]

        logger.debug(
            f"Recommended {len(recommendations)} items for item {item_id}"
        )
        return recommendations

    def recommend_batch(
        self,
        item_ids: List[int],
        k: int = 10,
        exclude_items: set | None = None,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Recommend similar items for multiple query items.

        Args:
            item_ids: List of item IDs.
            k: Number of recommendations per item.
            exclude_items: Set of item IDs to exclude globally.

        Returns:
            Dict mapping item_id to list of recommendations.
        """
        results = {}
        for item_id in item_ids:
            results[item_id] = self.recommend(
                item_id, k=k, exclude_items=exclude_items
            )
        return results

    def get_model_info(self) -> dict:
        """
        Get model information and statistics.

        Returns:
            Dictionary with model info.
        """
        if self.item_features is None:
            return {"status": "not fitted"}

        return {
            "status": "fitted",
            "n_items": len(self.item_features),
            "feature_matrix_shape": self.feature_matrix.shape,
            "feature_matrix_sparsity": 1 - (self.feature_matrix.nnz / np.prod(self.feature_matrix.shape)),
            "n_property_value_features": (
                self.feature_engineer.n_property_value_features
            ),
            "n_property_value_features_retained": (
                self.feature_engineer.n_property_value_features_retained
            ),
            "n_neighbors_used": self.n_neighbors,
        }


__all__ = ["ContentBasedRecommender"]
