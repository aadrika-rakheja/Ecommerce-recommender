"""Collaborative filtering recommendation model."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.data.loader import load_events, load_interactions

logger = logging.getLogger(__name__)


class CollaborativeFilteringRecommender:
    """
    User-based collaborative filtering recommender.

    Uses user-item interactions to find similar users and recommend
    items that similar users have interacted with.
    """

    def __init__(self, min_interactions: int = 2):
        """
        Initialize collaborative filtering recommender.

        Args:
            min_interactions: Minimum interactions per user to include in CF.
        """
        self.min_interactions = min_interactions
        self.interaction_matrix = None
        self.user_similarity = None
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        self.item_id_to_idx = {}
        self.idx_to_item_id = {}
        self.interaction_df = None
        self.mean_item_score = None

    def _build_interaction_matrix(
        self, interactions: pd.DataFrame
    ) -> Tuple[csr_matrix, Dict, Dict, Dict, Dict]:
        """
        Build sparse user-item interaction matrix.

        Args:
            interactions: DataFrame with columns [visitorid, itemid, interaction_score].

        Returns:
            Tuple of (matrix, user_mapping, item_mapping, reverse_user_mapping, reverse_item_mapping).
        """
        logger.info("Building interaction matrix")

        # Filter users with minimum interactions
        user_interaction_counts = interactions.groupby("visitorid").size()
        valid_users = user_interaction_counts[
            user_interaction_counts >= self.min_interactions
        ].index

        filtered_interactions = interactions[
            interactions["visitorid"].isin(valid_users)
        ].copy()

        logger.info(
            f"Filtered to {len(valid_users)} users with >= {self.min_interactions} interactions"
        )

        # Get unique user and item IDs
        user_ids = sorted(filtered_interactions["visitorid"].unique())
        item_ids = sorted(filtered_interactions["itemid"].unique())

        # Create mappings
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        idx_to_user_id = {idx: user_id for user_id, idx in user_id_to_idx.items()}
        item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        idx_to_item_id = {idx: item_id for item_id, idx in item_id_to_idx.items()}

        logger.info(f"User IDs: {len(user_ids)}, Item IDs: {len(item_ids)}")

        # Build sparse matrix
        matrix = lil_matrix(
            (len(user_ids), len(item_ids)), dtype=np.float32
        )

        for _, row in filtered_interactions.iterrows():
            user_idx = user_id_to_idx[row["visitorid"]]
            item_idx = item_id_to_idx[row["itemid"]]
            # Use interaction score as weight
            matrix[user_idx, item_idx] = float(row["interaction_score"])

        # Convert to CSR for efficient operations
        matrix = matrix.tocsr()
        logger.info(
            f"Interaction matrix shape: {matrix.shape}, "
            f"Sparsity: {1 - matrix.nnz / np.prod(matrix.shape):.2%}"
        )

        return matrix, user_id_to_idx, item_id_to_idx, idx_to_user_id, idx_to_item_id

    def fit(self, interactions: pd.DataFrame | None = None) -> None:
        """
        Fit the model with interaction data.

        Args:
            interactions: DataFrame with columns [visitorid, itemid, interaction_score].
                         If None, loads from disk.
        """
        logger.info("Fitting collaborative filtering model")

        if interactions is None:
            logger.info("Loading interactions from disk")
            interactions = load_interactions()
        else:
            self.interaction_df = interactions

        self.interaction_df = interactions

        # Build interaction matrix
        (
            self.interaction_matrix,
            self.user_id_to_idx,
            self.item_id_to_idx,
            self.idx_to_user_id,
            self.idx_to_item_id,
        ) = self._build_interaction_matrix(interactions)

        # Normalize interaction matrix row-wise (by user)
        logger.info("Normalizing interaction matrix")
        normalized_matrix = self.interaction_matrix.copy().astype(np.float32)

        # L2 normalization per user
        row_norms = np.array(
            np.sqrt(normalized_matrix.multiply(normalized_matrix).sum(axis=1))
        ).flatten()
        row_norms[row_norms == 0] = 1  # Avoid division by zero

        normalized_matrix = normalized_matrix.tocsr()
        for i in range(normalized_matrix.shape[0]):
            normalized_matrix.data[
                normalized_matrix.indptr[i] : normalized_matrix.indptr[i + 1]
            ] /= row_norms[i]

        # Compute user similarity
        logger.info("Computing user similarity matrix")
        user_similarity = cosine_similarity(normalized_matrix, dense_output=False)
        self.user_similarity = user_similarity

        # Compute mean item score for fallback recommendations
        mean_scores = np.array(self.interaction_matrix.mean(axis=0)).flatten()
        self.mean_item_score = {
            self.idx_to_item_id[idx]: score
            for idx, score in enumerate(mean_scores)
        }

        logger.info("Collaborative filtering model fitted")

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        exclude_items: set | None = None,
        n_similar_users: int = 20,
    ) -> List[Tuple[int, float]]:
        """
        Recommend items for a user based on similar users.

        Args:
            user_id: ID of the user.
            k: Number of recommendations.
            exclude_items: Set of item IDs to exclude.
            n_similar_users: Number of similar users to consider.

        Returns:
            List of (item_id, score) tuples.
        """
        if self.user_similarity is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if exclude_items is None:
            exclude_items = set()

        # Handle cold-start users
        if user_id not in self.user_id_to_idx:
            logger.debug(f"User {user_id} is cold-start. Returning popular items.")
            return self._popular_items_fallback(k, exclude_items)

        user_idx = self.user_id_to_idx[user_id]

        # Get similar users
        similarities = self.user_similarity[user_idx].toarray().flatten()

        # Find top similar users (excluding self)
        similar_user_indices = np.argsort(-similarities)[
            1 : n_similar_users + 1
        ]  # Skip self (index 0)

        # Aggregate recommendations from similar users
        recommendations_scores = {}

        for sim_user_idx in similar_user_indices:
            sim_user_id = self.idx_to_user_id[sim_user_idx]
            sim_score = similarities[sim_user_idx]

            # Get items interacted with by similar user
            user_interactions = self.interaction_matrix[sim_user_idx].toarray().flatten()
            interacted_items = np.where(user_interactions > 0)[0]

            for item_idx in interacted_items:
                item_id = self.idx_to_item_id[item_idx]

                # Skip excluded items
                if item_id in exclude_items:
                    continue

                # Aggregate score: interaction strength * user similarity
                aggregated_score = user_interactions[item_idx] * sim_score

                if item_id not in recommendations_scores:
                    recommendations_scores[item_id] = aggregated_score
                else:
                    recommendations_scores[item_id] += aggregated_score

        # Sort by score and return top k
        recommendations = sorted(
            recommendations_scores.items(), key=lambda x: x[1], reverse=True
        )[:k]

        if not recommendations:
            logger.debug(f"No recommendations from similar users. Using popular items.")
            recommendations = self._popular_items_fallback(k, exclude_items | {user_id})

        return recommendations

    def _popular_items_fallback(
        self, k: int, exclude_items: set
    ) -> List[Tuple[int, float]]:
        """
        Return popular items as fallback (for cold-start users).

        Args:
            k: Number of items.
            exclude_items: Items to exclude.

        Returns:
            List of (item_id, score) tuples.
        """
        popular = sorted(
            [
                (item_id, score)
                for item_id, score in self.mean_item_score.items()
                if item_id not in exclude_items
            ],
            key=lambda x: x[1],
            reverse=True,
        )[:k]
        return popular

    def recommend_batch(
        self,
        user_ids: List[int],
        k: int = 10,
        exclude_items: set | None = None,
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Recommend items for multiple users.

        Args:
            user_ids: List of user IDs.
            k: Number of recommendations per user.
            exclude_items: Set of item IDs to exclude globally.

        Returns:
            Dict mapping user_id to recommendations.
        """
        results = {}
        for user_id in user_ids:
            results[user_id] = self.recommend(user_id, k=k, exclude_items=exclude_items)
        return results

    def get_model_info(self) -> dict:
        """
        Get model information and statistics.

        Returns:
            Dictionary with model info.
        """
        if self.interaction_matrix is None:
            return {"status": "not fitted"}

        return {
            "status": "fitted",
            "n_users": self.interaction_matrix.shape[0],
            "n_items": self.interaction_matrix.shape[1],
            "matrix_shape": self.interaction_matrix.shape,
            "sparsity": 1
            - (
                self.interaction_matrix.nnz
                / np.prod(self.interaction_matrix.shape)
            ),
            "min_interactions": self.min_interactions,
        }


__all__ = ["CollaborativeFilteringRecommender"]
