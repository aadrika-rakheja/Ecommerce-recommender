"""Evaluation metrics for recommendation systems."""

import logging
from typing import Dict, List, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def precision_at_k(
    recommendations: List[int],
    relevant_items: Set[int],
    k: int = 10,
) -> float:
    """
    Calculate Precision@K.

    Args:
        recommendations: List of recommended item IDs (in order).
        relevant_items: Set of truly relevant item IDs.
        k: Number of top recommendations to consider.

    Returns:
        Precision@K score (0 to 1).
    """
    if k <= 0:
        return 0.0

    top_k = recommendations[:k]
    hits = sum(1 for item in top_k if item in relevant_items)
    return hits / k if k > 0 else 0.0


def recall_at_k(
    recommendations: List[int],
    relevant_items: Set[int],
    k: int = 10,
) -> float:
    """
    Calculate Recall@K.

    Args:
        recommendations: List of recommended item IDs (in order).
        relevant_items: Set of truly relevant item IDs.
        k: Number of top recommendations to consider.

    Returns:
        Recall@K score (0 to 1).
    """
    if len(relevant_items) == 0:
        return 0.0

    top_k = recommendations[:k]
    hits = sum(1 for item in top_k if item in relevant_items)
    return hits / len(relevant_items)


def ndcg_at_k(
    recommendations: List[int],
    relevant_items: Set[int],
    k: int = 10,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).

    Args:
        recommendations: List of recommended item IDs (in order).
        relevant_items: Set of truly relevant item IDs.
        k: Number of top recommendations to consider.

    Returns:
        NDCG@K score (0 to 1).
    """
    if len(relevant_items) == 0:
        return 0.0

    # Calculate DCG
    top_k = recommendations[:k]
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)  # i + 2 because log2(1) = 0

    # Calculate IDCG (ideal DCG)
    ideal_k = min(k, len(relevant_items))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(
    predictions: Dict[int, List[int]],
    ground_truth: Dict[int, Set[int]],
    k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Evaluate recommendations across multiple users and metrics.

    Args:
        predictions: Dict mapping user_id to list of recommended item IDs.
        ground_truth: Dict mapping user_id to set of relevant item IDs.
        k_values: List of k values to evaluate (default: [5, 10, 20]).

    Returns:
        Dict with average metrics across all users.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    logger.info(f"Evaluating recommendations for {len(predictions)} users")

    metrics = {f"precision@{k}": [] for k in k_values}
    metrics.update({f"recall@{k}": [] for k in k_values})
    metrics.update({f"ndcg@{k}": [] for k in k_values})

    evaluated_users = 0

    for user_id, recommended_items in predictions.items():
        # Skip users without ground truth
        if user_id not in ground_truth:
            continue

        relevant_items = ground_truth[user_id]

        # Skip users with no relevant items
        if len(relevant_items) == 0:
            continue

        evaluated_users += 1

        for k in k_values:
            p_at_k = precision_at_k(recommended_items, relevant_items, k)
            r_at_k = recall_at_k(recommended_items, relevant_items, k)
            ndcg = ndcg_at_k(recommended_items, relevant_items, k)

            metrics[f"precision@{k}"].append(p_at_k)
            metrics[f"recall@{k}"].append(r_at_k)
            metrics[f"ndcg@{k}"].append(ndcg)

    # Calculate averages
    results = {}
    for metric_name, values in metrics.items():
        if values:
            results[metric_name] = np.mean(values)
            results[f"{metric_name}_std"] = np.std(values)
        else:
            results[metric_name] = 0.0
            results[f"{metric_name}_std"] = 0.0

    results["evaluated_users"] = evaluated_users

    logger.info(f"Evaluated {evaluated_users} users")
    for k in k_values:
        logger.info(
            f"  Precision@{k}: {results[f'precision@{k}']:.4f} "
            f"(±{results[f'precision@{k}_std']:.4f})"
        )
        logger.info(
            f"  Recall@{k}: {results[f'recall@{k}']:.4f} "
            f"(±{results[f'recall@{k}_std']:.4f})"
        )
        logger.info(
            f"  NDCG@{k}: {results[f'ndcg@{k}']:.4f} "
            f"(±{results[f'ndcg@{k}_std']:.4f})"
        )

    return results


def prepare_train_test_split(
    interactions: pd.DataFrame,
    test_ratio: float = 0.2,
    random_seed: int = 42,
    time_aware: bool = True,
    min_interactions: int = 5,
) -> tuple:
    """
    Prepare train/test split for recommendation evaluation.

    For time-aware splitting, the latest interactions of each eligible user
    are placed in the test set.

    Users with fewer than `min_interactions` are excluded because they do not
    provide enough history for meaningful train/test evaluation.

    Args:
        interactions: DataFrame with visitorid, itemid and last_interaction.
        test_ratio: Fraction of each user's interactions for test.
        random_seed: Random seed for reproducibility.
        time_aware: If True, latest interactions are used for test.
        min_interactions: Minimum interactions required per user.

    Returns:
        Tuple of (train_interactions, test_interactions).
    """

    logger.info("Preparing train/test split")

    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    interactions = interactions.copy()

    # ---------------------------------------------------------
    # Remove users with too few interactions
    # ---------------------------------------------------------
    user_counts = interactions.groupby("visitorid")["itemid"].transform("size")

    eligible_mask = user_counts >= min_interactions
    interactions = interactions[eligible_mask].copy()

    logger.info(
        f"Users/interactions after minimum-history filtering: "
        f"{len(interactions)} interactions"
    )

    # ---------------------------------------------------------
    # Time-aware split
    # ---------------------------------------------------------
    if time_aware:

        # Sort by user and time
        interactions = interactions.sort_values(
            ["visitorid", "last_interaction"]
        ).reset_index(drop=True)

        # Number of interactions for each user
        user_counts = interactions.groupby("visitorid")["itemid"].transform(
            "size"
        )

        # Number of test interactions for each user
        n_test = np.floor(user_counts * test_ratio).astype(int)

        # At least one test interaction
        n_test = np.maximum(n_test, 1)

        # Number interactions from the END of each user's history
        # 0 = latest interaction
        # 1 = second latest
        # 2 = third latest, ...
        reverse_rank = interactions.groupby("visitorid").cumcount(
            ascending=False
        )

        # Latest n_test interactions → test
        test_mask = reverse_rank < n_test

        test = interactions[test_mask].copy()
        train = interactions[~test_mask].copy()

    # ---------------------------------------------------------
    # Random split
    # ---------------------------------------------------------
    else:

        np.random.seed(random_seed)

        # Generate random numbers for each interaction
        random_values = np.random.rand(len(interactions))

        # Random test selection
        test_mask = random_values < test_ratio

        test = interactions[test_mask].copy()
        train = interactions[~test_mask].copy()

    # Reset indices
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    logger.info(
        f"Train: {len(train)} interactions, "
        f"Test: {len(test)} interactions"
    )

    return train, test


def prepare_ground_truth(interactions: pd.DataFrame) -> Dict[int, Set[int]]:
    """
    Prepare ground truth dict from interactions.

    Args:
        interactions: DataFrame with columns [visitorid, itemid, ...].

    Returns:
        Dict mapping user_id to set of item IDs they interacted with.
    """
    ground_truth = {}
    for user_id, group in interactions.groupby("visitorid"):
        ground_truth[user_id] = set(group["itemid"].unique())

    return ground_truth


__all__ = [
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "evaluate_recommendations",
    "prepare_train_test_split",
    "prepare_ground_truth",
]
