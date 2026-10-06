"""Test script to verify both recommendation models work correctly."""

import logging
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_interactions, load_item_features
from src.models.content_based import ContentBasedRecommender
from src.models.collaborative import CollaborativeFilteringRecommender
from src.evaluation.metrics import prepare_train_test_split, prepare_ground_truth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def test_content_based():
    """Test content-based recommender."""
    print("\n" + "=" * 70)
    print("TESTING CONTENT-BASED RECOMMENDER")
    print("=" * 70)

    # Load item features
    item_features = load_item_features()
    logger.info(f"Loaded item features for {len(item_features)} items")

    # Create and fit model
    cb_model = ContentBasedRecommender()
    cb_model.fit(item_features)

    # Test single recommendation
    sample_items = item_features["itemid"].head(10).tolist()
    print(f"\nTesting recommendations for sample items:")
    print(f"-" * 70)

    for item_id in sample_items[:3]:
        recommendations = cb_model.recommend(item_id, k=5)
        print(f"\nItem {item_id}:")
        print(f"  Recommended {len(recommendations)} similar items:")
        for rec_item, score in recommendations[:3]:
            print(f"    - Item {rec_item}: similarity = {score:.4f}")

    # Get model info
    info = cb_model.get_model_info()
    print(f"\n{'-' * 70}")
    print("Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print(f"\n[OK] Content-Based Recommender test PASSED")
    return cb_model


def test_collaborative_filtering():
    """Test collaborative filtering recommender."""
    print("\n" + "=" * 70)
    print("TESTING COLLABORATIVE FILTERING RECOMMENDER")
    print("=" * 70)

    # Load interactions
    interactions = load_interactions()
    logger.info(f"Loaded {len(interactions)} interactions")

    # Split data
    train_interactions, test_interactions = prepare_train_test_split(
        interactions, test_ratio=0.2, random_seed=42, time_aware=True
    )
    logger.info(f"Train: {len(train_interactions)}, Test: {len(test_interactions)}")

    # Create and fit model
    cf_model = CollaborativeFilteringRecommender(min_interactions=2)
    cf_model.fit(train_interactions)

    # Test single recommendation
    sample_users = train_interactions["visitorid"].unique()[:10].tolist()
    print(f"\nTesting recommendations for sample users:")
    print(f"-" * 70)

    for user_id in sample_users[:3]:
        recommendations = cf_model.recommend(user_id, k=5)
        print(f"\nUser {user_id}:")
        print(f"  Recommended {len(recommendations)} items:")
        for rec_item, score in recommendations[:3]:
            print(f"    - Item {rec_item}: score = {score:.4f}")

    # Test cold-start user
    print(f"\n{'-' * 70}")
    print(f"Testing cold-start user (not in training):")
    cold_start_recs = cf_model.recommend(999999999, k=5)
    print(f"Cold-start fallback recommendations: {len(cold_start_recs)} items")
    for item, score in cold_start_recs[:3]:
        print(f"  - Item {item}: score = {score:.4f}")

    # Get model info
    info = cf_model.get_model_info()
    print(f"\n{'-' * 70}")
    print("Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print(f"\n[OK] Collaborative Filtering Recommender test PASSED")
    return cf_model, train_interactions, test_interactions


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RECOMMENDATION SYSTEM TEST SUITE")
    print("=" * 70)

    try:
        # Test content-based
        cb_model = test_content_based()

        # Test collaborative filtering
        cf_model, train_interactions, test_interactions = test_collaborative_filtering()

        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        print("[OK] Content-Based Recommender: Functional")
        print("[OK] Collaborative Filtering Recommender: Functional")
        print("\nBoth models are ready for evaluation!")
        print("\nNext steps:")
        print("  1. Run: jupyter notebook notebooks/02_initial_models.ipynb")
        print("  2. This will train both models and evaluate them")
        print("  3. View detailed metrics and recommendations")

    except Exception as e:
        print(f"\n[ERROR] Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
