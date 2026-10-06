#!/usr/bin/env python3
"""
E-commerce Recommendation System - Command Reference

This file documents all commands needed to run the system.
Run from project root: ecommerce-recommender/
"""

# ============================================================================
# SETUP & INSTALLATION
# ============================================================================

# Activate virtual environment (Windows)
# .\.venv\Scripts\Activate.ps1

# Activate virtual environment (macOS/Linux)
# source .venv/bin/activate

# Install dependencies (one-time)
# pip install -r requirements.txt


# ============================================================================
# QUICK VERIFICATION (1 minute)
# ============================================================================

# Run comprehensive test suite - verifies both models work
# python test_models.py
# Expected: [OK] Both models pass

# This tests:
# ✓ Content-Based Recommender - 417K items fitted
# ✓ Collaborative Filtering - 117K users fitted
# ✓ Single recommendations
# ✓ Batch recommendations
# ✓ Cold-start handling


# ============================================================================
# FEATURE GENERATION (6 minutes, run once)
# ============================================================================

# Generate item features from raw properties
# python generate_item_features.py
# 
# Generates: data/processed/item_features.parquet
# Contains: 417K items with 1226 features
# Output: [OK] Generated features for 417053 items


# ============================================================================
# VIEW DEMO NOTEBOOK (10-20 minutes)
# ============================================================================

# Start Jupyter notebook server
# jupyter notebook

# Then:
# 1. Open notebooks/02_initial_models.ipynb
# 2. Run all cells (Shift+Enter or Run All)
# 3. View:
#    - Model training output
#    - Example recommendations
#    - Evaluation metrics
#    - Comparative analysis
#    - Model limitations


# ============================================================================
# PYTHON API - CONTENT-BASED FILTERING
# ============================================================================

"""
from src.models.content_based import ContentBasedRecommender
from src.data.loader import load_item_features

# Initialize model
model = ContentBasedRecommender(n_neighbors=100)

# Train on item features
item_features = load_item_features()
model.fit(item_features)

# Get recommendations for single item
recommendations = model.recommend(
    item_id=12345,
    k=10,
    exclude_items={12345}  # Don't recommend same item
)
# Returns: [(item_id, similarity_score), ...]
# Example: [(5656, 1.0), (5828, 1.0), (5858, 0.99)]

# Get recommendations for multiple items
batch_results = model.recommend_batch(
    item_ids=[12345, 67890, 11111],
    k=5
)
# Returns: {item_id: [(rec_item, score), ...]}

# Get model info
info = model.get_model_info()
# Returns: {
#   'status': 'fitted',
#   'n_items': 417053,
#   'feature_matrix_shape': (417053, 1226),
#   'feature_matrix_sparsity': 0.9984,
#   'n_neighbors_used': 100
# }
"""


# ============================================================================
# PYTHON API - COLLABORATIVE FILTERING
# ============================================================================

"""
from src.models.collaborative import CollaborativeFilteringRecommender
from src.data.loader import load_interactions

# Initialize model
model = CollaborativeFilteringRecommender(min_interactions=2)

# Train on interactions
interactions = load_interactions()
model.fit(interactions)

# Get recommendations for single user
recommendations = model.recommend(
    user_id=67890,
    k=10,
    exclude_items=set(),
    n_similar_users=20
)
# Returns: [(item_id, score), ...]
# Example: [(285930, 15.23), (357564, 10.08)]

# Get recommendations for multiple users
batch_results = model.recommend_batch(
    user_ids=[67890, 12345, 54321],
    k=5
)
# Returns: {user_id: [(item_id, score), ...]}

# Get model info
info = model.get_model_info()
# Returns: {
#   'status': 'fitted',
#   'n_users': 117529,
#   'n_items': 107745,
#   'matrix_shape': (117529, 107745),
#   'sparsity': 0.99996,
#   'min_interactions': 2
# }
"""


# ============================================================================
# PYTHON API - EVALUATION METRICS
# ============================================================================

"""
from src.evaluation.metrics import (
    prepare_train_test_split,
    prepare_ground_truth,
    evaluate_recommendations,
    precision_at_k,
    recall_at_k,
    ndcg_at_k
)
from src.data.loader import load_interactions

# Prepare train/test split (time-aware)
interactions = load_interactions()
train, test = prepare_train_test_split(
    interactions,
    test_ratio=0.2,
    random_seed=42,
    time_aware=True
)
# Returns: (train_df, test_df)

# Prepare ground truth
test_ground_truth = prepare_ground_truth(test)
# Returns: {user_id: {set of item_ids}, ...}

# Evaluate recommendations
predictions = {
    user_id: [item1, item2, ...],  # List of recommended items
    ...
}

metrics = evaluate_recommendations(
    predictions,
    test_ground_truth,
    k_values=[5, 10, 20]
)
# Returns: {
#   'precision@5': 0.15,
#   'precision@5_std': 0.08,
#   'recall@5': 0.22,
#   'recall@5_std': 0.12,
#   'ndcg@5': 0.18,
#   'ndcg@5_std': 0.10,
#   ...
#   'evaluated_users': 5000
# }

# Single metric calculation
p_at_10 = precision_at_k(
    recommendations=[item1, item2, ...],
    relevant_items={relevant_item1, relevant_item2, ...},
    k=10
)
# Returns: 0.3 (30% of top-10 were relevant)

r_at_10 = recall_at_k(
    recommendations=[item1, item2, ...],
    relevant_items={relevant_item1, relevant_item2, ...},
    k=10
)
# Returns: 0.5 (found 50% of all relevant items)

ndcg = ndcg_at_k(
    recommendations=[item1, item2, ...],
    relevant_items={relevant_item1, relevant_item2, ...},
    k=10
)
# Returns: 0.65 (65% of ideal ranking)
"""


# ============================================================================
# DATA LOADING API
# ============================================================================

"""
from src.data.loader import (
    get_project_root,
    load_events,
    load_interactions,
    load_item_features,
    load_category_tree,
    load_item_properties,
    save_processed_data
)

# Get project root
root = get_project_root()
# Returns: PosixPath('/home/user/ecommerce-recommender')

# Load events
events = load_events(processed=True)
# Returns: DataFrame with 2.7M events
# Columns: timestamp, visitorid, event, itemid, datetime, ...

# Load interactions
interactions = load_interactions()
# Returns: DataFrame with 2.1M aggregated interactions
# Columns: visitorid, itemid, interaction_score, interaction_count, ...

# Load item features
item_features = load_item_features()
# Returns: DataFrame with 417K items
# Columns: itemid, categoryid, category_in_top, has_parent, num_properties

# Load category tree
categories = load_category_tree()
# Returns: DataFrame with 1,669 categories
# Columns: categoryid, parentid

# Load item properties
properties = load_item_properties()
# Returns: DataFrame with 20.3M property records
# Columns: timestamp, itemid, property, value

# Save processed data
save_processed_data(
    df=my_dataframe,
    filename='my_features',
    filepath=None  # Saves to data/processed/
)
# Creates: data/processed/my_features.parquet
"""


# ============================================================================
# PROJECT DIRECTORIES
# ============================================================================

"""
Source code:         src/
  ├── data/          Data loading module
  ├── features/      Feature engineering
  ├── models/        Recommendation models
  └── evaluation/    Metrics and evaluation

Data:                data/
  ├── raw/           Original datasets (protected)
  └── processed/     Cleaned & engineered data

Notebooks:           notebooks/
  ├── 01_EDA.ipynb   Dataset analysis
  └── 02_initial_models.ipynb   Models demo

Tests:               tests/
  └── (ready for unit tests)

Documentation:       
  ├── README.md      Full guide
  ├── QUICK_START.md Quick reference
  ├── MILESTONE_1_REPORT.md  Detailed report
  └── DELIVERABLES.md   What's included
"""


# ============================================================================
# FILE SIZES & LOCATIONS
# ============================================================================

"""
Raw data:
  events.csv ...................... 92 MB
  category_tree.csv ............... ~100 KB
  item_properties_part1.csv ....... 484 MB
  item_properties_part2.csv ....... 409 MB
  Total raw:               ~985 MB

Processed data:
  events_clean.parquet ............ 55 MB
  interactions.parquet ............ 32 MB
  item_features.parquet ........... 2.8 MB
  Total processed:           ~90 MB

Source code:
  src/ (all modules) .............. ~38 KB
  Notebooks ........................ ~22 KB
  Scripts .......................... ~5 KB
  Total code:              ~65 KB
"""


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

"""
One-time operations:
  Generate item features .......... 6 minutes
  Load interactions + split ....... ~10 minutes
  Compute CF user similarity ...... ~5 seconds

Per-request operations:
  Content-Based recommendation .... <10ms
  Collaborative recommendation .... <50ms
  Batch (100 users) ............... <1 second

Memory usage:
  Content-Based model ............ ~500 MB
  Collaborative model ............ ~800 MB
  Total with data ................ ~1.4 GB
"""


# ============================================================================
# NEXT STEPS (MILESTONE 2)
# ============================================================================

"""
After completing Milestone 1:

1. Review:
   - Run: python test_models.py
   - View: notebooks/02_initial_models.ipynb
   - Read: MILESTONE_1_REPORT.md

2. Plan Milestone 2 (Advanced Models):
   - Hybrid model (content + collaborative)
   - Matrix factorization for CF
   - Deep learning approaches
   - Enhanced item features

3. Start Milestone 2 Development:
   - Implement hybrid recommender
   - Add advanced features
   - Build FastAPI backend
   - Create React frontend
"""


# ============================================================================
# HELP & DEBUGGING
# ============================================================================

"""
Check data is loaded:
  python -c "from src.data.loader import load_interactions; \
    df = load_interactions(); print(f'Loaded {len(df)} interactions')"

Verify models work:
  python test_models.py

View model predictions:
  python -c "from src.models.content_based import ContentBasedRecommender; \
    from src.data.loader import load_item_features; \
    m = ContentBasedRecommender(); m.fit(load_item_features()); \
    print(m.recommend(12345, k=5))"

Run with verbose logging:
  python -c "
  import logging
  logging.basicConfig(level=logging.DEBUG)
  # Then run your code
  "

Check Python version:
  python --version
  # Should be 3.12+

Check installed packages:
  pip list | grep -E 'pandas|numpy|scikit-learn|scipy'
"""


# ============================================================================
# COMMON WORKFLOWS
# ============================================================================

"""
Workflow 1: Test Models Work
  1. python test_models.py
  2. Should see: [OK] Both models pass
  3. Takes: ~15 minutes

Workflow 2: Generate Recommendations
  1. from src.models.content_based import ContentBasedRecommender
  2. model = ContentBasedRecommender()
  3. model.fit()
  4. recs = model.recommend(item_id, k=10)
  5. Process recs list

Workflow 3: Evaluate Models
  1. Load train/test data
  2. Train models on train data
  3. Generate predictions on test users
  4. Call evaluate_recommendations()
  5. Print metrics

Workflow 4: Explore Data
  1. jupyter notebook
  2. Open notebooks/01_EDA.ipynb
  3. Review existing analysis
  4. or Open notebooks/02_initial_models.ipynb
  5. Run all cells to see models in action
"""


# ============================================================================
# END OF COMMAND REFERENCE
# ============================================================================

# For more information:
# - README.md: Full documentation
# - QUICK_START.md: Getting started
# - DELIVERABLES.md: What's included
# - MILESTONE_1_REPORT.md: Detailed progress report
