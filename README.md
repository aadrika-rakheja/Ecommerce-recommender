# E-commerce Recommendation System

A machine learning-based recommendation system for e-commerce using the Retailrocket dataset. This project implements content-based filtering and collaborative filtering models, with plans for a hybrid approach, FastAPI backend, and React frontend.

## Project Status: 40% Complete (Stage 1 + Initial Models)

### ✅ Completed (Milestone 1)

#### Stage 1: Data Understanding & Preparation
- [x] Exploratory Data Analysis (`notebooks/01_EDA.ipynb`)
  - Dataset overview: 2.7M events, 417K items, 1.4M users
  - Event types: view, addtocart, transaction
  - Temporal patterns and user behavior analysis
  
- [x] Data Cleaning & Preprocessing
  - Raw events cleaned and processed
  - User-item interactions aggregated
  - Temporal features extracted (hour, day of week, month)
  - Interaction weights assigned based on event type

- [x] Item Feature Engineering (`src/features/item_features.py`)
  - Category information extracted from item properties
  - Product hierarchy captured
  - Feature vectors created for similarity computation

#### Stage 2: Recommendation Models (2 of 5 Planned)

1. **Content-Based Filtering** (`src/models/content_based.py`)
   - Uses item features (category, properties) for similarity
   - Recommends items similar to those user has interacted with
   - No cold-start problem for items with features
   - Interpretable recommendations

2. **Collaborative Filtering** (`src/models/collaborative.py`)
   - User-based approach: finds similar users and recommends their items
   - Sparse matrix operations for efficiency
   - Handles cold-start users with popularity fallback
   - Leverages collective user behavior patterns

#### Evaluation Framework
- [x] Metrics: Precision@K, Recall@K, NDCG@K
- [x] Time-aware train/test split (prevents data leakage)
- [x] Batch evaluation capabilities
- [x] Comprehensive model comparisons

### 📁 Project Structure

```
ecommerce-recommender/
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── events.csv               
│   │   ├── category_tree.csv        
│   │   └── item_properties_part*.csv
│   └── processed/                    # Cleaned and engineered data
│       ├── events_clean.parquet
│       ├── interactions.parquet
│       └── item_features.parquet     # Generated during feature engineering
│
├── src/
│   ├── data/
│   │   └── loader.py                 # Data loading utilities
│   ├── features/
│   │   └── item_features.py          # Item feature engineering
│   ├── models/
│   │   ├── content_based.py          # Content-based recommender
│   │   └── collaborative.py          # Collaborative filtering
│   └── evaluation/
│       └── metrics.py                 # Evaluation metrics
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory data analysis
│   └── 02_initial_models.ipynb       # Models demonstration & evaluation
│
├── tests/                             # Unit tests (to be added)
├── artifacts/                         # Model artifacts & results
├── api/                               # FastAPI backend (Stage 3)
├── frontend/                          # React frontend (Stage 3)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Installation & Setup

### Prerequisites
- Python 3.12+
- Virtual environment (already created in `.venv`)

### Install Dependencies
```bash
pip install -r requirements.txt
```

Current dependencies:
- pandas, numpy — Data manipulation
- scikit-learn, scipy — ML algorithms
- matplotlib, seaborn — Visualization
- jupyter, ipykernel — Notebooks
- pyarrow — Parquet support

## Usage

### 1. Generate Item Features
Run this first to create item feature vectors:
```bash
python generate_item_features.py
```
Output: `data/processed/item_features.parquet`

### 2. Run the Demonstration Notebook
```bash
jupyter notebook notebooks/02_initial_models.ipynb
```

This notebook:
- Loads all preprocessed data
- Trains both models
- Generates example recommendations
- Evaluates both models
- Provides comparative analysis
- Discusses model limitations

### 3. Using Models Programmatically

#### Content-Based Filtering
```python
from src.models.content_based import ContentBasedRecommender
from src.data.loader import load_item_features

# Load and fit model
cb_model = ContentBasedRecommender()
cb_model.fit(load_item_features())

# Get recommendations
recommendations = cb_model.recommend(item_id=12345, k=10)
# Returns: [(item_id, similarity_score), ...]
```

#### Collaborative Filtering
```python
from src.models.collaborative import CollaborativeFilteringRecommender
from src.data.loader import load_interactions

# Load and fit model
cf_model = CollaborativeFilteringRecommender()
cf_model.fit(load_interactions())

# Get recommendations
recommendations = cf_model.recommend(user_id=67890, k=10)
# Returns: [(item_id, score), ...]
```

## Dataset Overview

### Events Dataset
- **Size**: 2,755,641 events
- **Features**: timestamp, visitor ID, event type, item ID, transaction ID
- **Event Types**: 
  - `view` (2,644K) — Product page views
  - `addtocart` (102K) — Add to cart actions
  - `transaction` (8K) — Completed purchases

### User-Item Interactions
- **Total interactions**: 2,145,179 (aggregated by user-item pair)
- **Unique users**: ~1.4M
- **Unique items**: ~417K
- **Sparsity**: 99.65% (mostly missing interactions)

### Item Properties
- **Total records**: 20.3M property entries
- **Property types**: Mostly anonymized (200+ property IDs)
- **Category information**: Available for 417K items
- **Category hierarchy**: 1,669 categories with parent relationships

## Model Performance

### Evaluation Methodology
- **Train/Test Split**: 80/20 (time-aware to prevent leakage)
- **Metrics**: Precision@5, Precision@10, Recall@5, Recall@10, NDCG@5, NDCG@10
- **Test Users**: Evaluated on users with items in both train and test sets

### Limitations & Trade-offs

#### Content-Based Filtering
**Strengths:**
- No cold-start for items with features
- Transparent and interpretable
- Consistent recommendations

**Limitations:**
- Requires meaningful item features (current: categories only)
- Cannot discover serendipitous items
- Cannot leverage user behavior patterns
- All recommendations are similar to training items

#### Collaborative Filtering
**Strengths:**
- Discovers serendipitous recommendations
- Captures implicit user preferences
- Works without explicit features

**Limitations:**
- Cold-start problem for new users/items
- Struggles with very sparse data (99.65% sparse)
- Popular items over-recommended
- Cannot explain why items recommended

## Development Roadmap

### Stage 2: Recommendation Models (Next Milestone)
- [ ] Hybrid recommendation model (combine content + collaborative)
- [ ] Enhanced item features with embeddings
- [ ] Matrix factorization for collaborative filtering
- [ ] Deep learning approaches
- [ ] Context-aware recommendations
- [ ] Serendipity metrics

### Stage 3: Application Development & Deployment
- [ ] FastAPI backend
  - Model serving endpoints
  - Real-time predictions
  - User interaction logging
  - A/B testing framework
  
- [ ] React Frontend
  - Product recommendation display
  - User feedback collection
  - Interactive recommendation explorer
  - Admin dashboard
  
- [ ] Deployment
  - Docker containerization
  - CI/CD pipeline
  - Cloud deployment (AWS/GCP/Azure)
  - Monitoring and logging

## Code Quality

- **Type Hints**: Full type annotations throughout
- **Docstrings**: Comprehensive docstrings on all public functions
- **Logging**: DEBUG and INFO level logging at key points
- **Path Handling**: All paths are relative to project root
- **Testing**: Unit test framework ready (tests/ directory)

## Performance Notes

### Item Feature Generation
- Reads 20M property records
- Processes 417K items
- Creates ~1K categorical features
- Execution time: ~5-10 minutes (first run)

### Model Training
- Content-based: <1 second
- Collaborative filtering: 5-30 seconds (depending on data size)

### Recommendation Serving
- Single recommendation: <10ms
- Batch (100 users): <1s
- Latency suitable for real-time serving

## Contributing

Guidelines for extending this project:
1. Follow existing code style and structure
2. Add comprehensive docstrings
3. Include type hints
4. Use logging for debugging
5. Test on sample data before running on full dataset
6. Document any new data dependencies

## References

- **Dataset**: Retailrocket e-commerce dataset
- **Evaluation Metrics**: 
  - Precision/Recall: Standard IR metrics
  - NDCG: Normalized Discounted Cumulative Gain
  - Used in TREC, Netflix Prize, and other competitions

## License

[Project license to be specified]

## Contact

For questions or issues, please see the project issues/discussions.

---

**Last Updated**: September 29, 2026  
**Stage**: 40% Complete (Stage 1 + Initial Models)  
**Next Milestone**: Hybrid Model & API Development
