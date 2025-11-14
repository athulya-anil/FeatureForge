# FeatureForge

**Production-Grade Feature Experimentation Platform for CTR Prediction**

A comprehensive ML platform for Click-Through Rate (CTR) prediction with A/B testing capabilities for feature experimentation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Phase 1: Baseline Model](#phase-1-baseline-model)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

FeatureForge is an end-to-end feature engineering and experimentation platform designed for CTR prediction. It implements:

- **Baseline Features**: 15-20 fundamental features serving as the CONTROL group
- **Experimental Features** (Phase 2): 70+ advanced features for A/B testing
- **Statistical Testing**: Rigorous comparison between feature sets
- **Production Pipeline**: PySpark-based scalable data processing
- **Comprehensive Evaluation**: Multiple metrics and visualizations

### Why FeatureForge?

- **Resume-Ready**: Demonstrates production ML engineering skills
- **Scalable**: PySpark handles datasets of any size
- **Scientific**: A/B testing with statistical significance
- **Complete**: End-to-end pipeline from raw data to deployed model

---

## ✨ Features

### Phase 1 (Current)
- ✅ Production-grade project structure
- ✅ PySpark data loading and processing
- ✅ 15-20 baseline features (missing indicators, count encoding, target encoding)
- ✅ XGBoost/LightGBM training with class imbalance handling
- ✅ Comprehensive evaluation framework
- ✅ Feature importance analysis
- ✅ Jupyter notebooks for EDA and modeling

### Phase 2 (Upcoming)
- 🔜 70+ experimental features
- 🔜 A/B testing framework
- 🔜 Statistical significance testing
- 🔜 Feature selection algorithms
- 🔜 Model deployment pipeline

---

## 📁 Project Structure

```
featureforge/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
├── .gitignore                     # Git ignore rules
│
├── config/
│   └── config.yaml               # Configuration settings
│
├── data/
│   ├── raw/                      # Raw Criteo data (gitignored)
│   ├── processed/                # Processed parquet files
│   └── sample/                   # 1M row sample for testing
│
├── notebooks/
│   ├── 01_data_download.ipynb    # Data download guide
│   ├── 02_eda.ipynb              # Exploratory Data Analysis
│   └── 03_baseline_model.ipynb   # Baseline model training
│
├── src/
│   ├── config.py                 # Configuration management
│   ├── data/
│   │   ├── loader.py            # PySpark data loading
│   │   └── splitter.py          # Train/val/test split
│   ├── features/
│   │   ├── base_features.py     # Baseline features (15-20)
│   │   └── feature_engine.py    # Feature pipeline orchestrator
│   ├── models/
│   │   ├── trainer.py           # XGBoost/LightGBM training
│   │   └── evaluator.py         # Comprehensive evaluation
│   └── utils/
│       ├── spark_utils.py       # PySpark utilities
│       └── logging_utils.py     # Logging configuration
│
├── scripts/
│   ├── download_data.sh         # Download Criteo dataset
│   └── create_sample.py         # Create 1M row sample
│
├── tests/
│   └── test_features.py         # Unit tests
│
├── models/                       # Saved models (gitignored)
├── results/                      # Evaluation results
└── logs/                         # Log files (gitignored)
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- Java 8+ (for PySpark)
- 8GB+ RAM (16GB recommended)
- Kaggle account (for dataset download)

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/featureforge.git
   cd featureforge
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install package**:
   ```bash
   pip install -e .
   ```

5. **Verify installation**:
   ```bash
   python -c "import pyspark; print(f'PySpark {pyspark.__version__} installed!')"
   ```

---

## ⚡ Quick Start

### Option 1: Use Synthetic Data (Fastest)

```bash
# Generate 10K synthetic rows for testing
jupyter notebook notebooks/01_data_download.ipynb
# Run the "Generate Synthetic Data" cell

# Run EDA
jupyter notebook notebooks/02_eda.ipynb

# Train baseline model
jupyter notebook notebooks/03_baseline_model.ipynb
```

### Option 2: Use Full Criteo Dataset

```bash
# 1. Setup Kaggle API credentials
#    - Go to https://www.kaggle.com/settings/account
#    - Create API token
#    - Move kaggle.json to ~/.kaggle/

# 2. Download dataset (~10GB)
bash scripts/download_data.sh

# 3. Create 1M row sample
python scripts/create_sample.py

# 4. Run notebooks
jupyter notebook notebooks/02_eda.ipynb
jupyter notebook notebooks/03_baseline_model.ipynb
```

---

## 📊 Dataset

### Criteo Click-Through Rate Prediction

- **Source**: [Kaggle Competition](https://www.kaggle.com/c/criteo-display-ad-challenge)
- **Size**: ~40M rows, ~10GB
- **Task**: Binary classification (click or no click)
- **Features**:
  - **Target**: `click` (0 or 1)
  - **Numerical**: 13 features (I1-I13)
  - **Categorical**: 26 features (C1-C26)

### Key Characteristics

- **Class Imbalance**: ~2-3% click rate (handled with `scale_pos_weight`)
- **Missing Values**: Present in many columns (handled with imputation)
- **High Cardinality**: Categorical features have 100s-1000s of unique values
- **Real-World**: Actual advertising data from Criteo

---

## 🎯 Phase 1: Baseline Model

### Baseline Features (15-20 features)

1. **Original Numerical Features** (13 features):
   - I1, I2, ..., I13 (with missing value imputation)

2. **Missing Value Indicators** (13 features):
   - `I1_missing`, `I2_missing`, ..., `I13_missing`
   - Binary flags indicating if value was missing

3. **Count Encoding** (26 features):
   - For each categorical feature C1-C26
   - Replace category with its frequency count
   - Example: If C1="abc" appears 1000 times → C1_count=1000

4. **Target Encoding** (5 features × 2):
   - For top 5 categorical features
   - `C{i}_mean_ctr`: Mean click rate for each category
   - `C{i}_target_count`: Number of samples for each category
   - Helps capture category-target relationships

**Total: ~57 baseline features**

### Model Training

- **Algorithm**: XGBoost (gradient boosting)
- **Class Imbalance**: Handled with `scale_pos_weight` parameter
- **Early Stopping**: Prevents overfitting
- **Evaluation**: Multiple metrics (F1, AUC-ROC, AUC-PR, etc.)

### Baseline Metrics (Example)

```
Baseline F1-Score: 0.XXXX
AUC-ROC: 0.XXXX
AUC-PR: 0.XXXX
Precision: 0.XXXX
Recall: 0.XXXX
```

*This baseline serves as the CONTROL group for A/B testing in Phase 2.*

---

## 💻 Usage

### Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  sample_size: 1000000  # Number of rows in sample

spark:
  executor_memory: "4g"
  driver_memory: "4g"

model:
  xgboost_params:
    max_depth: 6
    learning_rate: 0.1
    n_estimators: 100
```

### Training a Model

```python
from src.config import Config
from src.utils.spark_utils import create_spark_session
from src.data.loader import CriteoDataLoader
from src.features.feature_engine import FeatureEngine
from src.models.trainer import XGBoostTrainer

# Load config
config = Config('config/config.yaml')

# Create Spark session
spark = create_spark_session(
    app_name=config['spark']['app_name'],
    master=config['spark']['master']
)

# Load data
loader = CriteoDataLoader(spark, config)
df = loader.load_parquet('data/sample/')

# Create features
feature_engine = FeatureEngine(config)
df_features = feature_engine.create_baseline_features(df)

# ... rest of pipeline
```

### Evaluating a Model

```python
from src.models.evaluator import ModelEvaluator

evaluator = ModelEvaluator(output_dir='results')

# Evaluate and create visualizations
metrics = evaluator.create_evaluation_report(
    y_true=y_test,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    feature_importance=importance_df,
    save_prefix='baseline'
)
```

---

## 📈 Results

Results are saved in the `results/` directory:

- `baseline_results.json`: Metrics summary
- `baseline_confusion_matrix.png`: Confusion matrix
- `baseline_roc_curve.png`: ROC curve
- `baseline_pr_curve.png`: Precision-Recall curve
- `baseline_prediction_dist.png`: Prediction distribution
- `baseline_feature_importance.png`: Top 20 features
- `baseline_feature_importance.csv`: Full feature importance

### Viewing Results

```bash
# View metrics
cat results/baseline_results.json

# View feature importance
head -20 results/baseline_feature_importance.csv
```

---

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Run specific test:

```bash
pytest tests/test_features.py
```

---

## 🛠️ Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
pylint src/

# Type checking
mypy src/
```

### Adding New Features

1. Create feature class in `src/features/`
2. Add to `FeatureEngine`
3. Write tests in `tests/`
4. Update documentation

---

## 🚀 Next Steps (Phase 2)

- [ ] Implement 70+ experimental features:
  - Interaction features
  - Polynomial features
  - Frequency-based features
  - Statistical aggregations
  - Time-based features
- [ ] A/B testing framework
- [ ] Statistical significance testing
- [ ] Feature selection algorithms
- [ ] Model deployment pipeline
- [ ] REST API for predictions
- [ ] Docker containerization

---

## 📝 Project Highlights for Resume

This project demonstrates:

- ✅ **Big Data Processing**: PySpark for scalable ETL
- ✅ **Feature Engineering**: Multiple encoding strategies
- ✅ **ML Modeling**: XGBoost with hyperparameter tuning
- ✅ **Evaluation**: Comprehensive metrics and visualizations
- ✅ **Production Code**: Modular, tested, documented
- ✅ **Version Control**: Git best practices
- ✅ **Experimentation**: A/B testing framework (Phase 2)

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or feedback:
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- Criteo for providing the dataset
- Kaggle for hosting the competition
- Open source community for amazing tools (PySpark, XGBoost, scikit-learn)

---

## 📚 References

- [Criteo CTR Dataset](https://www.kaggle.com/c/criteo-display-ad-challenge)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [Feature Engineering Best Practices](https://www.kaggle.com/learn/feature-engineering)

---

**Built with ❤️ for production ML engineering**
