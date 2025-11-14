# FeatureForge - Quick Start Guide

Get started with FeatureForge in 5 minutes!

---

## 🚀 Super Quick Start (Testing with Synthetic Data)

This is the fastest way to test the entire pipeline without downloading the full dataset.

### 1. Install Dependencies (2 minutes)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Generate Synthetic Data (1 minute)

```bash
# Start Jupyter
jupyter notebook

# Open notebooks/01_data_download.ipynb
# Run the "Generate Synthetic Data" cell (creates 10,000 rows)
```

### 3. Run EDA (1 minute)

```bash
# Open notebooks/02_eda.ipynb
# Run all cells
```

You'll see:
- Data distribution analysis
- Click rate: ~3%
- Missing value analysis
- Feature distributions
- Correlation heatmaps

### 4. Train Baseline Model (1 minute)

```bash
# Open notebooks/03_baseline_model.ipynb
# Run all cells
```

You'll get:
- ✅ Baseline F1-Score
- ✅ Feature importance analysis
- ✅ ROC/PR curves
- ✅ Confusion matrix
- ✅ Saved model in `models/`

---

## 📊 Full Dataset Setup (with Criteo Data)

For production-grade results with the full 40M row dataset.

### 1. Setup Kaggle API

```bash
# Install Kaggle CLI
pip install kaggle

# Get API credentials:
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New API Token"
# 3. Move kaggle.json to ~/.kaggle/

mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Download Dataset (~10GB, 10-20 minutes)

```bash
bash scripts/download_data.sh
```

### 3. Create 1M Row Sample (5-10 minutes)

```bash
python scripts/create_sample.py
```

This creates a stratified 1M row sample in `data/sample/` for faster iteration.

### 4. Run Notebooks

```bash
# EDA
jupyter notebook notebooks/02_eda.ipynb

# Train baseline model
jupyter notebook notebooks/03_baseline_model.ipynb
```

---

## 📁 What You'll Get

After running the baseline model notebook:

```
results/
├── baseline_results.json           # Metrics summary
├── baseline_confusion_matrix.png   # Confusion matrix
├── baseline_roc_curve.png          # ROC curve
├── baseline_pr_curve.png           # Precision-Recall curve
├── baseline_prediction_dist.png    # Prediction distribution
├── baseline_feature_importance.png # Top 20 features
└── baseline_feature_importance.csv # Full importance list

models/
└── baseline_xgboost.model          # Trained model

logs/
└── baseline_model.log              # Training logs
```

---

## 🎯 Understanding the Results

### Key Metrics

- **F1-Score**: Harmonic mean of precision and recall (balanced metric)
- **AUC-ROC**: Area under ROC curve (overall discrimination)
- **AUC-PR**: Area under PR curve (important for imbalanced data)
- **Precision**: Of predicted clicks, how many are correct?
- **Recall**: Of actual clicks, how many did we catch?

### What's a Good Score?

For CTR prediction with 2-3% click rate:
- **F1-Score**: 0.20-0.40 is good, >0.40 is excellent
- **AUC-ROC**: 0.70-0.80 is good, >0.80 is excellent
- **AUC-PR**: 0.10-0.20 is good, >0.20 is excellent

The severe class imbalance makes this a challenging task!

---

## 🔧 Customization

### Adjust Sample Size

Edit `config/config.yaml`:

```yaml
data:
  sample_size: 5000000  # 5M rows instead of 1M
```

### Tune Model Parameters

Edit `config/config.yaml`:

```yaml
model:
  xgboost_params:
    max_depth: 8           # Increase tree depth
    learning_rate: 0.05    # Lower learning rate
    n_estimators: 200      # More trees
```

### Change Spark Memory

Edit `config/config.yaml`:

```yaml
spark:
  executor_memory: "8g"    # Increase if you have more RAM
  driver_memory: "8g"
```

---

## 🐛 Troubleshooting

### Issue: "Java not found" when starting Spark

**Solution**: Install Java 8 or higher

```bash
# macOS
brew install openjdk@11

# Ubuntu
sudo apt-get install openjdk-11-jdk

# Windows
# Download from https://adoptopenjdk.net/
```

### Issue: "Out of memory" error

**Solutions**:
1. Reduce sample size in `config.yaml`
2. Increase Spark memory in `config.yaml`
3. Use fewer features in baseline

### Issue: "Kaggle credentials not found"

**Solution**: Setup Kaggle API properly

```bash
# Make sure kaggle.json exists
ls ~/.kaggle/kaggle.json

# Set proper permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Issue: Notebook kernel crashes

**Solutions**:
1. Restart kernel and clear output
2. Reduce data size (use smaller sample)
3. Close other applications to free memory

---

## 📊 Expected Output (Example)

```
==========================================
BASELINE MODEL - PHASE 1 RESULTS
==========================================

📊 BASELINE F1-SCORE: 0.3245

This F1-score serves as the CONTROL group for A/B testing.
In Phase 2, experimental features will be compared against this baseline.

Other Key Metrics:
  - AUC-ROC: 0.7823
  - AUC-PR:  0.1876
  - Precision: 0.2901
  - Recall:    0.3654
==========================================
```

---

## ✅ Verification Checklist

- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] Data loaded (synthetic or Criteo)
- [ ] EDA notebook runs successfully
- [ ] Baseline model trains without errors
- [ ] Results saved in `results/` directory
- [ ] Model saved in `models/` directory
- [ ] F1-score documented

---

## 🚀 Next Steps

### Phase 2: Experimental Features

1. Create 70+ advanced features:
   - Feature interactions (I1 × I2, C1 × C2, etc.)
   - Polynomial features (I1², I1³)
   - Statistical aggregations (mean, std by category)
   - Frequency-based features
   - Time-based features

2. A/B Testing:
   - Compare experimental vs baseline
   - Statistical significance testing
   - Feature selection

3. Model Optimization:
   - Hyperparameter tuning
   - Ensemble methods
   - Model deployment

### Ready for Phase 2?

Once you have your baseline F1-score documented, you're ready to move to Phase 2!

Report back with:
- ✅ Your baseline F1-score
- ✅ Number of features created
- ✅ Dataset size used
- ✅ Any issues encountered

---

## 📧 Need Help?

- Check the main [README.md](README.md) for detailed documentation
- Review the code comments in each module
- Check logs in `logs/` directory
- Open an issue on GitHub

---

**Happy Feature Engineering! 🎯**
