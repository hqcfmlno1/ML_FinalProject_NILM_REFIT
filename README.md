# NILM REFIT - Non-Intrusive Load Monitoring Using Machine Learning

This project focuses on **Non-Intrusive Load Monitoring (NILM)** (non-intrusive household load monitoring) using Machine Learning and Deep Learning models on the **REFIT (House 2)** dataset. The goal is to disaggregate aggregate household electricity consumption (Aggregate mains) into individual appliance power consumption without installing individual sensors on each appliance.

---

## Target Appliances (House 2)
The project analyzes and predicts load profiles for **9 key appliances** in House 2 of the REFIT dataset:
1. **Appliance 1**: Fridge-Freezer
2. **Appliance 2**: Washing Machine
3. **Appliance 3**: Dishwasher
4. **Appliance 4**: Television
5. **Appliance 5**: Microwave
6. **Appliance 6**: Toaster
7. **Appliance 7**: Kettle
8. **Appliance 8**: Cooker
9. **Appliance 9**: Tumble Dryer

---

## Implemented Models

The project implements various models ranging from baseline machine learning algorithms to advanced deep learning architectures for comparison and optimization:

### 1. Baseline Machine Learning Models
*   **Linear Regression with Window Shifting**: Employs a sliding window approach (window size $W=20$) on the aggregate mains power combined with temporal features (Hour, Minute, DayOfWeek, Month, IsWeekend) to predict instantaneous power for 9 independent appliances.
*   **Random Forest Regressor**: Trained on windowed features for improved non-linear regression capability.
*   **XGBoost Regressor**: A powerful gradient boosting algorithm optimized for speed and accuracy in load disaggregation regression.
*   **Hidden Markov Model (HMM) & Factorial HMM (FHMM)**: A probabilistic time-series approach using the `hmmlearn` library, disaggregating aggregate power based on the hidden operational states of individual appliances.

### 2. Advanced Deep Learning Models
*   **Sequence-to-Point (S2P) / S2PwA (S2P with Attention)**: Takes a sequence of aggregate mains power (window length of 300) and predicts the power consumption of a target appliance at the midpoint of the window. The network consists of 1D CNN layers (local spatial feature extraction), BiLSTM (two-way sequential modeling), and a self-attention mechanism to focus on key activation regions.
*   **Hybrid S2PwA (Classifier + Regressor)**:
    *   *Classifier (S2qwaClassifier)*: Predicts the ON/OFF state of an appliance using a deep 1D CNN with Weighted Binary Cross-Entropy Loss to handle class imbalance.
    *   *Regressor (S2PwaRegressor)*: Focuses solely on estimating the active power consumption when the appliance is active, utilizing a Masked L1 Loss.
    *   *Final Output* = Regressor Power × Classifier State (ON/OFF).
*   **AugLPN (Augmented Light-weight Feature Pyramid Network)**: The state-of-the-art deep learning model in this project. It features a Light-weight Feature Pyramid Network (**LFPN**) to construct multi-scale feature pyramids from aggregate mains data, combined with two parallel branches: a **Spatial Branch L** (extracts spatial features via Attention and Dilated Convolutions) and a **Temporal Branch R** (extracts temporal features via Depthwise Separable Convolutions and BiGRU). It is optimized using Weighted MAE Loss.

---

## Project Structure

```
ML_FinalProject_NILM_REFIT/
├── README.md                    # Project documentation (This file)
├── requirements.txt             # Project library dependencies
│
├── data/                        # Directory for datasets
│   ├── processed_data/          # Preprocessed House 2 data
│   │   ├── House2_full.csv      # Merged dataset (combined from parts 1-5)
│   │   └── House2_part1-5.csv   # Split raw dataset files
│   ├── train/                   # Training set data (numpy, npy, h5 formats)
│   └── test/                    # Test set data (numpy, npy, h5 formats)
│
├── notebooks/                   # Jupyter Notebooks for experimentation
│   ├── data_merge.ipynb         # Merges CSV parts 1-5 into House2_full.csv
│   ├── eda.ipynb                # Exploratory Data Analysis (EDA)
│   ├── test_metrics.ipynb       # Tests evaluation metrics (Precision, Recall, F1...)
│   ├── window_shilfter_test.ipynb # Tests window shifting technique
│   │
│   ├── fhmm/                    # FHMM & HMM models
│   │   ├── hmm_basic.ipynb      # Basic HMM experiment
│   │   ├── fhmm_data_prep.ipynb # Data preparation for FHMM
│   │   ├── fhmm_training.ipynb  # Training and evaluating Factorial HMM
│   │   └── fhmm_results.png     # Plot of FHMM results
│   │
│   ├── randomforest/            # Random Forest models
│   │   ├── random_forest_test.ipynb  # Random Forest training
│   │   └── random_forest_nowd_rg.ipynb # Random Forest without window shifting
│   │
│   ├── xgboost/                 # XGBoost models
│   │   └── xgboost_test.ipynb   # Training and evaluating XGBoost Regressor
│   │
│   └── s2pwa/                   # Sequence-to-Point with Attention models
│       ├── train_test_split.ipynb # Train/test split for deep learning
│       ├── produce_dataset.ipynb  # Time-series sequence dataset generation (npy/h5 formats)
│       ├── s2pwa_prototype_draft.ipynb # S2PwA baseline model draft
│       └── s2pwa_hybrid.ipynb     # Hybrid S2PwA experiments
│
├── src/                         # Modularized source code
│   ├── metrics/                 # Evaluation metrics
│   │   └── energy_base_metrics.py # Precision, Recall, F1 (Energy-based), MAE, NEP
│   ├── tools/                   # Utility scripts
│   │   └── window_shifter.py    # WindowShifter class for window sliding
│   └── models/                  # Deep learning models & training scripts
│       ├── linear_regression_window_shift.py # Trains 9 Linear Regression models
│       ├── s2pwa_baseline.py    # Defines S2PwA baseline model & training loop
│       ├── hybrid_s2pwa.py      # Defines Hybrid S2PwA model (Classifier + Regressor)
│       └── auglpn.py            # Defines AugLPN advanced model & training loop
│
└── checkpoints/                 # Saved model weights (checkpoints)
    ├── basic models/            # Baseline model checkpoints (Linear Regression, etc.)
    ├── advanced models/         # S2PwA baseline model checkpoints (s2p_app*.pth)
    ├── advanced hybrid models/  # Hybrid S2PwA model checkpoints
    │   ├── classifier/          # ON/OFF classifiers (classifier_app*.pth)
    │   └── regressor/           # Power estimators (regressor_app*.pth)
    └── auglpn/                  # Advanced AugLPN model checkpoints (auglpn_app*.pth)
```

---

## Evaluation Metrics

The performance of the models is evaluated using standard NILM disaggregation metrics (defined in [energy_base_metrics.py](file:///c:/Users/HOANG%20QUOC%20CUONG/OneDrive/M%C3%A1y%20t%C3%ADnh/20252/ML/ML_FinalProject_NILM_REFIT/src/metrics/energy_base_metrics.py)):

1.  **Precision (Energy-based)**: The ratio of correctly predicted energy to the total predicted energy of the appliance.
    $$\text{Precision} = \frac{\sum \min(\hat{y}_t, y_t)}{\sum \hat{y}_t}$$
2.  **Recall (Energy-based)**: The ratio of correctly predicted energy to the total actual energy consumption of the appliance.
    $$\text{Recall} = \frac{\sum \min(\hat{y}_t, y_t)}{\sum y_t}$$
3.  **F1-score (Energy-based)**: The harmonic mean of Precision and Recall.
4.  **MAE (Mean Absolute Error)**: The average absolute error in power consumption prediction (in Watts).
5.  **NEP (Normalized Error in Power)**: The power error normalized by the total actual consumption.

---

## Installation & Usage Guidelines

### 1. Environment Setup
The project requires Python 3.8+ and a virtual environment. To install all the required libraries:
```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment (Windows)
.\.venv\Scripts\activate

# Upgrade pip and install dependencies
pip install -r requirements.txt
```

### 2. Data Preprocessing
*   If you don't have `House2_full.csv` in your `data/processed_data/` directory, run the [data_merge.ipynb](file:///c:/Users/HOANG%20QUOC%20CUONG/OneDrive/M%C3%A1y%20t%C3%ADnh/20252/ML/ML_FinalProject_NILM_REFIT/notebooks/data_merge.ipynb) notebook to merge the raw `House2_part1-5.csv` files.
*   Run the [train_test_split.ipynb](file:///c:/Users/HOANG%20QUOC%20CUONG/OneDrive/M%C3%A1y%20t%C3%ADnh/20252/ML/ML_FinalProject_NILM_REFIT/notebooks/s2pwa/train_test_split.ipynb) notebook to split the dataset into training and testing subsets.
*   Run the [produce_dataset.ipynb](file:///c:/Users/HOANG%20QUOC%20CUONG/OneDrive/M%C3%A1y%20t%C3%ADnh/20252/ML/ML_FinalProject_NILM_REFIT/notebooks/s2pwa/produce_dataset.ipynb) notebook to create time-series sequence datasets saved in `.npy` format to speed up training for deep learning models.

### 3. Training & Running Models

#### A. Linear Regression Model
Run the Python script to train the 9 windowed linear regression models and output evaluation metrics alongside visualizations:
```bash
python src/models/linear_regression_window_shift.py
```
The plot results will be saved in `outputs/linear_regression_window_shift/plots/` including:
*   *Overview*: Compares Ground Truth and Predictions across the entire test set.
*   *Focused*: Zooms into the periods of highest appliance activity for detailed analysis.

#### B. Other Machine Learning Models (Random Forest, XGBoost, FHMM)
Open the respective notebooks in the `notebooks/` directory to review preprocessing, training, and evaluation steps:
*   [xgboost_test.ipynb](file:///c:/Users/HOANG%20QUOC%20CUONG/OneDrive/M%C3%A1y%20t%C3%ADnh/20252/ML/ML_FinalProject_NILM_REFIT/notebooks/xgboost/xgboost_test.ipynb) for XGBoost.
*   [random_forest_test.ipynb](file:///c:/Users/HOANG%20QUOC%20CUONG/OneDrive/M%C3%A1y%20t%C3%ADnh/20252/ML/ML_FinalProject_NILM_REFIT/notebooks/randomforest/random_forest_test.ipynb) for Random Forest.
*   [fhmm_training.ipynb](file:///c:/Users/HOANG%20QUOC%20CUONG/OneDrive/M%C3%A1y%20t%C3%ADnh/20252/ML/ML_FinalProject_NILM_REFIT/notebooks/fhmm/fhmm_training.ipynb) for Factorial HMM.

#### C. PyTorch Deep Learning Models
The notebooks in `notebooks/auglpn/` and `notebooks/s2pwa/` provide the full workflow for neural network design, CPU/GPU training, checkpoint saving, and test-set evaluation:
*   [auglpn_nilm.ipynb](file:///c:/Users/HOANG%20QUOC%20CUONG/OneDrive/M%C3%A1y%20t%C3%ADnh/20252/ML/ML_FinalProject_NILM_REFIT/notebooks/auglpn/auglpn_nilm.ipynb): Trains and evaluates the advanced **AugLPN** model.
*   [s2pwa_hybrid.ipynb](file:///c:/Users/HOANG%20QUOC%20CUONG/OneDrive/M%C3%A1y%20t%C3%ADnh/20252/ML/ML_FinalProject_NILM_REFIT/notebooks/s2pwa/s2pwa_hybrid.ipynb): Trains the **Hybrid S2PwA** model.
