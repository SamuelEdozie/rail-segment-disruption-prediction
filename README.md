# Predicting Rail Segment Disruptions Utilising Imbalance-Aware Machine Learning

An imbalance-aware machine learning project for predicting **rail segment disruption risk** using historical NJ Transit and Amtrak (NEC) performance data.

Originally developed as my final-year project and reorganised here as a portfolio project with cleaned notebooks, reproducibility documentation, selected results, and supporting material.

The project compares **Logistic Regression, Random Forest, and XGBoost** using chronological evaluation, class weighting, train-only SMOTE experiments, probability calibration, and leakage-safe temporal features.

The final prototype produces **ranked rail-segment risk scores and risk tiers** intended to support early-warning and operational decision-making.

---

## Project Overview

Rail disruption prediction is a class-imbalanced machine learning problem.

Most train movements operate without significant disruption, which means a model can achieve apparently strong accuracy simply by predicting the majority class. For an early-warning system, however, detecting the less common disruption events is the more important objective.

For this reason, the project focuses primarily on:

- **PR-AUC / Average Precision**
- Precision
- Recall
- F1-score
- Confusion matrices
- Precision-Recall curves

rather than relying on accuracy as the main measure of model performance.

The project also treats rail data as **time-ordered data**. Random train/test splitting was avoided because allowing future observations to influence model training could create temporal leakage and produce unrealistic performance estimates.

---

## Project Aim

The aim of the project was to:

> Build and evaluate an imbalance-aware early-warning prototype that predicts segment-level rail disruption risk and produces calibrated risk scores suitable for decision support.

The system was designed to:

1. Transform stop-level railway performance records into directional rail segments.
2. Define a reproducible disruption target.
3. Engineer temporal and service-context features.
4. Prevent temporal leakage through chronological evaluation.
5. Compare Logistic Regression, Random Forest, and XGBoost.
6. Evaluate class weighting and train-only SMOTE for class imbalance.
7. Evaluate model performance using imbalance-appropriate metrics.
8. Calibrate model probabilities and investigate operating thresholds.
9. Produce ranked segment-level risk outputs and risk tiers.

---

## Dataset

The project uses the public **NJ Transit + Amtrak (NEC) Rail Performance** dataset.

**Source:** Kaggle  
**Dataset:** [NJ Transit + Amtrak (NEC) Rail Performance](https://www.kaggle.com/datasets/pranavbadami/nj-transit-amtrak-nec-performance)

The original dataset contains stop-level operational records including:

- Scheduled and actual service times
- Delay measurements
- Origin and destination stations
- Train identifiers
- Stop sequence
- Rail line
- Service status
- Operator information

A directional segment is constructed using consecutive station pairs:

```text
segment_id = from_id -> to_id
```

The primary disruption target is:

```python
disruption = 1 if delay_minutes >= 5 else 0
```

A stricter **10-minute disruption threshold** is also evaluated as a sensitivity experiment.

The raw CSV files are not distributed through this repository.

See the full dataset documentation and reproduction instructions here:

**[Dataset Documentation](data/README.md)**

---

## Prototype Development

The project was developed iteratively so that each prototype introduced additional modelling or evaluation complexity.

| Prototype | Data Window | Main Purpose | Best Model | PR-AUC |
|---|---|---|---|---:|
| **Prototype 1** | March 2018 | Feasibility baseline | Logistic Regression + class weighting | 0.2513 |
| **Prototype 1.5** | March-May 2018 | Three-month stability check | Logistic Regression + class weighting | 0.2901 |
| **Prototype 2** | January-June 2019 | Multi-model and imbalance comparison | XGBoost + class weighting | 0.4357 |
| **Prototype 3** | Oct-Dec 2018 + Jan-May 2019 | Cross-year final prototype | XGBoost + class weighting | **0.5134** |

### Prototype 1: Feasibility Baseline

Prototype 1 established the first working pipeline using one month of data.

It introduced:

- Data loading and cleaning
- Directional segment construction
- Binary disruption labelling
- Temporal and schedule features
- Time-aware splitting
- Logistic Regression baseline
- Class weighting
- PR-AUC evaluation
- Confusion-matrix analysis

**Notebook:**  
[`01_prototype_1_feasibility_baseline.ipynb`](notebooks/01_prototype_1_feasibility_baseline.ipynb)

### Prototype 1.5: Baseline Stability

Prototype 1.5 expanded the original one-month baseline to three months.

The purpose was to check whether the original modelling pipeline and performance remained stable when applied to a larger dataset before progressing to more complex models.

**Notebook:**  
[`02_prototype_1_5_baseline_stability.ipynb`](notebooks/02_prototype_1_5_baseline_stability.ipynb)

### Prototype 2: Multi-Model Comparison

Prototype 2 expanded the modelling period to six months and introduced systematic comparison between:

- Logistic Regression
- Random Forest
- XGBoost

It also investigated:

- Class weighting
- SMOTE applied to training data only
- 5-minute vs 10-minute disruption thresholds
- Probability calibration
- Threshold analysis
- Feature importance
- Segment-level risk outputs

XGBoost with class weighting achieved the strongest PR-AUC at this stage:

**PR-AUC = 0.4357**

**Notebook:**  
[`03_prototype_2_model_comparison.ipynb`](notebooks/03_prototype_2_model_comparison.ipynb)

### Prototype 3: Cross-Year Early-Warning Prototype

Prototype 3 represents the final research prototype.

It introduced:

- Cross-year train/validation/test evaluation
- Seasonal analysis
- Leakage-safe historical segment features
- 7-day rolling segment disruption history
- Probability calibration
- Threshold selection
- Ranked segment risk scores
- Risk tiers for decision support

The cross-year split was:

| Dataset | Period |
|---|---|
| Training | October-December 2018 |
| Validation | January-February 2019 |
| Testing | March-May 2019 |

Historical segment features were shifted so that only information available **before the prediction period** was used.

**Notebook:**  
[`04_prototype_3_cross_year_early_warning.ipynb`](notebooks/04_prototype_3_cross_year_early_warning.ipynb)

---

## Final Model Results

The strongest model under the final cross-year evaluation was **XGBoost with class weighting**.

| Model | PR-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **XGBoost + Class Weighting** | **0.5134** | **0.4729** | 0.5516 | 0.5093 |
| XGBoost + Class Weighting + SMOTE | 0.5074 | 0.4017 | **0.7087** | **0.5128** |
| Random Forest + Class Weighting | 0.4840 | 0.5112 | 0.3945 | 0.4453 |
| Logistic Regression + Class Weighting | 0.4597 | 0.4235 | 0.5373 | 0.4737 |

Although SMOTE increased recall, it also reduced precision and slightly reduced PR-AUC.

For the intended early-warning setting, excessive false alarms could reduce the usefulness of the system. **Class weighting was therefore retained as the preferred imbalance-handling strategy.**

---

## Prototype Performance Progression

The strongest PR-AUC increased across each stage of development:

```text
Prototype 1     0.2513
      ↓
Prototype 1.5   0.2901
      ↓
Prototype 2     0.4357
      ↓
Prototype 3     0.5134
```

A major finding was that the improvement between Prototype 2 and Prototype 3 was driven primarily by **leakage-safe historical segment features**, particularly the rolling 7-day segment disruption history, rather than simply selecting a more complex algorithm.

---

## Imbalance-Aware Evaluation

The project deliberately avoids using accuracy as the primary metric.

A model predicting mostly "no disruption" could achieve high accuracy while failing to detect the events that matter.

The primary evaluation metric is therefore **PR-AUC**, supported by:

- Precision
- Recall
- F1-score
- Precision-Recall curves
- Confusion matrices

A stricter disruption definition of **10 minutes or greater** was also tested.

Increasing the threshold substantially increased class imbalance and reduced PR-AUC, demonstrating that severe disruption prediction becomes considerably harder as positive events become rarer.

---

## Preventing Temporal Leakage

Temporal leakage was treated as a major validity risk throughout the project.

The pipeline therefore uses:

- Chronological train/validation/test splitting
- No random shuffling of time-ordered observations
- Train-only SMOTE
- Historical features constructed using past information only
- Shifted rolling segment features
- Cross-year hold-out evaluation in Prototype 3

For example, Prototype 3 computes a rolling 7-day disruption rate for each segment and shifts the feature so that the current observation cannot contribute information to its own prediction.

---

## Early-Warning Output

The final prototype moves beyond binary classification.

Predicted probabilities are used to produce a **ranked segment-level risk view**, allowing higher-risk rail segments to be prioritised.

Outputs include:

- Calibrated disruption probabilities
- Ranked segments
- Risk tiers
- Precision-Recall evaluation
- Calibration analysis
- Threshold analysis
- Predicted vs observed segment-risk comparisons

This allows the project to function as a **decision-support prototype**, rather than simply outputting a disruption/no-disruption classification.

---

## Repository Structure

```text
rail-segment-disruption-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_prototype_1_feasibility_baseline.ipynb
│   ├── 02_prototype_1_5_baseline_stability.ipynb
│   ├── 03_prototype_2_model_comparison.ipynb
│   └── 04_prototype_3_cross_year_early_warning.ipynb
│
├── results/
│   ├── figures/
│   └── tables/
│
└── docs/
    └── project_presentation.pdf
```

---

## Results and Visualisations

The `results/` directory contains selected outputs from the experiments.

### Figures

The planned portfolio-facing figures include:

- Prototype PR-AUC progression
- Final model comparison
- Precision-Recall curves
- Calibration / reliability plots
- Threshold analysis
- Risk-tier distribution
- Predicted vs observed segment risk

### Tables

The selected tables include:

- Prototype performance comparison
- Final model comparison
- Threshold sensitivity analysis
- Ranked high-risk segments

See:

[`results/figures/`](results/figures/)  
[`results/tables/`](results/tables/)

---

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/SamuelEdozie/rail-segment-disruption-prediction.git
cd rail-segment-disruption-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the required monthly CSV files from Kaggle:

[NJ Transit + Amtrak (NEC) Rail Performance](https://www.kaggle.com/datasets/pranavbadami/nj-transit-amtrak-nec-performance)

Follow the directory and file instructions in:

[`data/README.md`](data/README.md)

### 4. Run the notebooks

The notebooks demonstrate the progression of the project and should be viewed in numerical order:

```text
01 -> Prototype 1
02 -> Prototype 1.5
03 -> Prototype 2
04 -> Prototype 3
```

---

## Technologies

The project was developed primarily in Python using:

- Python
- Jupyter Notebook / Google Colab
- pandas
- NumPy
- scikit-learn
- XGBoost
- imbalanced-learn
- Matplotlib
- openpyxl

Machine learning techniques include:

- Logistic Regression
- Random Forest
- XGBoost
- Class weighting
- SMOTE
- Probability calibration
- Threshold tuning
- Time-aware validation
- Temporal feature engineering

---

## Key Findings

1. **XGBoost with class weighting produced the strongest final PR-AUC.**
2. **SMOTE increased recall but also increased false alarms**, reducing precision and slightly reducing PR-AUC.
3. **Historical segment context mattered more than increasing model complexity alone.**
4. **Chronological evaluation was essential** for preventing unrealistic results caused by temporal leakage.
5. **The disruption definition strongly affects difficulty.** Increasing the threshold from 5 to 10 minutes substantially increased class imbalance and reduced model performance.
6. **Calibrated probabilities made the model more useful for decision support**, allowing segments to be ranked by estimated disruption risk.

---

## Limitations

The project has several important limitations:

- The **5-minute delay threshold is a proxy for disruption**, not a verified incident-cause label.
- The model estimates disruption risk but does not identify whether a disruption was caused by weather, signalling, infrastructure failure, staffing, or another specific cause.
- The experiments use a single NJ Transit/Amtrak dataset family.
- Cross-network generalisation has not been validated.
- Risk estimates can be less stable for rarely observed segments.
- The prototype operates in batch mode rather than as a live production system.
- Real-world operational deployment and user evaluation were outside the scope of the project.

---

## Future Work

Potential extensions include:

- Weather and environmental data
- Planned engineering works
- Operational incident logs
- Cross-network validation
- Multi-class disruption severity prediction
- Graph-based modelling of disruption propagation
- Graph Neural Networks
- Streaming data ingestion
- Near-real-time risk scoring
- Interactive operational dashboard

A future production system could combine live data ingestion with continuously updated historical segment features and expose risk scores through an operational monitoring interface.

---

## Project Presentation

A presentation covering the project's motivation, methodology, prototype progression, results, limitations, and final demonstration is available here:

**[Project Presentation](docs/project_presentation.pdf)**

---

## About This Repository

This repository is the **portfolio version of my final-year project**.

The original development work has been reorganised and documented to make the project easier to understand, reproduce, and review as a standalone machine learning project.

---

## Author

**Samuel Edozie**
