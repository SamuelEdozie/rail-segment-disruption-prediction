# Dataset

This directory contains information about the dataset used for the project **Predicting Rail Segment Disruptions Utilising Imbalance-Aware Machine Learning**.

The raw dataset files are **not stored in this GitHub repository**. They can be downloaded separately from Kaggle and placed into the local `data/raw/` directory as described below.

## Data Source

**Dataset:** NJ Transit + Amtrak (NEC) Rail Performance  
**Source:** Kaggle  
**Dataset page:** [NJ Transit + Amtrak (NEC) Rail Performance](https://www.kaggle.com/datasets/pranavbadami/nj-transit-amtrak-nec-performance)

The dataset contains historical stop-level rail performance records for NJ Transit and Amtrak services, including station movements, scheduled and actual times, delay measurements, service status, rail line, and operator information.

The project uses these stop-level records to construct **directional rail segments** representing movement from one station to the next.

## Why This Dataset Was Used

The project required historical operational data suitable for supervised machine learning.

The NJ Transit + Amtrak dataset provides:

- Timestamped historical rail performance data
- Scheduled and actual service times
- Delay measurements
- Origin and destination station information
- Rail line and operator information
- Multiple months of consistently structured CSV data

This made it suitable for building a reproducible prototype without requiring a live data collection system.

The project focuses on **predicting disruption risk**, rather than identifying the specific cause of a disruption, because reliable incident-cause labels were not consistently available in the dataset.

## Raw Dataset Fields

The main fields used from the source dataset are:

| Field | Description |
|---|---|
| `date` | Date of the train service |
| `train_id` | Unique identifier for the train service |
| `stop_sequence` | Position of the stop within the train journey |
| `from` | Origin station name |
| `from_id` | Numeric identifier for the origin station |
| `to` | Destination station name |
| `to_id` | Numeric identifier for the destination station |
| `scheduled_time` | Scheduled arrival or departure time |
| `actual_time` | Actual arrival or departure time |
| `delay_minutes` | Delay in minutes between actual and scheduled time |
| `status` | Operational status such as departed, arrived, estimated, or cancelled |
| `line` | NJ Transit rail line |
| `type` | Operator type: NJ Transit or Amtrak |

Negative values in `delay_minutes` represent services arriving earlier than scheduled.

## Segment Construction

The original dataset contains stop-level observations.

For this project, a directional segment is constructed from each station pair:

```text
segment_id = from_id -> to_id
```

For example:

```text
Newark Penn Station -> Newark Airport
```

A directional segment is used instead of predicting only at station level because disruption observed at a station may have originated from an upstream section of the route.

Segment-level modelling therefore provides a more localised representation of disruption risk.

## Disruption Label

The primary supervised-learning target used throughout the project is:

```python
disruption = 1 if delay_minutes >= 5 else 0
```

Therefore:

```text
0 = No disruption
1 = Disruption
```

A delay threshold of **5 minutes or greater** is used as the primary disruption definition.

A stricter threshold of **10 minutes or greater** is also tested during sensitivity analysis to investigate how increasing disruption severity affects class imbalance and model performance.

The 10-minute definition is a sensitivity test and is **not** the primary target used for the final system.

## Data Used by Each Prototype

Development progressed through three main prototype stages, with Prototype 1.5 acting as an intermediate stability extension of Prototype 1.

| Iteration | Period Used | Purpose |
|---|---|---|
| **Prototype 1** | March 2018 | Single-month feasibility baseline |
| **Prototype 1.5** | March–May 2018 | Three-month baseline stability check |
| **Prototype 2** | January–June 2019 | Six-month multi-model comparison |
| **Prototype 3** | October–December 2018 + January–May 2019 | Eight-month cross-year final prototype |

### Prototype 1

Required file:

```text
2018_03.csv
```

Prototype 1 established the initial end-to-end pipeline using one month of data.

### Prototype 1.5

Required files:

```text
2018_03.csv
2018_04.csv
2018_05.csv
```

Prototype 1.5 expanded the baseline from one month to three months to test whether the initial approach remained stable with more data.

### Prototype 2

Required files:

```text
2019_01.csv
2019_02.csv
2019_03.csv
2019_04.csv
2019_05.csv
2019_06.csv
```

Prototype 2 expanded the modelling period to six months and introduced systematic comparison between:

- Logistic Regression
- Random Forest
- XGBoost

It also introduced additional experiments involving class weighting, train-only SMOTE, probability calibration, and disruption-threshold sensitivity.

### Prototype 3

Required files:

```text
2018_10.csv
2018_11.csv
2018_12.csv
2019_01.csv
2019_02.csv
2019_03.csv
2019_04.csv
2019_05.csv
```

Prototype 3 is the final research prototype.

It uses a cross-year dataset so that earlier observations can be used for training and later observations can be held out for validation and testing.

The final chronological split is:

| Set | Period |
|---|---|
| **Training** | October–December 2018 |
| **Validation** | January–February 2019 |
| **Testing** | March–May 2019 |

This design reflects a forward-looking prediction setting and reduces the risk of temporal leakage.

## Dataset Characteristics Across Prototypes

The following figures are based on the cleaned modelling datasets using the primary 5-minute disruption definition.

| Prototype | Modelling Records | Segments | Positive Rate | Approx. Imbalance |
|---|---:|---:|---:|---:|
| Prototype 1 | 243,028 | 412 | 20.5% | 3.9:1 |
| Prototype 1.5 | 739,846 | 412 | 21.3% | 3.7:1 |
| Prototype 2 | 1,265,092 | 561 | 26.3% | 2.8:1 |
| Prototype 3 | 1,678,487 | 572 | 29.1% | 2.4:1 |

These figures are based on the processed modelling data and may differ from the number of rows contained in the original monthly CSV files because preprocessing removes or transforms records before modelling.

## Engineered Features

The raw data is transformed into additional features during preprocessing.

Features introduced across the prototype progression include:

| Feature | Introduced | Description |
|---|---|---|
| `segment_id` | Prototype 1 | Directional station pair (`from_id -> to_id`) |
| `disruption` | Prototype 1 | Binary target based on the delay threshold |
| `day_of_week` | Prototype 1 | Day of week extracted from the service date |
| `sched_hour` | Prototype 1 | Hour extracted from the scheduled service time |
| `stop_sequence` | Prototype 1 | Stop position within the train journey |
| `line_*` | Prototype 1 | One-hot encoded rail line features |
| `month` | Prototype 1.5 | Month extracted from the service date |
| `is_weekend` | Prototype 2 | Indicates Saturday or Sunday |
| `seg_rolling_7d_rate` | Prototype 3 | Previous 7-day disruption rate for the segment |
| `seg_rolling_7d_count` | Prototype 3 | Previous 7-day observation count for the segment |

Prototype 3's rolling segment features are shifted so that only historical information is available to the model at prediction time.

This is important for preventing **temporal data leakage**.

## Recommended Local Directory Structure

After downloading the required CSV files from Kaggle, organise them locally as:

```text
data/
├── README.md
└── raw/
    ├── 2018/
    │   ├── 2018_03.csv
    │   ├── 2018_04.csv
    │   ├── 2018_05.csv
    │   ├── 2018_10.csv
    │   ├── 2018_11.csv
    │   └── 2018_12.csv
    │
    └── 2019/
        ├── 2019_01.csv
        ├── 2019_02.csv
        ├── 2019_03.csv
        ├── 2019_04.csv
        ├── 2019_05.csv
        └── 2019_06.csv
```

These 12 monthly files cover all data required to reproduce Prototype 1, Prototype 1.5, Prototype 2, and Prototype 3.

## Why the Raw Data Is Not Included

The source CSV files are intentionally excluded from this repository.

This keeps the repository focused on the project's code, methodology, and results while allowing the original dataset to remain distributed through its source.

Users who wish to reproduce the project should download the required files directly from Kaggle and review the dataset's current usage and licensing terms there.

The repository's `.gitignore` should therefore exclude:

```gitignore
data/raw/
```

## Reproducing the Project

To reproduce the notebook experiments:

1. Download the required monthly CSV files from the Kaggle dataset.
2. Create the `data/raw/2018/` and `data/raw/2019/` directories.
3. Place each CSV in the corresponding year folder.
4. Install the project dependencies.
5. Run the notebooks in numerical order from the repository's `notebooks/` directory.

The cleaned portfolio notebooks use repository-relative paths rather than the original Google Colab/Google Drive paths used during development.

## Important Notes

- The project predicts **rail segment disruption risk**, not the underlying cause of a disruption.
- The primary disruption definition is a delay of **5 minutes or greater**.
- A **10-minute threshold** is used only for severity sensitivity testing.
- Model evaluation uses chronological/time-aware splitting rather than random shuffling.
- SMOTE, when used, is applied only to the training data.
- Prototype 3's rolling segment-history features use past observations only.
- Raw dataset files should not be committed to the repository.

---

This dataset supports the experimental work for **Predicting Rail Segment Disruptions Utilising Imbalance-Aware Machine Learning**, a BSc Computer Science Final Year Project.