# TikTok TechJam 2025 - ML for Trustworthy Location Reviews

## Project Overview

This project builds an ML + NLP system to automatically evaluate and filter location-based reviews according to well-defined policies, improving trust, fairness, and usability of review platforms.

## Project Structure

```
TikTok-TechJam-2025/
├── 01 Data Collection/          # Data collection and preprocessing
├── 02 EDA/                      # Exploratory Data Analysis
├── 03 Feature Engineering/      # Feature extraction and engineering
├── 04 Model analysis/           # ML model development and analysis
├── 05 Policy Module/            # Policy enforcement implementation
├── 06 Evaluation/               # Model evaluation and validation
└── 07 Final Report/             # Project documentation and results
```

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup
**Note**: Raw datasets are not included in this repository due to size constraints.

#### Download Required Datasets:

1. **Google Maps Restaurant Reviews (Kaggle)**
   - Source: [Google Maps Restaurant Reviews](https://www.kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews)
   - Download: `reviews.csv` and `dataset/` folder
   - Place in: `01 Data Collection/google_dataset/kaggle/`

2. **McAuley Lab Google Local Dataset**
   - Source: [McAuley Lab Google Local](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/)
   - Download: `review-Alabama.json.gz` and `meta-Alabama.json.gz`
   - Place in: `01 Data Collection/` and `01 Data Collection/google_dataset/google/` respectively

#### Expected Directory Structure:
```
01 Data Collection/
├── google_dataset/
│   ├── kaggle/
│   │   ├── reviews.csv                    # Google dataset reviews
│   │   └── dataset/                       # Image dataset folders
│   └── google/
│       └── meta-Alabama.json.gz           # Business metadata
├── review-Alabama.json.gz                 # McAuley Lab reviews
└── unified_dataset_processor_v2.py        # Data processor
```

### 3. Run Data Processing
```bash
cd "01 Data Collection"
python unified_dataset_processor_v2.py
```

This will create a unified dataset combining both sources with policy violation labels.

## Current Status

- ✅ **Phase 1 Complete**: Data Collection & Pipeline Setup
- 🔄 **Phase 2**: Baseline Models & Prompt Engineering (In Progress)
- ⏳ **Phase 3**: Policy Enforcement Module
- ⏳ **Phase 4**: Evaluation & Refinement
- ⏳ **Phase 5**: Deliverables & Demo Prep

## Success Metrics

- Precision ≥ 0.80 for detecting ads/irrelevance
- Recall ≥ 0.75 across all policy classes
- F1-score ≥ 0.77 on validation set
- Model inference latency < 500ms per review

## Technologies

- **Language**: Python
- **ML/NLP**: Hugging Face Transformers, PyTorch, scikit-learn
- **Data Processing**: pandas, NumPy
- **Visualization**: matplotlib, seaborn, Plotly
- **Optional UI**: Streamlit

## Contributing

This is a hackathon project for TikTok TechJam 2025. The code is designed to be runnable on Kaggle, Jupyter, or other Python environments.

## License

See [LICENSE](LICENSE) file for details.

