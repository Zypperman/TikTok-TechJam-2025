# Data Collection Phase - TikTok TechJam 2025

## Overview
This phase collects and processes multiple review datasets to create a unified, labeled dataset for ML-based review filtering.

## Data Setup Instructions

**Note**: Raw datasets are not included in this repository due to size constraints.

### Download Required Datasets:

1. **Google Maps Restaurant Reviews (Kaggle)**
   - Source: [Google Maps Restaurant Reviews](https://www.kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews)
   - Download: `reviews.csv` and `dataset/` folder
   - Place in: `google_dataset/kaggle/`

2. **McAuley Lab Google Local Dataset**
   - Source: [McAuley Lab Google Local](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/)
   - Download: `review-Alabama.json.gz` and `meta-Alabama.json.gz`
   - Place in: root directory and `google_dataset/google/` respectively

### Expected Directory Structure:
```
01 Data Collection/
├── google_dataset/
│   ├── kaggle/
│   │   ├── reviews.csv                    # Google dataset reviews (1,100 reviews)
│   │   └── dataset/                       # Image dataset folders (taste, menu, indoor/outdoor atmosphere)
│   └── google/
│       ├── review-Alabama.json.gz         # McAuley Lab Google Local reviews (609MB)
│       └── meta-Alabama.json.gz           # McAuley Lab business metadata (16MB)
├── review-Alabama.json.gz                 # McAuley Lab reviews (root level)
├── collected_data/                         # Output directory for processed data
│   ├── unified_reviews_dataset.csv        # Main unified dataset (16MB)
│   ├── ml_ready_dataset.csv               # ML-ready version with extra features (17MB)
│   └── unified_dataset_summary.txt        # Comprehensive dataset summary
├── unified_dataset_processor_v2.py        # Main unified processor
└── TTTJP1 link 1 EDA.ipynb               # Original EDA notebook
```

## Datasets

### 1. Google Dataset (Kaggle)
- **Source**: Google Maps restaurant reviews
- **Size**: 1,100 reviews
- **Columns**: business_name, author_name, text, photo, rating, rating_category
- **Features**: Text reviews with 4-class labeling (Taste, Menu, Indoor atmosphere, Outdoor atmosphere)
- **Images**: Associated photos for each review in dataset/ folders
- **Labeling Format**: Restaurant-specific categories (taste, menu, atmosphere)

### 2. McAuley Lab Dataset
- **Source**: Google Local business reviews
- **Size**: ~50,000 reviews (limited for demo)
- **Format**: JSON.gz with reviews and business metadata
- **Features**: Rich business information (location, category, price, hours)
- **Labeling Format**: General business reviews with metadata

## Quick Start

### Run the Unified Processor
```bash
cd "01 Data Collection"
python unified_dataset_processor_v2.py
```

### Check Results
```bash
ls collected_data/
```

## What the Processor Does

### **Phase 1: Google Dataset Processing**
- Loads `google_dataset/kaggle/reviews.csv`
- Preserves restaurant-specific rating categories (taste, menu, atmosphere)
- Links to image dataset folders
- Adds photo information and restaurant metadata

### **Phase 2: McAuley Lab Processing**
- Loads `review-Alabama.json.gz` (reviews)
- Loads `google_dataset/google/meta-Alabama.json.gz` (business metadata)
- Links reviews to business information
- Adds location, category, and price data

### **Phase 3: Dataset Merging**
- Combines both datasets into unified schema
- Preserves original labeling from each source
- Adds policy violation detection
- Creates ML-ready features

### **Phase 4: Output Generation**
- `unified_reviews_dataset.csv` - Main unified dataset
- `ml_ready_dataset.csv` - ML-ready version with extra features
- `unified_dataset_summary.txt` - Comprehensive summary

## Unified Dataset Schema

| Column | Description | Source |
|--------|-------------|---------|
| `review_id` | Unique identifier | Generated |
| `business_name` | Business name | Both datasets |
| `author_name` | Reviewer name | Both datasets |
| `text` | Review text | Both datasets |
| `rating` | 1-5 star rating | Both datasets |
| `rating_category` | Review category | Google (restaurant-specific) / McAuley (general) |
| `data_source` | Dataset origin | Generated |
| `dataset_type` | Dataset type | Generated |
| `business_category` | Business category | McAuley Lab |
| `business_address` | Business address | McAuley Lab |
| `business_latitude` | GPS latitude | McAuley Lab |
| `business_longitude` | GPS longitude | McAuley Lab |
| `business_price` | Price level | McAuley Lab |
| `policy_violation` | Violation type | Generated (heuristics) |
| `violation_confidence` | Confidence score | Generated |
| `text_length` | Character count | Generated |
| `word_count` | Word count | Generated |
| `has_photo` | Has associated photo | Generated |
| `photo_count` | Number of photos | Generated |
| `has_url` | Contains URL | Generated |
| `has_phone` | Contains phone number | Generated |
| `exclamation_count` | Exclamation marks | Generated |

## Key Features

1. **Proper Path Handling**: Correctly finds both datasets
2. **Labeling Preservation**: Keeps original rating categories from each source
3. **Image Linking**: Preserves photo information from Google dataset
4. **Business Metadata**: Rich location and category data from McAuley Lab
5. **Unified Schema**: Consistent column structure across both datasets
6. **Error Handling**: Better error reporting and fallback options

## Policy Violation Labels

The processor automatically labels reviews using basic heuristics:

- **`valid`** - Normal review (default)
- **`advertisement`** - Contains promotional content
- **`irrelevant`** - Off-topic content
- **`rant`** - Very negative, long reviews

## Requirements

Install required packages:
```bash
pip install pandas numpy
```

Or install from main requirements:
```bash
cd ..
pip install -r requirements.txt
```

## Expected Output

After running the processor, you should see:

- **Phase 1**: Processing Google Dataset → 1,100 reviews
- **Phase 2**: Processing McAuley Lab Dataset → ~50k reviews  
- **Phase 3**: Merging Datasets → ~51k total reviews
- **Final dataset**: Both sources combined with preserved labeling

## Next Steps

After data collection:
1. **EDA**: Use the unified dataset for exploratory analysis
2. **Feature Engineering**: Build on the ML-ready features
3. **Model Development**: Train ML models for policy violation detection
4. **Policy Module**: Implement enforcement rules

## Notes

- **Google Dataset**: Preserves restaurant-specific rating categories (taste, menu, atmosphere)
- **McAuley Lab**: Adds rich business metadata (location, category, price)
- **Unified Schema**: Both datasets use consistent column structure
- **Policy Detection**: Automatic violation labeling for ML training
- **Image Support**: Photo information preserved from Google dataset
