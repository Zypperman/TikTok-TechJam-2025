# What this does

* Loads CSV review data and prints sample records.
* Reports column info, missing values, and summary stats.
* Visualizes and counts policy violation types.
* Plots ratings, rating categories, review lengths, and word counts.
* Shows distribution of review flags: photo, URL, phone, email.
* Highlights top businesses and total unique reviewers.
* Breaks down business categories, dataset types, data sources.
* Analyzes review dates: year and month trends.
* Displays a table of policy violation type per rating.
* Generates wordclouds for all reviews and each violation class.
* Lists most common words for each violation.
* Performs optional sentiment analysis using VADER if installed.
* Computes avg text length and word count by violation type.
* Outputs all plots and statistics for EDA coverage.

# Usage

Below workflow assumes presence of `ml_ready_dataset.csv` and `unified_reviews_dataset.csv` in the current directory.

1. Create a virtual environment and install all requirements within `requirements.txt`
2. Run `python3 eda_full_reviews.py`
