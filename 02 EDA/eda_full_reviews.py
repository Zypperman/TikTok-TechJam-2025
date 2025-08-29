import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import string
import os

# Plot saving function
def save_plot(name):
    os.makedirs("eda_plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"eda_plots/{name}.png")
    plt.close()

# =================== CONFIGURATION ===================
INPUT_FILE = "ml_ready_dataset.csv"  # Change as needed

# =================== LOAD DATA ===================
df = pd.read_csv(INPUT_FILE)
print(f"\nLoaded {len(df)} rows from {INPUT_FILE}")

# =================== BASIC OVERVIEW ===================
print("\n======== HEAD ========")
print(df.head(3))
print("\n======== INFO ========")
print(df.info())
print("\n======== MISSING VALUES ========")
print(df.isnull().sum())
print("\n======== DESCRIPTIVE STATS ========")
print(df.describe(include='all'))

# =================== POLICY VIOLATION ANALYSIS ===================
print("\nPolicy Violation Breakdown:")
print(df['policy_violation'].value_counts(dropna=False))
sns.countplot(x='policy_violation', data=df, order=df['policy_violation'].value_counts().index)
plt.title("Policy Violation Distribution")
save_plot("policy_violation_distribution")

if 'violation_confidence' in df.columns:
    print("\nViolation Confidence Statistics:")
    print(df['violation_confidence'].describe())
    sns.histplot(df['violation_confidence'].dropna(), bins=30, kde=True)
    plt.title("Violation Confidence Distribution")
    plt.xlabel("Violation Confidence")
    save_plot("violation_confidence_distribution")

# =================== RATING AND CATEGORY DISTRIBUTIONS ==============
print("\nReview Rating Breakdown:")
print(df['rating'].value_counts(dropna=False).sort_index())
sns.countplot(x='rating', data=df)
plt.title("Review Ratings Distribution")
save_plot("review_ratings_distribution")

if 'rating_category' in df.columns:
    print("\nRating Category Breakdown:")
    print(df['rating_category'].value_counts(dropna=False))
    sns.countplot(x='rating_category', data=df, order=df['rating_category'].value_counts().index)
    plt.title("Rating Category Distribution")
    save_plot("rating_category_distribution")

# =================== TEXT STATISTICS ===================
for col in ['text_length', 'word_count']:
    if col in df.columns:
        print(f"\n{col.replace('_', ' ').title()} Stats:")
        print(df[col].describe())
        sns.histplot(df[col].dropna(), bins=30, kde=True)
        plt.title(f"{col.replace('_', ' ').title()} Distribution")
        save_plot(f"{col}_distribution")

if 'is_long_review' in df.columns:
    print("\nLong Review Stats:")
    print(df['is_long_review'].value_counts(dropna=False))
    sns.countplot(x='is_long_review', data=df)
    plt.title("Long Review Flag Distribution")
    save_plot("long_review_flag_distribution")

if 'is_short_review' in df.columns:
    print("\nShort Review Stats:")
    print(df['is_short_review'].value_counts(dropna=False))
    sns.countplot(x='is_short_review', data=df)
    plt.title("Short Review Flag Distribution")
    save_plot("short_review_flag_distribution")

# =================== BUSINESS & AUTHOR DISTRIBUTIONS ===================
if 'business_name' in df.columns:
    print("\nUnique Businesses:", df['business_name'].nunique())
    top_businesses = df['business_name'].value_counts().head(10)
    print("\nTop 10 Businesses by Review Count:")
    print(top_businesses)
    sns.barplot(x=top_businesses.values, y=top_businesses.index)
    plt.title("Top 10 Businesses by Reviews")
    plt.xlabel("Number of Reviews")
    plt.ylabel("Business Name")
    save_plot("top_10_businesses_review_count")

if 'author_name' in df.columns:
    print("\nUnique Authors:", df['author_name'].nunique())

# =================== CATEGORICAL BREAKDOWNS ===================
for col in ['business_category', 'dataset_type', 'data_source']:
    if col in df.columns:
        print(f"\n{col.replace('_', ' ').title()} Breakdown:")
        print(df[col].value_counts(dropna=False))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f"{col.replace('_', ' ').title()} Distribution")
        save_plot(f"{col}_distribution")

# =================== FLAGS: PHOTO/URL/PHONE/EMAIL ===================
flag_cols = [col for col in ['has_photo', 'has_url', 'has_phone', 'has_email'] if col in df.columns]
for flag in flag_cols:
    print(f"\n{flag.replace('_', ' ').title()} Distribution:")
    print(df[flag].value_counts(dropna=False))
    sns.countplot(x=flag, data=df)
    plt.title(f"{flag.replace('_', ' ').title()} Distribution")
    save_plot(f"{flag}_distribution")

# =================== TEMPORAL ANALYSIS ===================
if 'review_date' in df.columns:
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    print("\nReview Date Range:", df['review_date'].min(), "to", df['review_date'].max())
    if df['review_date'].notnull().any():
        df['review_year'] = df['review_date'].dt.year
        sns.histplot(df['review_year'].dropna(), bins=15)
        plt.title("Review Year Distribution")
        plt.xlabel("Year")
        save_plot("review_year_distribution")
    df['review_month'] = df['review_date'].dt.to_period('M')
    month_counts = df['review_month'].value_counts().sort_index()
    if len(month_counts) > 0:
        month_counts.plot(kind='bar', figsize=(16, 4))
        plt.title("Review Counts by Month")
        plt.xlabel("Month")
        plt.ylabel("Count")
        plt.tight_layout()
        save_plot("review_counts_by_month")

# =================== POLICY VIOLATION BY RATING TABLE ===================
if 'rating' in df.columns and 'policy_violation' in df.columns:
    table = pd.crosstab(df['rating'], df['policy_violation'])
    print("\nPolicy Violation by Rating Table:\n", table)
    table.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title("Policy Violation Type by Review Rating")
    plt.ylabel("Number of Reviews")
    plt.xlabel("Star Rating")
    plt.tight_layout()
    save_plot("policy_violation_by_rating")

# =================== ADVANCED NLP/TEXTUAL ANALYSIS ===================
def clean_text(text):
    text = str(text).lower()
    return "".join(c for c in text if c not in string.punctuation)

def get_top_n_words(texts, n=20):
    all_words = []
    for t in texts:
        all_words.extend(clean_text(t).split())
    return Counter(all_words).most_common(n)

if 'text' in df.columns:
    print("\nGenerating Global WordCloud for all reviews...")
    text_blob = " ".join(clean_text(t) for t in df['text'].dropna())
    wc = WordCloud(width=900, height=400, background_color='white', max_words=150).generate(text_blob)
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('WordCloud - All Reviews')
    save_plot("wordcloud_all_reviews")

    for violation in df['policy_violation'].dropna().unique():
        texts = df[df['policy_violation'] == violation]['text'].dropna().astype(str)
        print(f"\nTop Words for '{violation}' Reviews:")
        print(get_top_n_words(texts, n=20))
        wc = WordCloud(width=900, height=400, background_color='white', max_words=100).generate(
            " ".join(clean_text(t) for t in texts))
        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"WordCloud - {violation.title()} Reviews")
        save_plot(f"wordcloud_{violation}_reviews")

# =================== SENTIMENT ANALYSIS (OPTIONAL, if nltk/vader installed) ===================
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    print("\nCalculating Sentiment Scores for Sample Reviews...")
    df['sentiment'] = df['text'].dropna().apply(lambda x: sia.polarity_scores(str(x))['compound'] if isinstance(x, str) else 0)
    sns.histplot(df['sentiment'], bins=30, kde=True)
    plt.title("Compound Sentiment Distribution")
    plt.xlabel("Sentiment Score (VADER compound)")
    save_plot("compound_sentiment_distribution")
    sentiment_by_policy = df.groupby('policy_violation')['sentiment'].mean()
    print("\nMean Sentiment by Policy Violation:")
    print(sentiment_by_policy)
    sentiment_by_policy.plot(kind='bar')
    plt.title("Mean Sentiment Score by Policy Violation")
    plt.ylabel("Mean Compound Sentiment")
    plt.tight_layout()
    save_plot("mean_sentiment_by_policy_violation")
except Exception as e:
    print(f"\nSentiment analysis skipped: {e}")

# =================== ADDITIONAL ANALYSIS EXAMPLES ===================
if 'text_length' in df.columns and 'policy_violation' in df.columns:
    tl_table = df.groupby('policy_violation')['text_length'].mean()
    wc_table = df.groupby('policy_violation')['word_count'].mean()
    print("\nAverage Text Length by Policy Violation:\n", tl_table)
    print("\nAverage Word Count by Policy Violation:\n", wc_table)
    tl_table.plot(kind='bar')
    plt.title("Average Text Length by Violation")
    plt.ylabel("Average Text Length")
    save_plot("avg_text_length_by_violation")
    wc_table.plot(kind='bar')
    plt.title("Average Word Count by Violation")
    plt.ylabel("Average Word Count")
    save_plot("avg_word_count_by_violation")

print("\nEDA complete! All plots saved to 'eda_plots' folder.")
