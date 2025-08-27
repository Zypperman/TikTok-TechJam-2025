"""
Unified Dataset Processor V2 for TikTok TechJam 2025
Properly combines McAuley Lab and Google dataset with different labeling formats
"""

import pandas as pd
import numpy as np
import json
import gzip
import os
from typing import List, Dict, Optional
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class UnifiedDatasetProcessorV2:
    def __init__(self, output_dir: str = "collected_data"):
        self.output_dir = output_dir
        self.ensure_output_dir()
        
    def ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
    
    def process_google_dataset(self, csv_path: str = "google_dataset/kaggle/reviews.csv") -> pd.DataFrame:
        """
        Process the Google dataset (reviews.csv + linked images)
        """
        print("🔍 Processing Google Dataset...")
        
        try:
            # Check if the CSV file exists
            if not os.path.exists(csv_path):
                print(f"❌ Google dataset not found at: {csv_path}")
                return pd.DataFrame()
            
            print(f"📄 Processing: {csv_path}")
            
            # Read reviews CSV directly
            df = pd.read_csv(csv_path)
            
            print(f"✅ Loaded {len(df)} reviews from Google dataset")
            print(f"📊 Columns found: {list(df.columns)}")
            
            # Standardize columns
            df['data_source'] = 'google_dataset'
            df['collection_date'] = pd.Timestamp.now().date()
            df['review_id'] = range(1, len(df) + 1)
            
            # Ensure required columns exist
            required_columns = ['business_name', 'author_name', 'text', 'rating', 'rating_category']
            for col in required_columns:
                if col not in df.columns:
                    print(f"⚠️ Missing column: {col}")
                    if col == 'rating_category':
                        df[col] = 'general'
                    else:
                        df[col] = ''
            
            # Clean text data
            df['text'] = df['text'].fillna('').astype(str)
            df = df[df['text'].str.strip() != '']
            
            # Convert rating to numeric
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df = df.dropna(subset=['rating'])
            
            # Add text features
            df['text_length'] = df['text'].str.len()
            df['word_count'] = df['text'].str.split().str.len()
            
            # Process photo information
            df['has_photo'] = df['photo'].notna() & (df['photo'] != '')
            df['photo_count'] = df['photo'].apply(lambda x: 1 if pd.notna(x) and x != '' else 0)
            
            # Add dataset type
            df['dataset_type'] = 'google_restaurant_review'
            df['business_category'] = 'Restaurant'
            df['business_address'] = 'Unknown'
            df['business_latitude'] = np.nan
            df['business_longitude'] = np.nan
            df['business_price'] = 'Unknown'
            df['review_date'] = pd.Timestamp.now()
            
            print(f"✅ Processed {len(df)} Google reviews")
            
            # Save processed Google data
            output_path = os.path.join(self.output_dir, "google_processed.csv")
            df.to_csv(output_path, index=False)
            print(f"💾 Saved to: {output_path}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error processing Google dataset: {e}")
            print(f"   Full error details: {str(e)}")
            return pd.DataFrame()
    
    def process_mcauleylab_dataset(self, review_path: str = "review-Alabama.json.gz", 
                                  meta_path: str = "google_dataset/google/meta-Alabama.json.gz") -> pd.DataFrame:
        """
        Process the McAuley Lab Google Local dataset
        """
        print("🔍 Processing McAuley Lab Dataset...")
        
        try:
            reviews_data = []
            business_data = {}
            
            # First, load business metadata from meta file
            if os.path.exists(meta_path):
                print("📊 Loading business metadata...")
                with gzip.open(meta_path, 'rt', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if 'name' in data and 'gmap_id' in data:
                                business_data[data['gmap_id']] = data
                        except:
                            continue
                print(f"✅ Loaded {len(business_data)} business metadata records")
            else:
                print("⚠️ Meta file not found, proceeding without business metadata")
            
            # Read the gzipped JSON file for reviews
            if os.path.exists(review_path):
                print(f"📊 Loading reviews from: {review_path}")
                with gzip.open(review_path, 'rt', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 50000:  # Limit to first 50k reviews for demo
                            break
                        
                        try:
                            data = json.loads(line.strip())
                            
                            # Check if this is a review
                            if 'text' in data and 'rating' in data:
                                # This is a review
                                reviews_data.append(data)
                                
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            continue
            else:
                print(f"❌ Review file not found at: {review_path}")
                return pd.DataFrame()
            
            print(f"🔍 Found {len(reviews_data)} reviews and {len(business_data)} businesses")
            
            if not reviews_data:
                print("❌ No review data found")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(reviews_data)
            
            # Standardize columns
            column_mapping = {
                'user_id': 'author_name',
                'name': 'reviewer_name',
                'time': 'review_timestamp',
                'rating': 'rating',
                'text': 'text',
                'gmap_id': 'business_id'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Add missing columns
            df['business_name'] = df['business_id'].map(
                lambda x: business_data.get(x, {}).get('name', 'Unknown Business')
            )
            df['rating_category'] = 'general'
            df['photo'] = df.get('pics', '').apply(lambda x: len(x) if isinstance(x, list) else 0)
            df['data_source'] = 'mcauleylab_google_local'
            df['collection_date'] = pd.Timestamp.now().date()
            df['review_id'] = range(len(df) + 1, len(df) + len(df) + 1)
            
            # Convert timestamp to readable date
            df['review_date'] = pd.to_datetime(df['review_timestamp'], unit='ms', errors='coerce')
            
            # Clean text data
            df['text'] = df['text'].fillna('').astype(str)
            df = df[df['text'].str.strip() != '']
            
            # Convert rating to numeric
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df = df.dropna(subset=['rating'])
            
            # Add text features
            df['text_length'] = df['text'].str.len()
            df['word_count'] = df['text'].str.split().str.len()
            
            # Add business metadata
            df['business_category'] = df['business_id'].map(
                lambda x: business_data.get(x, {}).get('category', ['Unknown'])[0] if business_data.get(x, {}).get('category') else 'Unknown'
            )
            df['business_address'] = df['business_id'].map(
                lambda x: business_data.get(x, {}).get('address', 'Unknown')
            )
            df['business_latitude'] = df['business_id'].map(
                lambda x: business_data.get(x, {}).get('latitude', np.nan)
            )
            df['business_longitude'] = df['business_id'].map(
                lambda x: business_data.get(x, {}).get('longitude', np.nan)
            )
            df['business_price'] = df['business_id'].map(
                lambda x: business_data.get(x, {}).get('price', 'Unknown')
            )
            
            # Add dataset type
            df['dataset_type'] = 'mcauleylab_google_local'
            df['has_photo'] = df['photo'] > 0
            df['photo_count'] = df['photo']
            
            print(f"✅ Processed {len(df)} McAuley Lab reviews")
            
            # Save processed McAuley Lab data
            output_path = os.path.join(self.output_dir, "mcauleylab_processed.csv")
            df.to_csv(output_path, index=False)
            print(f"💾 Saved to: {output_path}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error processing McAuley Lab dataset: {e}")
            print(f"   Full error details: {str(e)}")
            return pd.DataFrame()
    
    def merge_datasets(self, google_df: pd.DataFrame, mcauleylab_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the two datasets into one unified dataset
        """
        print("🔄 Merging datasets...")
        
        if google_df.empty and mcauleylab_df.empty:
            print("❌ No datasets to merge")
            return pd.DataFrame()
        
        # Prepare datasets for merging
        datasets = []
        
        if not google_df.empty:
            # Select and rename Google columns to match unified schema
            google_unified = google_df[['review_id', 'business_name', 'author_name', 'text', 'rating', 'rating_category', 
                                      'photo_count', 'data_source', 'collection_date', 'text_length', 'word_count',
                                      'dataset_type', 'business_category', 'business_address', 'business_latitude', 
                                      'business_longitude', 'business_price', 'review_date', 'has_photo']].copy()
            datasets.append(google_unified)
            print(f"📊 Added {len(google_unified)} Google reviews")
        
        if not mcauleylab_df.empty:
            # Select and rename McAuley Lab columns to match unified schema
            mcauleylab_unified = mcauleylab_df[['review_id', 'business_name', 'author_name', 'text', 'rating', 'rating_category', 
                                               'photo_count', 'data_source', 'collection_date', 'text_length', 'word_count',
                                               'dataset_type', 'business_category', 'business_address', 'business_latitude', 
                                               'business_longitude', 'business_price', 'review_date', 'has_photo']].copy()
            datasets.append(mcauleylab_unified)
            print(f"📊 Added {len(mcauleylab_unified)} McAuley Lab reviews")
        
        # Merge datasets
        merged_df = pd.concat(datasets, ignore_index=True)
        
        # Reassign review IDs
        merged_df['review_id'] = range(1, len(merged_df) + 1)
        
        # Add unified features
        merged_df['is_long_review'] = merged_df['text_length'] > 100
        merged_df['is_short_review'] = merged_df['text_length'] < 20
        
        # Add policy violation labels (basic heuristics)
        merged_df['policy_violation'] = 'valid'  # Default to valid
        
        # Simple heuristics for policy violations
        text_lower = merged_df['text'].str.lower()
        
        # Advertisement detection
        ad_keywords = ['website', 'www.', 'http', 'call us', 'visit us', 'check out', 'deal', 'offer', 'discount']
        ad_mask = text_lower.str.contains('|'.join(ad_keywords), na=False)
        merged_df.loc[ad_mask, 'policy_violation'] = 'advertisement'
        
        # Irrelevant content detection
        irrelevant_keywords = ['weather', 'traffic', 'netflix', 'movie', 'dog', 'cat', 'pet', 'vacation', 'workout']
        irrelevant_mask = text_lower.str.contains('|'.join(irrelevant_keywords), na=False)
        merged_df.loc[irrelevant_mask & (merged_df['policy_violation'] == 'valid'), 'policy_violation'] = 'irrelevant'
        
        # Rant detection (very negative reviews)
        rant_mask = (merged_df['rating'] <= 2) & (merged_df['text_length'] > 50)
        merged_df.loc[rant_mask & (merged_df['policy_violation'] == 'valid'), 'policy_violation'] = 'rant'
        
        # Add confidence scores
        merged_df['violation_confidence'] = 0.0
        merged_df.loc[merged_df['policy_violation'] != 'valid', 'violation_confidence'] = 0.7
        
        print(f"✅ Merged dataset contains {len(merged_df)} total reviews")
        
        return merged_df
    
    def save_unified_dataset(self, df: pd.DataFrame) -> None:
        """
        Save the unified dataset and generate summary
        """
        print("💾 Saving unified dataset...")
        
        # Save main dataset
        main_output_path = os.path.join(self.output_dir, "unified_reviews_dataset.csv")
        df.to_csv(main_output_path, index=False)
        print(f"💾 Main dataset saved to: {main_output_path}")
        
        # Save ML-ready version (with additional features)
        ml_df = df.copy()
        
        # Add more ML features
        ml_df['has_url'] = ml_df['text'].str.contains(r'http[s]?://', regex=True, na=False)
        ml_df['has_phone'] = ml_df['text'].str.contains(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', regex=True, na=False)
        ml_df['has_email'] = ml_df['text'].str.contains(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', regex=True, na=False)
        ml_df['exclamation_count'] = ml_df['text'].str.count(r'\!')
        ml_df['question_count'] = ml_df['text'].str.count(r'\?')
        ml_df['capitalization_ratio'] = ml_df['text'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
        
        ml_output_path = os.path.join(self.output_dir, "ml_ready_dataset.csv")
        ml_df.to_csv(ml_output_path, index=False)
        print(f"💾 ML-ready dataset saved to: {ml_output_path}")
        
        # Generate summary
        self.generate_summary(df)
    
    def generate_summary(self, df: pd.DataFrame) -> None:
        """
        Generate comprehensive summary of the unified dataset
        """
        print("\n" + "="*60)
        print("UNIFIED DATASET SUMMARY")
        print("="*60)
        
        print(f"Total Reviews: {len(df):,}")
        print(f"Data Sources: {df['data_source'].nunique()}")
        print(f"Dataset Types: {df['dataset_type'].nunique()}")
        print(f"Businesses: {df['business_name'].nunique()}")
        print(f"Authors: {df['author_name'].nunique()}")
        
        print(f"\nData Source Distribution:")
        source_counts = df['data_source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count:,} reviews ({count/len(df)*100:.1f}%)")
        
        print(f"\nDataset Type Distribution:")
        type_counts = df['dataset_type'].value_counts()
        for dtype, count in type_counts.items():
            print(f"  {dtype}: {count:,} reviews ({count/len(df)*100:.1f}%)")
        
        print(f"\nRating Distribution:")
        rating_counts = df['rating'].value_counts().sort_index()
        for rating, count in rating_counts.items():
            print(f"  {rating} stars: {count:,} reviews ({count/len(df)*100:.1f}%)")
        
        print(f"\nPolicy Violations:")
        violation_counts = df['policy_violation'].value_counts()
        for violation, count in violation_counts.items():
            print(f"  {violation}: {count:,} reviews ({count/len(df)*100:.1f}%)")
        
        print(f"\nText Length Statistics:")
        print(f"  Average: {df['text_length'].mean():.1f} characters")
        print(f"  Median: {df['text_length'].median():.1f} characters")
        print(f"  Min: {df['text_length'].min()} characters")
        print(f"  Max: {df['text_length'].max()} characters")
        
        print(f"\nWord Count Statistics:")
        print(f"  Average: {df['word_count'].mean():.1f} words")
        print(f"  Median: {df['word_count'].median():.1f} words")
        print(f"  Min: {df['word_count'].min()} words")
        print(f"  Max: {df['word_count'].max()} words")
        
        print(f"\nBusiness Categories:")
        category_counts = df['business_category'].value_counts().head(10)
        for category, count in category_counts.items():
            print(f"  {category}: {count:,} reviews")
        
        # Save summary to file
        summary_path = os.path.join(self.output_dir, "unified_dataset_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("UNIFIED DATASET SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Total Reviews: {len(df):,}\n")
            f.write(f"Data Sources: {df['data_source'].nunique()}\n")
            f.write(f"Dataset Types: {df['dataset_type'].nunique()}\n")
            f.write(f"Businesses: {df['business_name'].nunique()}\n")
            f.write(f"Authors: {df['author_name'].nunique()}\n")
            f.write(f"\nData Source Distribution:\n")
            for source, count in source_counts.items():
                f.write(f"  {source}: {count:,} reviews ({count/len(df)*100:.1f}%)\n")
            f.write(f"\nPolicy Violations:\n")
            for violation, count in violation_counts.items():
                f.write(f"  {violation}: {count:,} reviews ({count/len(df)*100:.1f}%)\n")
        
        print(f"\n💾 Summary saved to: {summary_path}")
        print("="*60)

def main():
    """
    Main function to run the unified dataset processing pipeline
    """
    print("🚀 Starting Unified Dataset Processing Pipeline V2")
    print("="*60)
    
    # Initialize processor
    processor = UnifiedDatasetProcessorV2()
    
    # Process Google dataset
    print("\n📥 PHASE 1: Processing Google Dataset")
    print("-" * 40)
    google_df = processor.process_google_dataset()
    
    # Process McAuley Lab dataset
    print("\n📥 PHASE 2: Processing McAuley Lab Dataset")
    print("-" * 40)
    mcauleylab_df = processor.process_mcauleylab_dataset()
    
    # Merge datasets
    print("\n🔄 PHASE 3: Merging Datasets")
    print("-" * 40)
    unified_df = processor.merge_datasets(google_df, mcauleylab_df)
    
    if not unified_df.empty:
        # Save unified dataset
        print("\n💾 PHASE 4: Saving Results")
        print("-" * 40)
        processor.save_unified_dataset(unified_df)
        
        print(f"\n✅ Unified dataset processing completed successfully!")
        print(f"📊 Final dataset: {len(unified_df):,} reviews")
        print(f"💾 All files saved to: {processor.output_dir}")
        
        # Display sample of unified dataset
        print(f"\n📋 Sample of unified dataset:")
        print(unified_df[['review_id', 'business_name', 'author_name', 'text', 'rating', 'policy_violation', 'data_source']].head())
        
    else:
        print("\n❌ Dataset processing failed - no data generated")

if __name__ == "__main__":
    main()
