#!/usr/bin/env python3
"""
Simplified Sentiment Analysis for Bank Reviews
Uses reliable models that work without spaCy
"""

import pandas as pd
import numpy as np
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re

class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analysis models"""
        print("🔄 Loading sentiment analysis models...")
        
        # Initialize multiple models for robust analysis
        try:
            self.transformers_sentiment = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True
            )
            print("✅ Transformers model loaded")
        except Exception as e:
            print(f"❌ Transformers model failed: {e}")
            self.transformers_sentiment = None
        
        # VADER for social media/text sentiment
        self.vader_analyzer = SentimentIntensityAnalyzer()
        print("✅ VADER model loaded")
    
    def clean_text(self, text):
        """Clean and preprocess text for analysis"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters and extra spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_sentiment_transformers(self, text):
        """Analyze sentiment using Transformers (DistilBERT)"""
        if not text or len(text.strip()) < 3:
            return {'label': 'NEUTRAL', 'score': 0.5}
        
        try:
            # Truncate very long texts
            truncated_text = text[:512]
            result = self.transformers_sentiment(truncated_text)[0]
            return result
        except:
            return {'label': 'NEUTRAL', 'score': 0.5}
    
    def analyze_sentiment_vader(self, text):
        """Analyze sentiment using VADER"""
        if not text:
            return {'compound': 0.0, 'neg': 0.0, 'neu': 0.0, 'pos': 0.0}
        
        return self.vader_analyzer.polarity_scores(text)
    
    def analyze_sentiment_textblob(self, text):
        """Analyze sentiment using TextBlob"""
        if not text:
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def get_final_sentiment(self, text):
        """Get final sentiment label using ensemble approach"""
        if not text or len(text.strip()) < 3:
            return 'NEUTRAL', 0.5
        
        # Get predictions from all models
        transformers_result = self.analyze_sentiment_transformers(text)
        vader_result = self.analyze_sentiment_vader(text)
        textblob_result = self.analyze_sentiment_textblob(text)
        
        # Ensemble voting
        scores = []
        
        # Transformers score
        if transformers_result['label'] == 'POSITIVE':
            scores.append(transformers_result['score'])
        else:
            scores.append(-transformers_result['score'])
        
        # VADER score (compound from -1 to 1)
        scores.append(vader_result['compound'])
        
        # TextBlob score (polarity from -1 to 1)
        scores.append(textblob_result['polarity'])
        
        # Average the scores
        final_score = sum(scores) / len(scores)
        
        # Classify based on final score
        if final_score > 0.1:
            return 'POSITIVE', final_score
        elif final_score < -0.1:
            return 'NEGATIVE', final_score
        else:
            return 'NEUTRAL', final_score
    
    def analyze_reviews(self, df):
        """Analyze sentiment for all reviews in DataFrame"""
        print("🎯 Starting sentiment analysis...")
        
        results = []
        
        for idx, row in df.iterrows():
            review_text = self.clean_text(row['review_text'])
            
            # Get final sentiment
            sentiment_label, sentiment_score = self.get_final_sentiment(review_text)
            
            # Store results
            result = {
                'review_id': row.get('review_id', f"review_{idx}"),
                'review_text': review_text,
                'original_text': row['review_text'],
                'rating': row['rating'],
                'bank_name': row['bank_name'],
                'sentiment_label': sentiment_label,
                'sentiment_score': round(sentiment_score, 4),
                'date': row.get('date', '')
            }
            
            results.append(result)
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(df)} reviews...")
        
        return pd.DataFrame(results)

def main():
    """Main function to run sentiment analysis"""
    print("=" * 60)
    print("🎭 TASK 2: SENTIMENT ANALYSIS")
    print("=" * 60)
    
    # Load the data
    print("📁 Loading review data...")
    try:
        df = pd.read_csv('data/raw/all_banks_reviews.csv')
        print(f"✅ Loaded {len(df)} reviews from all_banks_reviews.csv")
    except FileNotFoundError:
        print("❌ Could not find review data. Please run Task 1 first.")
        return
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze sentiment
    print("\n⏳ Analyzing sentiment (this will take 5-10 minutes)...")
    results_df = analyzer.analyze_reviews(df)
    
    # Save results
    output_file = 'data/processed/reviews_with_sentiment.csv'
    results_df.to_csv(output_file, index=False)
    print(f"💾 Saved sentiment analysis results to {output_file}")
    
    # Show summary
    print("\n📊 SENTIMENT ANALYSIS SUMMARY:")
    print("=" * 40)
    
    sentiment_summary = results_df.groupby(['bank_name', 'sentiment_label']).size().unstack(fill_value=0)
    print(sentiment_summary)
    
    print(f"\n📈 Overall Sentiment Distribution:")
    overall_sentiment = results_df['sentiment_label'].value_counts()
    for sentiment, count in overall_sentiment.items():
        percentage = (count / len(results_df)) * 100
        print(f"   {sentiment}: {count} reviews ({percentage:.1f}%)")
    
    # Compare sentiment vs rating
    print(f"\n🔍 Sentiment vs Rating Analysis:")
    sentiment_rating = results_df.groupby('sentiment_label')['rating'].mean().round(2)
    print(sentiment_rating)
    
    print(f"\n✅ TASK 2 - SENTIMENT ANALYSIS COMPLETED!")

if __name__ == "__main__":
    main()