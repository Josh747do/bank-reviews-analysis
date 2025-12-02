#!/usr/bin/env python3
"""
Simplified Thematic Analysis for Bank Reviews
Extract keywords and group into themes without visualization imports
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter

class ThematicAnalyzer:
    def __init__(self):
        """Initialize thematic analysis tools"""
        self.theme_categories = {
            'login_issues': [
                'login', 'password', 'sign in', 'authentication', 'biometric',
                'fingerprint', 'face id', 'cannot login', 'login failed'
            ],
            'transaction_problems': [
                'transfer', 'payment', 'transaction', 'failed', 'money',
                'send money', 'receive money', 'transaction failed', 'deducted'
            ],
            'performance_issues': [
                'slow', 'crash', 'freeze', 'loading', 'lag', 'hanging',
                'not responding', 'crashes', 'freezes', 'slow loading'
            ],
            'ui_ux_feedback': [
                'interface', 'design', 'navigation', 'easy', 'beautiful',
                'user friendly', 'layout', 'menu', 'hard to use', 'confusing'
            ],
            'feature_requests': [
                'should have', 'please add', 'would like', 'missing',
                'need feature', 'wish had', 'add feature', 'new feature'
            ],
            'customer_support': [
                'support', 'help', 'response', 'service', 'contact',
                'customer service', 'help desk', 'no response', 'slow response'
            ],
            'security_concerns': [
                'secure', 'security', 'safe', 'trust', 'hack',
                'privacy', 'data', 'protection', 'fraud'
            ],
            'app_updates': [
                'update', 'version', 'new version', 'after update',
                'latest update', 'upgrade', 'downgrade'
            ]
        }
    
    def extract_keywords_tfidf(self, texts, max_features=50):
        """Extract important keywords using TF-IDF"""
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Use TF-IDF to find important words
        tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)  # Single words and bigrams
        )
        
        tfidf_matrix = tfidf.fit_transform(cleaned_texts)
        feature_names = tfidf.get_feature_names_out()
        
        # Get average TF-IDF scores
        avg_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        
        # Create keyword-score pairs
        keywords_with_scores = list(zip(feature_names, avg_scores))
        keywords_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [kw for kw, score in keywords_with_scores[:max_features]]
    
    def clean_text(self, text):
        """Clean text for analysis"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def assign_themes(self, text):
        """Assign themes to text based on keyword matching"""
        text_lower = self.clean_text(text)
        assigned_themes = []
        
        for theme, keywords in self.theme_categories.items():
            # Check if any keyword from this theme appears in text
            for keyword in keywords:
                if keyword in text_lower:
                    assigned_themes.append(theme)
                    break  # No need to check other keywords for this theme
        
        return assigned_themes if assigned_themes else ['general_feedback']
    
    def analyze_themes_by_bank(self, df):
        """Analyze themes for each bank"""
        print("🎨 Starting thematic analysis...")
        
        bank_themes = {}
        
        for bank in df['bank_name'].unique():
            print(f"\n🏦 Analyzing themes for {bank}...")
            bank_reviews = df[df['bank_name'] == bank]
            
            # Extract keywords
            keywords = self.extract_keywords_tfidf(bank_reviews['review_text'].tolist())
            
            # Assign themes to each review
            themes_list = []
            for text in bank_reviews['review_text']:
                themes = self.assign_themes(text)
                themes_list.extend(themes)
            
            # Count theme frequency
            theme_counts = Counter(themes_list)
            
            bank_themes[bank] = {
                'top_keywords': keywords[:20],  # Top 20 keywords
                'theme_distribution': dict(theme_counts.most_common()),
                'total_reviews': len(bank_reviews)
            }
            
            print(f"   📊 Top 5 themes: {list(theme_counts.keys())[:5]}")
            print(f"   🔑 Top 5 keywords: {keywords[:5]}")
        
        return bank_themes
    
    def save_thematic_results(self, df, bank_themes):
        """Save thematic analysis results"""
        # Add themes to each review
        print("\n💾 Saving thematic analysis results...")
        
        themes_data = []
        for idx, row in df.iterrows():
            themes = self.assign_themes(row['review_text'])
            
            theme_record = {
                'review_id': row.get('review_id', f"review_{idx}"),
                'bank_name': row['bank_name'],
                'review_text': row['review_text'],
                'sentiment_label': row.get('sentiment_label', ''),
                'rating': row['rating'],
                'themes': ', '.join(themes),
                'primary_theme': themes[0] if themes else 'general_feedback'
            }
            themes_data.append(theme_record)
        
        # Save to CSV
        themes_df = pd.DataFrame(themes_data)
        themes_df.to_csv('data/processed/reviews_with_themes.csv', index=False)
        
        # Save bank-level theme summaries
        theme_summaries = []
        for bank, data in bank_themes.items():
            for theme, count in data['theme_distribution'].items():
                theme_summaries.append({
                    'bank_name': bank,
                    'theme': theme,
                    'count': count,
                    'percentage': (count / data['total_reviews']) * 100
                })
        
        summary_df = pd.DataFrame(theme_summaries)
        summary_df.to_csv('data/processed/theme_summary_by_bank.csv', index=False)
        
        return themes_df

def main():
    """Main function to run thematic analysis"""
    print("=" * 60)
    print("🎨 TASK 2: THEMATIC ANALYSIS")
    print("=" * 60)
    
    # Load sentiment data
    print("📁 Loading sentiment analysis data...")
    try:
        df = pd.read_csv('data/processed/reviews_with_sentiment.csv')
        print(f"✅ Loaded {len(df)} reviews with sentiment labels")
    except FileNotFoundError:
        print("❌ Could not find sentiment data. Please run sentiment analysis first.")
        return
    
    # Initialize analyzer
    analyzer = ThematicAnalyzer()
    
    # Analyze themes
    bank_themes = analyzer.analyze_themes_by_bank(df)
    
    # Save results
    themes_df = analyzer.save_thematic_results(df, bank_themes)
    
    print(f"💾 Saved thematic analysis results")
    print(f"   - Individual reviews with themes: data/processed/reviews_with_themes.csv")
    print(f"   - Bank-level theme summaries: data/processed/theme_summary_by_bank.csv")
    
    # Show final summary
    print("\n📊 THEMATIC ANALYSIS SUMMARY:")
    print("=" * 40)
    
    for bank, data in bank_themes.items():
        print(f"\n🏦 {bank}:")
        print(f"   Total reviews: {data['total_reviews']}")
        print(f"   Top 3 themes:")
        for theme, count in list(data['theme_distribution'].items())[:3]:
            percentage = (count / data['total_reviews']) * 100
            print(f"     - {theme}: {count} reviews ({percentage:.1f}%)")
        print(f"   Top 5 keywords: {', '.join(data['top_keywords'][:5])}")
    
    print(f"\n✅ TASK 2 - THEMATIC ANALYSIS COMPLETED!")

if __name__ == "__main__":
    main()