#!/usr/bin/env python3
"""
Task 4: Create 10 Professional Visualizations WITHOUT WordCloud
Alternative version for Windows compatibility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from collections import Counter
import re
from matplotlib.patches import Patch

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['figure.figsize'] = [12, 8]
matplotlib.rcParams['font.size'] = 12

class VisualizationGenerator:
    def __init__(self):
        """Initialize with data"""
        print("📊 Loading data for visualization...")
        
        # Load data
        self.reviews_df = pd.read_csv('data/processed/reviews_with_sentiment.csv')
        self.themes_df = pd.read_csv('data/processed/reviews_with_themes.csv')
        self.theme_summary = pd.read_csv('data/processed/theme_summary_by_bank.csv')
        
        print(f"✅ Loaded {len(self.reviews_df)} reviews for visualization")
    
    def plot_1_rating_distribution(self):
        """Plot 1: Rating Distribution by Bank"""
        print("📈 Creating Plot 1: Rating Distribution by Bank...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        banks = ['CBE', 'BOA', 'DASHEN']
        
        for idx, bank in enumerate(banks):
            bank_data = self.reviews_df[self.reviews_df['bank_name'] == bank]
            rating_counts = bank_data['rating'].value_counts().sort_index()
            
            colors = ['#ff6b6b', '#ffa726', '#ffee58', '#90ee90', '#4caf50']
            axes[idx].bar(rating_counts.index, rating_counts.values, color=colors)
            axes[idx].set_title(f'{bank} - Rating Distribution', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Rating (Stars)', fontsize=12)
            axes[idx].set_ylabel('Number of Reviews', fontsize=12)
            axes[idx].set_xticks(range(1, 6))
            axes[idx].grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(rating_counts.values):
                axes[idx].text(rating_counts.index[i], v + 5, str(v), 
                              ha='center', fontweight='bold')
        
        plt.suptitle('Rating Distribution by Ethiopian Bank', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('images/plot1_rating_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_2_sentiment_comparison(self):
        """Plot 2: Sentiment Comparison Across Banks"""
        print("🎭 Creating Plot 2: Sentiment Comparison...")
        
        sentiment_by_bank = self.reviews_df.groupby(['bank_name', 'sentiment_label']).size().unstack()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sentiment_by_bank.plot(kind='bar', ax=ax, 
                              color=['#ff6b6b', '#ffa726', '#4caf50'])
        
        ax.set_title('Sentiment Distribution by Bank', fontsize=16, fontweight='bold')
        ax.set_xlabel('Bank', fontsize=14)
        ax.set_ylabel('Number of Reviews', fontsize=14)
        ax.legend(title='Sentiment', title_fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3, fontsize=10)
        
        plt.tight_layout()
        plt.savefig('images/plot2_sentiment_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_3_average_rating_trend(self):
        """Plot 3: Average Rating Comparison"""
        print("⭐ Creating Plot 3: Average Rating Comparison...")
        
        avg_ratings = self.reviews_df.groupby('bank_name')['rating'].mean().sort_values(ascending=False)
        avg_sentiment = self.reviews_df.groupby('bank_name')['sentiment_score'].mean()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Rating bars
        colors = ['#4caf50', '#4caf50', '#ffa726']
        bars1 = ax1.bar(avg_ratings.index, avg_ratings.values, color=colors)
        ax1.set_title('Average Rating by Bank', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Average Rating (Stars)', fontsize=12)
        ax1.set_ylim(0, 5)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}⭐', ha='center', va='bottom', fontweight='bold')
        
        # Sentiment bars
        bars2 = ax2.bar(avg_sentiment.index, avg_sentiment.values, color=colors)
        ax2.set_title('Average Sentiment Score by Bank', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Average Sentiment Score', fontsize=12)
        ax2.set_ylim(-1, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('images/plot3_average_ratings.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_4_theme_analysis(self):
        """Plot 4: Theme Frequency Analysis"""
        print("🎨 Creating Plot 4: Theme Analysis...")
        
        # Get top themes across all banks
        top_themes = self.theme_summary.groupby('theme')['count'].sum().nlargest(8)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_themes)))
        
        bars = ax.barh(range(len(top_themes)), top_themes.values, color=colors)
        ax.set_yticks(range(len(top_themes)))
        ax.set_yticklabels(top_themes.index, fontsize=11)
        ax.invert_yaxis()
        
        ax.set_title('Top 8 Customer Feedback Themes Across All Banks', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Number of Mentions', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, top_themes.values)):
            ax.text(count + 5, bar.get_y() + bar.get_height()/2, 
                   f'{count}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('images/plot4_theme_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_5_bank_themes_heatmap(self):
        """Plot 5: Bank vs Themes Heatmap"""
        print("🔥 Creating Plot 5: Bank-Themes Heatmap...")
        
        # Create pivot table
        pivot_data = self.theme_summary.pivot_table(
            index='theme', 
            columns='bank_name', 
            values='percentage',
            aggfunc='mean'
        ).fillna(0)
        
        # Get top 6 themes
        top_themes = pivot_data.sum(axis=1).nlargest(6).index
        pivot_data = pivot_data.loc[top_themes]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
        
        # Set labels
        ax.set_xticks(np.arange(len(pivot_data.columns)))
        ax.set_yticks(np.arange(len(pivot_data.index)))
        ax.set_xticklabels(pivot_data.columns, fontsize=12)
        ax.set_yticklabels(pivot_data.index, fontsize=11)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                text = ax.text(j, i, f'{pivot_data.iloc[i, j]:.1f}%',
                              ha="center", va="center", 
                              color="black", fontweight='bold')
        
        ax.set_title('Theme Distribution by Bank (Percentage of Reviews)', 
                    fontsize=16, fontweight='bold', pad=20)
        plt.colorbar(im, ax=ax, label='Percentage (%)')
        plt.tight_layout()
        plt.savefig('images/plot5_themes_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_6_keyword_frequency(self):
        """Plot 6: Keyword Frequency Analysis (Alternative to WordCloud)"""
        print("🔤 Creating Plot 6: Keyword Frequency...")
        
        # Extract common words from reviews
        all_text = ' '.join(self.reviews_df['review_text'].astype(str).str.lower())
        words = re.findall(r'\b[a-z]{4,}\b', all_text)
        
        # Remove common stop words
        stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'they', 'what', 
                     'when', 'where', 'which', 'who', 'will', 'your', 'their', 'could',
                     'would', 'should', 'there', 'about', 'like', 'just', 'very', 'good',
                     'app', 'bank', 'mobile', 'ethiopia', 'ethiopian'}
        
        filtered_words = [w for w in words if w not in stop_words]
        word_counts = Counter(filtered_words).most_common(20)
        
        # Create horizontal bar chart
        words, counts = zip(*word_counts)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        colors = plt.cm.viridis(np.linspace(0, 1, len(words)))
        
        bars = ax.barh(range(len(words)), counts, color=colors)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=11)
        ax.invert_yaxis()
        
        ax.set_title('Top 20 Keywords from Customer Reviews', fontsize=18, fontweight='bold')
        ax.set_xlabel('Frequency', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count + 2, bar.get_y() + bar.get_height()/2, 
                   str(count), va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('images/plot6_keyword_frequency.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_7_sentiment_keywords(self):
        """Plot 7: Keywords by Sentiment"""
        print("🔠 Creating Plot 7: Keywords by Sentiment...")
        
        # Get positive and negative reviews
        positive_reviews = self.reviews_df[self.reviews_df['sentiment_label'] == 'POSITIVE']
        negative_reviews = self.reviews_df[self.reviews_df['sentiment_label'] == 'NEGATIVE']
        
        def extract_top_words(text_series, n=10):
            all_text = ' '.join(text_series.astype(str).str.lower())
            words = re.findall(r'\b[a-z]{4,}\b', all_text)
            stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'they', 'what',
                         'when', 'where', 'which', 'who', 'will', 'your', 'their', 'could',
                         'would', 'should', 'there', 'about', 'like', 'just', 'very', 'good',
                         'app', 'bank', 'mobile', 'ethiopia', 'ethiopian'}
            filtered = [w for w in words if w not in stop_words]
            return dict(Counter(filtered).most_common(n))
        
        pos_words = extract_top_words(positive_reviews['review_text'])
        neg_words = extract_top_words(negative_reviews['review_text'])
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        
        # Positive keywords
        pos_items = list(pos_words.items())
        pos_words_list, pos_counts = zip(*pos_items)
        colors1 = ['#4caf50'] * len(pos_words_list)
        ax1.barh(range(len(pos_words_list)), pos_counts, color=colors1)
        ax1.set_yticks(range(len(pos_words_list)))
        ax1.set_yticklabels(pos_words_list, fontsize=11)
        ax1.invert_yaxis()
        ax1.set_title('Top Positive Review Keywords', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Frequency', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Negative keywords
        neg_items = list(neg_words.items())
        neg_words_list, neg_counts = zip(*neg_items)
        colors2 = ['#ff6b6b'] * len(neg_words_list)
        ax2.barh(range(len(neg_words_list)), neg_counts, color=colors2)
        ax2.set_yticks(range(len(neg_words_list)))
        ax2.set_yticklabels(neg_words_list, fontsize=11)
        ax2.invert_yaxis()
        ax2.set_title('Top Negative Review Keywords', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Frequency', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Keywords in Positive vs Negative Reviews', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('images/plot7_sentiment_keywords.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_8_sentiment_vs_rating(self):
        """Plot 8: Sentiment Score vs Rating Scatter Plot"""
        print("📈 Creating Plot 8: Sentiment vs Rating Correlation...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Map sentiment to colors
        color_map = {'POSITIVE': '#4caf50', 'NEGATIVE': '#ff6b6b', 'NEUTRAL': '#ffa726'}
        colors = self.reviews_df['sentiment_label'].map(color_map)
        
        scatter = ax.scatter(
            self.reviews_df['rating'],
            self.reviews_df['sentiment_score'],
            c=colors,
            alpha=0.6,
            s=50,
            edgecolors='w',
            linewidth=0.5
        )
        
        # Add trend line
        z = np.polyfit(self.reviews_df['rating'], self.reviews_df['sentiment_score'], 1)
        p = np.poly1d(z)
        ax.plot(self.reviews_df['rating'].sort_values(), 
                p(self.reviews_df['rating'].sort_values()), 
                "r--", alpha=0.8, linewidth=2)
        
        ax.set_title('Sentiment Score vs Rating Correlation', fontsize=16, fontweight='bold')
        ax.set_xlabel('Rating (Stars)', fontsize=14)
        ax.set_ylabel('Sentiment Score', fontsize=14)
        ax.set_xticks(range(1, 6))
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        correlation = self.reviews_df['rating'].corr(self.reviews_df['sentiment_score'])
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=12,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Create custom legend
        legend_elements = [
            Patch(facecolor='#4caf50', label='Positive'),
            Patch(facecolor='#ffa726', label='Neutral'),
            Patch(facecolor='#ff6b6b', label='Negative')
        ]
        ax.legend(handles=legend_elements, title='Sentiment', title_fontsize=12)
        
        plt.tight_layout()
        plt.savefig('images/plot8_sentiment_vs_rating.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_9_performance_issue_breakdown(self):
        """Plot 9: Performance Issues Breakdown by Bank"""
        print("⚡ Creating Plot 9: Performance Issues Analysis...")
        
        # Filter for performance issues
        perf_issues = self.themes_df[self.themes_df['themes'].str.contains('performance', case=False)]
        
        if len(perf_issues) > 0:
            perf_by_bank = perf_issues.groupby('bank_name').size()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create pie chart
            colors = ['#ff6b6b', '#ffa726', '#4caf50']
            wedges, texts, autotexts = ax.pie(
                perf_by_bank.values,
                labels=perf_by_bank.index,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 12}
            )
            
            # Make autotexts bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('Performance Issue Distribution by Bank', 
                        fontsize=16, fontweight='bold', pad=20)
            
        else:
            # Create bar chart showing issue types
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Count different issue types
            issue_types = self.themes_df['themes'].str.split(', ').explode().value_counts().head(5)
            
            colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(issue_types)))
            bars = ax.bar(range(len(issue_types)), issue_types.values, color=colors)
            
            ax.set_xticks(range(len(issue_types)))
            ax.set_xticklabels(issue_types.index, fontsize=11, rotation=45, ha='right')
            ax.set_title('Top 5 Issue Types Across All Banks', fontsize=16, fontweight='bold')
            ax.set_ylabel('Number of Reviews', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       str(int(height)), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('images/plot9_performance_issues.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def plot_10_comprehensive_summary(self):
        """Plot 10: Comprehensive Bank Performance Summary"""
        print("📋 Creating Plot 10: Comprehensive Summary...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Average ratings
        avg_ratings = self.reviews_df.groupby('bank_name')['rating'].mean()
        colors = ['#4caf50' if r > 4 else '#ffa726' if r > 3 else '#ff6b6b' for r in avg_ratings.values]
        ax1.bar(avg_ratings.index, avg_ratings.values, color=colors)
        ax1.set_title('Average Ratings', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Stars', fontsize=12)
        ax1.set_ylim(0, 5)
        ax1.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(avg_ratings.values):
            ax1.text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
        
        # 2. Sentiment distribution
        sentiment_pct = self.reviews_df['sentiment_label'].value_counts(normalize=True) * 100
        colors = ['#4caf50', '#ff6b6b', '#ffa726']
        ax2.pie(sentiment_pct.values, labels=sentiment_pct.index, colors=colors,
               autopct='%1.1f%%', startangle=90)
        ax2.set_title('Overall Sentiment Distribution', fontsize=14, fontweight='bold')
        
        # 3. Review counts
        review_counts = self.reviews_df['bank_name'].value_counts()
        ax3.bar(review_counts.index, review_counts.values, color=['#2196f3', '#2196f3', '#2196f3'])
        ax3.set_title('Number of Reviews Analyzed', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Count', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(review_counts.values):
            ax3.text(i, v + 10, str(v), ha='center', fontweight='bold')
        
        # 4. Performance comparison table
        summary_data = []
        for bank in ['CBE', 'BOA', 'DASHEN']:
            bank_data = self.reviews_df[self.reviews_df['bank_name'] == bank]
            summary_data.append([
                bank,
                f"{bank_data['rating'].mean():.2f}",
                f"{(bank_data['sentiment_label'] == 'POSITIVE').mean()*100:.1f}%",
                len(bank_data)
            ])
        
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=summary_data,
                         colLabels=['Bank', 'Avg Rating', '% Positive', 'Reviews'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2, 0.2, 0.3, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        ax4.set_title('Bank Performance Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Comprehensive Bank App Performance Analysis', fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('images/plot10_comprehensive_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        return fig
    
    def generate_all_plots(self):
        """Generate all 10 plots"""
        print("=" * 60)
        print("🎨 GENERATING 10 PROFESSIONAL VISUALIZATIONS")
        print("=" * 60)
        
        # Create images directory
        import os
        os.makedirs('images', exist_ok=True)
        
        # Generate all plots
        plots = []
        
        plots.append(self.plot_1_rating_distribution())
        plots.append(self.plot_2_sentiment_comparison())
        plots.append(self.plot_3_average_rating_trend())
        plots.append(self.plot_4_theme_analysis())
        plots.append(self.plot_5_bank_themes_heatmap())
        plots.append(self.plot_6_keyword_frequency())
        plots.append(self.plot_7_sentiment_keywords())
        plots.append(self.plot_8_sentiment_vs_rating())
        plots.append(self.plot_9_performance_issue_breakdown())
        plots.append(self.plot_10_comprehensive_summary())
        
        print("=" * 60)
        print("✅ ALL 10 VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 Images saved to 'images/' folder:")
        for i in range(1, 11):
            print(f"   plot{i}_*.png")
        
        return plots

def main():
    """Main function to generate visualizations"""
    print("📊 TASK 4: VISUALIZATION GENERATION")
    print("Creating 10 plots without WordCloud for Windows compatibility...")
    
    # Initialize generator
    generator = VisualizationGenerator()
    
    # Generate all plots
    generator.generate_all_plots()
    
    print("\n🎉 TASK 4 VISUALIZATION COMPLETED!")
    print("📈 10 professional plots ready for final report")

if __name__ == "__main__":
    main()