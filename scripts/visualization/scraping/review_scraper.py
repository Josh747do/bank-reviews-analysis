#!/usr/bin/env python3
"""
Limited scraping - only 500 reviews per bank as required
"""

import pandas as pd
from google_play_scraper import reviews_all, Sort
import time
from datetime import datetime

# CORRECT BANK APP IDs
BANK_APPS = {
    'cbe': 'com.combanketh.mobilebanking',
    'boa': 'com.boa.boaMobileBanking', 
    'dashen': 'com.dashen.dashensuperapp'
}

def scrape_limited_reviews(app_id, bank_name, count=500):
    """
    Scrape exactly the required number of reviews
    """
    print(f"📱 Scraping {count} reviews for {bank_name}...")
    
    try:
        reviews = reviews_all(
            app_id,
            lang='en',
            country='et',
            sort=Sort.NEWEST,
            count=count,  # Exactly 500 as required
            sleep_milliseconds=2000
        )
        
        # Take only the required number
        reviews = reviews[:count]
        print(f"✅ Successfully scraped {len(reviews)} reviews for {bank_name}")
        return reviews
        
    except Exception as e:
        print(f"❌ Error scraping {bank_name}: {e}")
        return []

def main():
    """
    Main function to scrape exactly 500 reviews per bank
    """
    print("🚀 Starting LIMITED Google Play Store review scraping...")
    print("📊 Target: 500 reviews per bank (1500 total)")
    print("=" * 50)
    
    all_reviews = []
    
    for bank_name, app_id in BANK_APPS.items():
        print(f"\n🏦 Processing {bank_name.upper()}...")
        
        # Scrape exactly 500 reviews
        raw_reviews = scrape_limited_reviews(app_id, bank_name, 500)
        
        if raw_reviews:
            # Process the limited reviews
            processed_reviews = []
            for review in raw_reviews:
                processed_review = {
                    'review_id': review.get('reviewId', ''),
                    'review_text': review.get('content', ''),
                    'rating': review.get('score', 0),
                    'date': review.get('at', ''),
                    'bank_name': bank_name.upper(),
                    'app_name': f"{bank_name.upper()} Mobile",
                    'source': 'Google Play Store',
                    'thumbs_up': review.get('thumbsUpCount', 0)
                }
                processed_reviews.append(processed_review)
            
            all_reviews.extend(processed_reviews)
            
            # Save individual bank data
            df = pd.DataFrame(processed_reviews)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            individual_filename = f"data/raw/{bank_name}_reviews.csv"
            df.to_csv(individual_filename, index=False, encoding='utf-8')
            print(f"💾 Saved {len(df)} reviews to {individual_filename}")
            
            # Show summary
            ratings = [r['rating'] for r in processed_reviews]
            avg_rating = sum(ratings) / len(ratings)
            print(f"   ⭐ Average rating: {avg_rating:.2f}/5")
            print(f"   📅 Date range: {processed_reviews[0]['date']} to {processed_reviews[-1]['date']}")
        
        # Be respectful - add delay between requests
        time.sleep(5)
    
    # Save combined data
    if all_reviews:
        combined_df = pd.DataFrame(all_reviews)
        combined_df['date'] = pd.to_datetime(combined_df['date']).dt.strftime('%Y-%m-%d')
        combined_filename = "data/raw/all_banks_reviews.csv"
        combined_df.to_csv(combined_filename, index=False, encoding='utf-8')
        
        print(f"\n🎉 LIMITED SCRAPING COMPLETED!")
        print(f"📈 Total reviews collected: {len(all_reviews)}")
        print(f"🏦 Banks processed: {len(BANK_APPS)}")
        
        # Final summary
        print(f"\n📊 FINAL SUMMARY:")
        summary = combined_df.groupby('bank_name').agg({
            'rating': ['count', 'mean']
        }).round(2)
        print(summary)
        
              # Verify we met requirements
        bank_counts = combined_df['bank_name'].value_counts()
        print(f"\n✅ REQUIREMENTS CHECK:")
        for bank, count in bank_counts.items():
            status = "PASS" if count >= 400 else "FAIL"
            print(f"   {bank}: {count} reviews - {status}")
        
    else:
        print("❌ No reviews were collected.")

if __name__ == "__main__":
    main()