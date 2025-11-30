#!/usr/bin/env python3
"""
Test script to scrape only CBE reviews first
"""

from google_play_scraper import reviews_all, Sort
import pandas as pd

def test_cbe():
    print("🧪 Testing CBE app scraping...")
    
    try:
        # Test only CBE first
        reviews = reviews_all(
            'com.combanketh.mobilebanking',  # CBE app ID
            lang='en',
            country='et', 
            sort=Sort.NEWEST,
            count=50,  # Start small
            sleep_milliseconds=2000
        )
        
        print(f"✅ SUCCESS: Found {len(reviews)} CBE reviews!")
        
        if reviews:
            # Show first review as sample
            first_review = reviews[0]
            print(f"\n📝 Sample Review:")
            print(f"   Text: {first_review['content'][:100]}...")
            print(f"   Rating: {first_review['score']}/5")
            print(f"   Date: {first_review['at']}")
            print(f"   Likes: {first_review.get('thumbsUpCount', 0)}")
            
            # Save the test data
            df = pd.DataFrame(reviews)
            df.to_csv('data/raw/test_cbe_reviews.csv', index=False)
            print(f"💾 Saved {len(reviews)} reviews to data/raw/test_cbe_reviews.csv")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    test_cbe()