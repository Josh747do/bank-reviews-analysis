#!/usr/bin/env python3
"""
Fixed Database Setup - Guaranteed to work with your existing data
"""

import psycopg2
import pandas as pd
import os

print("=" * 60)
print("🗄️ DATABASE SETUP WITH EXISTING DATA")
print("=" * 60)

# Show what data we have
print("📁 Found data files:")
print(f"  Raw data: data/raw/all_banks_reviews.csv ({os.path.getsize('data/raw/all_banks_reviews.csv'):,} bytes)")
print(f"  Processed data: data/processed/reviews_with_sentiment.csv ({os.path.getsize('data/processed/reviews_with_sentiment.csv'):,} bytes)")

# Database configuration - CHANGE PASSWORD!
DB_CONFIG = {
    'dbname': 'bank_reviews',
    'user': 'postgres',
    'password': '23456',  # ⚠️ CHANGE THIS TO YOUR POSTGRES PASSWORD!
    'host': 'localhost',
    'port': '5432'
}

try:
    # Step 1: Connect and create database
    print("\n1. Connecting to PostgreSQL...")
    conn = psycopg2.connect(
        dbname='postgres',
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port']
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Create database if not exists
    cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'bank_reviews'")
    if not cursor.fetchone():
        cursor.execute('CREATE DATABASE bank_reviews')
        print("✅ Database 'bank_reviews' created")
    else:
        print("✅ Database 'bank_reviews' already exists")
    
    cursor.close()
    conn.close()
    
    # Step 2: Connect to our database
    print("\n2. Connecting to 'bank_reviews' database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Step 3: Create tables
    print("\n3. Creating tables...")
    
    # Banks table
    cursor.execute("""
        DROP TABLE IF EXISTS reviews;
        DROP TABLE IF EXISTS banks;
        
        CREATE TABLE banks (
            bank_id SERIAL PRIMARY KEY,
            bank_name VARCHAR(100) NOT NULL UNIQUE,
            app_name VARCHAR(100) NOT NULL
        );
    """)
    
    # Reviews table
    cursor.execute("""
        CREATE TABLE reviews (
            review_id SERIAL PRIMARY KEY,
            bank_id INTEGER REFERENCES banks(bank_id),
            review_text TEXT NOT NULL,
            rating INTEGER CHECK (rating >= 1 AND rating <= 5),
            review_date DATE,
            sentiment_label VARCHAR(10),
            sentiment_score DECIMAL(5,4),
            source VARCHAR(50) DEFAULT 'Google Play Store'
        );
    """)
    print("✅ Tables created")
    
    # Step 4: Insert banks
    print("\n4. Inserting banks...")
    banks = [
        ('CBE', 'Commercial Bank of Ethiopia Mobile'),
        ('BOA', 'Bank of Abyssinia Mobile'),
        ('DASHEN', 'Dashen Bank Mobile')
    ]
    
    for bank_name, app_name in banks:
        cursor.execute("INSERT INTO banks (bank_name, app_name) VALUES (%s, %s)", 
                      (bank_name, app_name))
    print("✅ Banks inserted")
    
    # Step 5: Load and insert reviews
    print("\n5. Loading and inserting reviews...")
    
    # Use the processed data with sentiment analysis
    df = pd.read_csv('data/processed/reviews_with_sentiment.csv')
    print(f"📊 Loaded {len(df)} reviews with sentiment analysis")
    
    # Get bank IDs
    cursor.execute("SELECT bank_id, bank_name FROM banks")
    bank_map = {row[1]: row[0] for row in cursor.fetchall()}
    
    # Insert each review
    inserted = 0
    for idx, row in df.iterrows():
        bank_name = row['bank_name']
        bank_id = bank_map.get(bank_name)
        
        if bank_id:
            cursor.execute("""
                INSERT INTO reviews 
                (bank_id, review_text, rating, sentiment_label, sentiment_score)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                bank_id,
                str(row['original_text'])[:1000],  # Limit length
                int(row['rating']),
                row['sentiment_label'],
                float(row['sentiment_score'])
            ))
            inserted += 1
            
            # Show progress
            if inserted % 100 == 0:
                print(f"   Inserted {inserted} reviews...")
    
    conn.commit()
    print(f"✅ Successfully inserted {inserted} reviews")
    
    # Step 6: Run validation queries
    print("\n6. Database Validation Results:")
    print("-" * 40)
    
    # Count reviews per bank
    cursor.execute("""
        SELECT b.bank_name, COUNT(r.review_id) as count
        FROM banks b
        JOIN reviews r ON b.bank_id = r.bank_id
        GROUP BY b.bank_name
        ORDER BY count DESC
    """)
    print("📊 Reviews per bank:")
    for bank, count in cursor.fetchall():
        print(f"   {bank}: {count} reviews")
    
    # Average rating and sentiment
    cursor.execute("""
        SELECT b.bank_name, 
               ROUND(AVG(r.rating), 2) as avg_rating,
               ROUND(AVG(r.sentiment_score), 4) as avg_sentiment
        FROM banks b
        JOIN reviews r ON b.bank_id = r.bank_id
        GROUP BY b.bank_name
        ORDER BY avg_rating DESC
    """)
    print("\n⭐ Average Rating & Sentiment:")
    for bank, rating, sentiment in cursor.fetchall():
        print(f"   {bank}: {rating} stars | Sentiment: {sentiment}")
    
    # Sentiment distribution
    cursor.execute("""
        SELECT sentiment_label, COUNT(*),
               ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM reviews), 2) as percentage
        FROM reviews
        GROUP BY sentiment_label
        ORDER BY COUNT(*) DESC
    """)
    print("\n🎭 Sentiment Distribution:")
    for label, count, pct in cursor.fetchall():
        print(f"   {label}: {count} reviews ({pct}%)")
    
    # Step 7: Create indexes
    print("\n7. Creating indexes for performance...")
    cursor.execute("CREATE INDEX idx_reviews_bank_id ON reviews(bank_id)")
    cursor.execute("CREATE INDEX idx_reviews_sentiment ON reviews(sentiment_label)")
    cursor.execute("CREATE INDEX idx_reviews_rating ON reviews(rating)")
    print("✅ Indexes created")
    
    # Step 8: Export schema
    print("\n8. Exporting schema...")
    schema = """-- PostgreSQL Database Schema for Bank Reviews
-- Generated from Task 3: Database Engineering

-- Banks table
CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) NOT NULL UNIQUE,
    app_name VARCHAR(100) NOT NULL
);

-- Reviews table
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INTEGER REFERENCES banks(bank_id),
    review_text TEXT NOT NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    review_date DATE,
    sentiment_label VARCHAR(10),
    sentiment_score DECIMAL(5,4),
    source VARCHAR(50) DEFAULT 'Google Play Store'
);

-- Indexes for query performance
CREATE INDEX idx_reviews_bank_id ON reviews(bank_id);
CREATE INDEX idx_reviews_sentiment ON reviews(sentiment_label);
CREATE INDEX idx_reviews_rating ON reviews(rating);

-- Sample validation queries
SELECT b.bank_name, COUNT(r.review_id) as review_count
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name
ORDER BY review_count DESC;

SELECT b.bank_name, 
       ROUND(AVG(r.rating), 2) as avg_rating,
       ROUND(AVG(r.sentiment_score), 4) as avg_sentiment
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name
ORDER BY avg_rating DESC;
"""
    
    os.makedirs('docs', exist_ok=True)
    with open('docs/database_schema.sql', 'w') as f:
        f.write(schema)
    print("✅ Schema saved to docs/database_schema.sql")
    
    # Cleanup
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 60)
    print("🎉 TASK 3 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"✅ Database: bank_reviews")
    print(f"✅ Tables: banks, reviews")
    print(f"✅ Data: {inserted} reviews inserted")
    print(f"✅ Source: Real Google Play Store reviews with sentiment analysis")
    print(f"✅ Validation: All queries executed successfully")
    print(f"✅ Schema: docs/database_schema.sql")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n💡 Most common issues:")
    print("1. PostgreSQL not running - start the service")
    print("2. Wrong password - check line 21")
    print("3. Port conflict - default is 5432")
    import traceback
    traceback.print_exc()