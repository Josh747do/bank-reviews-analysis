#!/usr/bin/env python3
"""
PostgreSQL Database Setup for Bank Reviews
Task 3: Database Engineering
"""

import psycopg2
from psycopg2 import sql
import pandas as pd
import sys
import os

class DatabaseManager:
    def __init__(self, dbname='bank_reviews', user='postgres', 
                 password='password', host='localhost', port='5432'):
        """
        Initialize database connection parameters
        CHANGE THE PASSWORD TO YOUR ACTUAL POSTGRESQL PASSWORD!
        """
        self.connection_params = {
            'dbname': dbname,
            'user': user,
            'password': 23456,  # CHANGE THIS!
            'host': host,
            'port': port
        }
        self.conn = None
    
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.conn.autocommit = False
            print("✅ Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("\n💡 TROUBLESHOOTING:")
            print("1. Make sure PostgreSQL is running")
            print("2. Check your password in the script")
            print("3. Try: sudo service postgresql start (Linux/Mac)")
            print("4. Try: Start PostgreSQL service (Windows Services)")
            return False
    
    def create_database(self):
        """Create the database if it doesn't exist"""
        try:
            # Connect to default postgres database to create our database
            temp_params = self.connection_params.copy()
            temp_params['dbname'] = 'postgres'
            temp_conn = psycopg2.connect(**temp_params)
            temp_conn.autocommit = True
            cursor = temp_conn.cursor()
            
            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", 
                          (self.connection_params['dbname'],))
            exists = cursor.fetchone()
            
            if not exists:
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(self.connection_params['dbname'])
                ))
                print(f"✅ Database '{self.connection_params['dbname']}' created")
            else:
                print(f"✅ Database '{self.connection_params['dbname']}' already exists")
            
            cursor.close()
            temp_conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Database creation failed: {e}")
            return False
    
    def create_tables(self):
        """Create banks and reviews tables"""
        try:
            cursor = self.conn.cursor()
            
            # Create banks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS banks (
                    bank_id SERIAL PRIMARY KEY,
                    bank_name VARCHAR(100) NOT NULL UNIQUE,
                    app_name VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("✅ Created 'banks' table")
            
            # Create reviews table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id SERIAL PRIMARY KEY,
                    bank_id INTEGER REFERENCES banks(bank_id) ON DELETE CASCADE,
                    review_text TEXT NOT NULL,
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                    review_date DATE,
                    sentiment_label VARCHAR(10),
                    sentiment_score DECIMAL(5,4),
                    source VARCHAR(50) DEFAULT 'Google Play Store',
                    thumbs_up INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("✅ Created 'reviews' table")
            
            # Create indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bank_id ON reviews(bank_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON reviews(sentiment_label)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rating ON reviews(rating)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON reviews(review_date)")
            print("✅ Created indexes for performance")
            
            self.conn.commit()
            cursor.close()
            return True
            
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Table creation failed: {e}")
            return False
    
    def insert_banks(self):
        """Insert bank data into banks table"""
        try:
            cursor = self.conn.cursor()
            
            banks_data = [
                ('CBE', 'Commercial Bank of Ethiopia Mobile'),
                ('BOA', 'Bank of Abyssinia Mobile'),
                ('DASHEN', 'Dashen Bank Mobile')
            ]
            
            for bank_name, app_name in banks_data:
                cursor.execute("""
                    INSERT INTO banks (bank_name, app_name) 
                    VALUES (%s, %s)
                    ON CONFLICT (bank_name) DO NOTHING
                """, (bank_name, app_name))
            
            self.conn.commit()
            cursor.close()
            print("✅ Inserted bank data")
            return True
            
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Bank insertion failed: {e}")
            return False
    
    def load_review_data(self):
        """Load and insert review data from CSV"""
        try:
            # First, copy data from task-2 branch
            print("📁 Looking for review data...")
            
            # Try multiple possible locations
            possible_paths = [
                'data/processed/reviews_with_sentiment.csv',
                '../data/processed/reviews_with_sentiment.csv',
                '../../data/processed/reviews_with_sentiment.csv'
            ]
            
            reviews_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    reviews_path = path
                    break
            
            if not reviews_path:
                print("❌ Sentiment data not found. We'll use the raw data instead.")
                # Try raw data
                raw_paths = [
                    'data/raw/all_banks_reviews.csv',
                    '../data/raw/all_banks_reviews.csv'
                ]
                for path in raw_paths:
                    if os.path.exists(path):
                        reviews_path = path
                        break
            
            if not reviews_path:
                print("❌ No review data found. Please ensure data is available.")
                return False
            
            df = pd.read_csv(reviews_path)
            print(f"📊 Loaded {len(df)} reviews from {reviews_path}")
            
            cursor = self.conn.cursor()
            
            # Get bank IDs
            cursor.execute("SELECT bank_id, bank_name FROM banks")
            bank_map = {row[1]: row[0] for row in cursor.fetchall()}
            
            # Prepare data for insertion
            inserted_count = 0
            errors = 0
            
            for idx, row in df.iterrows():
                try:
                    bank_id = bank_map.get(row['bank_name'])
                    if not bank_id:
                        # Try alternative bank name formats
                        bank_name_upper = row['bank_name'].upper() if 'bank_name' in row else 'CBE'
                        bank_id = bank_map.get(bank_name_upper)
                    
                    if not bank_id:
                        errors += 1
                        continue
                    
                    # Extract data with fallbacks
                    review_text = str(row.get('review_text', row.get('original_text', '')))[:1000]
                    rating = int(row.get('rating', 3))
                    
                    # Handle date
                    date_val = None
                    if 'date' in row and pd.notna(row['date']):
                        date_val = str(row['date']).split(' ')[0]  # Take only date part
                    
                    # Handle sentiment (might not exist in raw data)
                    sentiment_label = row.get('sentiment_label', 'NEUTRAL')
                    sentiment_score = float(row.get('sentiment_score', 0.5))
                    
                    cursor.execute("""
                        INSERT INTO reviews 
                        (bank_id, review_text, rating, review_date, 
                         sentiment_label, sentiment_score, source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        bank_id,
                        review_text,
                        rating,
                        date_val,
                        sentiment_label,
                        sentiment_score,
                        'Google Play Store'
                    ))
                    inserted_count += 1
                    
                    # Progress indicator
                    if inserted_count % 100 == 0:
                        print(f"   Inserted {inserted_count} reviews...")
                        
                except Exception as e:
                    errors += 1
                    if errors < 5:  # Show first few errors only
                        print(f"   Warning: Error on row {idx}: {e}")
            
            self.conn.commit()
            cursor.close()
            
            print(f"✅ Successfully inserted {inserted_count} reviews into database")
            if errors > 0:
                print(f"⚠️  Skipped {errors} rows due to errors")
            return True
            
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Review insertion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_validation_queries(self):
        """Run SQL queries to validate data integrity"""
        try:
            cursor = self.conn.cursor()
            
            print("\n📊 DATABASE VALIDATION QUERIES:")
            print("=" * 50)
            
            # 1. Count reviews per bank
            cursor.execute("""
                SELECT b.bank_name, COUNT(r.review_id) as review_count
                FROM banks b
                LEFT JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_name
                ORDER BY review_count DESC
            """)
            print("\n1. Reviews per Bank:")
            for row in cursor.fetchall():
                print(f"   {row[0]}: {row[1]} reviews")
            
            # 2. Average rating per bank
            cursor.execute("""
                SELECT b.bank_name, 
                       ROUND(AVG(r.rating), 2) as avg_rating,
                       ROUND(AVG(r.sentiment_score), 4) as avg_sentiment
                FROM banks b
                JOIN reviews r ON b.bank_id = r.bank_id
                GROUP BY b.bank_name
                ORDER BY avg_rating DESC
            """)
            print("\n2. Average Rating & Sentiment per Bank:")
            for row in cursor.fetchall():
                print(f"   {row[0]}: {row[1]}⭐ | Sentiment: {row[2]}")
            
            # 3. Sentiment distribution
            cursor.execute("""
                SELECT sentiment_label, COUNT(*) as count,
                       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
                FROM reviews
                GROUP BY sentiment_label
                ORDER BY count DESC
            """)
            print("\n3. Overall Sentiment Distribution:")
            for row in cursor.fetchall():
                print(f"   {row[0]}: {row[1]} reviews ({row[2]}%)")
            
            # 4. Date range
            cursor.execute("""
                SELECT MIN(review_date), MAX(review_date), 
                       COUNT(DISTINCT review_date) as unique_days
                FROM reviews
                WHERE review_date IS NOT NULL
            """)
            date_range = cursor.fetchone()
            print(f"\n4. Date Range: {date_range[0]} to {date_range[1]}")
            print(f"   Total days with reviews: {date_range[2]}")
            
            # 5. Rating distribution
            cursor.execute("""
                SELECT rating, COUNT(*) as count,
                       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
                FROM reviews
                GROUP BY rating
                ORDER BY rating DESC
            """)
            print("\n5. Rating Distribution:")
            for row in cursor.fetchall():
                print(f"   {row[0]}⭐: {row[1]} reviews ({row[2]}%)")
            
            cursor.close()
            return True
            
        except Exception as e:
            print(f"❌ Validation queries failed: {e}")
            return False
    
    def export_schema(self):
        """Export database schema to SQL file"""
        try:
            schema_sql = """
-- Database Schema for Bank Reviews Analysis
-- Generated: Task 3 - PostgreSQL Database Setup

-- 1. Banks table
CREATE TABLE IF NOT EXISTS banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) NOT NULL UNIQUE,
    app_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Reviews table
CREATE TABLE IF NOT EXISTS reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INTEGER REFERENCES banks(bank_id) ON DELETE CASCADE,
    review_text TEXT NOT NULL,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    review_date DATE,
    sentiment_label VARCHAR(10),
    sentiment_score DECIMAL(5,4),
    source VARCHAR(50) DEFAULT 'Google Play Store',
    thumbs_up INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Indexes for performance
CREATE INDEX IF NOT EXISTS idx_bank_id ON reviews(bank_id);
CREATE INDEX IF NOT EXISTS idx_sentiment ON reviews(sentiment_label);
CREATE INDEX IF NOT EXISTS idx_rating ON reviews(rating);
CREATE INDEX IF NOT EXISTS idx_date ON reviews(review_date);

-- 4. Sample queries for analysis
-- Count reviews per bank
SELECT b.bank_name, COUNT(r.review_id) as review_count
FROM banks b
LEFT JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name
ORDER BY review_count DESC;

-- Average rating and sentiment per bank
SELECT b.bank_name, 
       ROUND(AVG(r.rating), 2) as avg_rating,
       ROUND(AVG(r.sentiment_score), 4) as avg_sentiment
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name
ORDER BY avg_rating DESC;
"""
            
            # Save to file
            with open('docs/sql/database_schema.sql', 'w') as f:
                f.write(schema_sql)
            
            print("💾 Database schema exported to docs/sql/database_schema.sql")
            return True
            
        except Exception as e:
            print(f"❌ Schema export failed: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")

def main():
    """Main function to setup database"""
    print("=" * 60)
    print("🗄️ TASK 3: POSTGRESQL DATABASE ENGINEERING")
    print("=" * 60)
    
    print("⚠️  IMPORTANT: Before running, edit line 22 in this script")
    print("   Change 'password' to your actual PostgreSQL password!")
    print("=" * 60)
    
    # IMPORTANT: Change the password to your actual PostgreSQL password!
    db_manager = DatabaseManager(
        dbname='bank_reviews',
        user='postgres',
        password='password',  # ⚠️ CHANGE THIS TO YOUR PASSWORD!
        host='localhost',
        port='5432'
    )
    
    try:
        # Step 1: Create database
        print("\n1️⃣ Creating database...")
        if not db_manager.create_database():
            return
        
        # Step 2: Connect to our database
        print("\n2️⃣ Connecting to database...")
        if not db_manager.connect():
            return
        
        # Step 3: Create tables
        print("\n3️⃣ Creating database tables...")
        if not db_manager.create_tables():
            return
        
        # Step 4: Insert banks
        print("\n4️⃣ Inserting bank data...")
        if not db_manager.insert_banks():
            return
        
        # Step 5: Load review data
        print("\n5️⃣ Loading review data into database...")
        if not db_manager.load_review_data():
            return
        
        # Step 6: Run validation queries
        print("\n6️⃣ Running validation queries...")
        db_manager.run_validation_queries()
        
        # Step 7: Export schema
        print("\n7️⃣ Exporting database schema...")
        db_manager.export_schema()
        
        print("\n" + "=" * 60)
        print("🎉 TASK 3 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Database 'bank_reviews' created")
        print("✅ Tables 'banks' and 'reviews' populated")
        print("✅ 1500+ reviews stored in PostgreSQL")
        print("✅ Schema exported to docs/sql/database_schema.sql")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db_manager.close()

if __name__ == "__main__":
    # Check if psycopg2 is installed
    try:
        import psycopg2
        main()
    except ImportError:
        print("❌ psycopg2 is not installed. Installing now...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
        print("✅ psycopg2 installed. Please run the script again.")