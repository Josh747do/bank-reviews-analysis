-- Database Schema for Bank Reviews Analysis
-- Task 3: PostgreSQL Database Setup

-- 1. Create database (run this first)
-- CREATE DATABASE bank_reviews;

-- 2. Banks table
CREATE TABLE IF NOT EXISTS banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) NOT NULL UNIQUE,
    app_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Reviews table
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

-- 4. Indexes for performance
CREATE INDEX IF NOT EXISTS idx_reviews_bank_id ON reviews(bank_id);
CREATE INDEX IF NOT EXISTS idx_reviews_sentiment ON reviews(sentiment_label);
CREATE INDEX IF NOT EXISTS idx_reviews_rating ON reviews(rating);
CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews(review_date);

-- 5. Sample data insertion
INSERT INTO banks (bank_name, app_name) VALUES
('CBE', 'Commercial Bank of Ethiopia Mobile'),
('BOA', 'Bank of Abyssinia Mobile'),
('DASHEN', 'Dashen Bank Mobile')
ON CONFLICT (bank_name) DO NOTHING;

-- 6. Useful queries for validation
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