# Database Configuration
DB_CONFIG = {
    'dbname': 'bank_reviews',
    'user': 'postgres',  # Change if different
    'password': 'password',  # Change to your PostgreSQL password
    'host': 'localhost',
    'port': '5432'
}

# App IDs for Ethiopian Banks (we'll verify these later)
BANK_APPS = {
    'cbe': 'com.cbe.mobilebanking',
    'boa': 'com.bankofabyssinia.mobilebanking', 
    'dashen': 'com.dashenbank.scmobile'
}

  
  

# Analysis Settings
SENTIMENT_THRESHOLDS = {
    'positive': 0.6,
    'neutral': 0.4,
    'negative': 0.0
}

# Theme Categories for Analysis
THEME_CATEGORIES = {
    'login_issues': ['login', 'password', 'authentication', 'biometric', 'sign in'],
    'transaction_problems': ['transfer', 'payment', 'transaction', 'failed', 'money'],
    'performance_issues': ['slow', 'crash', 'freeze', 'loading', 'lag'],
    'ui_ux_feedback': ['interface', 'design', 'navigation', 'easy', 'beautiful'],
    'feature_requests': ['should have', 'please add', 'would like', 'missing', 'wish'],
    'customer_support': ['support', 'help', 'response', 'service', 'contact']
}