import sqlite3
import os

def setup_database():
    """Initialize the predictions database"""
    os.makedirs('logs', exist_ok=True)
    
    conn = sqlite3.connect('logs/predictions.db')
    c = conn.cursor()
    
    # Create predictions table
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  features TEXT,
                  prediction INTEGER,
                  confidence REAL)''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

if __name__ == "__main__":
    setup_database()