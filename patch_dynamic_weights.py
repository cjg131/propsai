import sqlite3

def create_weights_table():
    conn = sqlite3.connect('/Users/cj/Dropbox/Windsurf/CascadeProjects/Sports Props Betting/backend/app/data/trading_engine.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS weather_api_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_name TEXT NOT NULL,
            city TEXT NOT NULL,
            target_date DATE NOT NULL,
            predicted_temp REAL,
            actual_temp REAL,
            error REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("Table created.")

create_weights_table()
