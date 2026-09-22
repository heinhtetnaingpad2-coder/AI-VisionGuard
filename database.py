import sqlite3

connection = sqlite3.connect("detections.db")

cursor = connection.cursor()

# Detection history
cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_name TEXT,
    confidence REAL,
    detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Important events
cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT,
    object_name TEXT,
    confidence REAL,
    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

connection.commit()
connection.close()

print("Database updated successfully!")