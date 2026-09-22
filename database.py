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

# CCTV events
cursor.execute("""
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT,
    object_name TEXT,
    confidence REAL,
    event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

# Person tracking sessions
cursor.execute("""
CREATE TABLE IF NOT EXISTS person_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds REAL
)
""")

connection.commit()
connection.close()

print("Database updated successfully!")