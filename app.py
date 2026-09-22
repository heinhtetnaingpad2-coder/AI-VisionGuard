from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import sqlite3
import json
import time
from datetime import datetime

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

DATABASE = "detections.db"
MODEL_FILE = "yolo11n.pt"

# How long a person can temporarily disappear before
# their tracking session is closed.
SESSION_GRACE_SECONDS = 2.0

# ============================================================
# LOAD AI MODEL
# ============================================================

print()
print("================================")
print("       AI VISIONGUARD")
print("================================")
print("Loading AI model...")

model = YOLO(MODEL_FILE)

print("AI model loaded successfully.")

# ============================================================
# OPEN CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("WARNING: Camera could not be opened.")

# ============================================================
# LIVE CCTV DATA
# ============================================================

cctv_data = {
    "people": 0,
    "status": "ONLINE" if camera.isOpened() else "OFFLINE",
    "tracked_ids": []
}

# ============================================================
# EVENT TRACKING
# ============================================================

previous_person_count = 0

# ============================================================
# PERSON SESSION TRACKING
#
# Format:
#
# active_sessions = {
#     track_id: {
#         "start_timestamp": ...,
#         "last_seen": ...,
#         "confidence": ...
#     }
# }
# ============================================================

active_sessions = {}

# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():

    connection = sqlite3.connect(DATABASE)

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

    # Person sessions
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


initialize_database()

# ============================================================
# SETTINGS
# ============================================================

def load_settings():

    try:

        with open("settings.json", "r") as file:
            return json.load(file)

    except:

        return {
            "person_detection": True,
            "multiple_people_detection": True,
            "confidence_threshold": 0.50
        }


def save_settings(settings):

    with open("settings.json", "w") as file:

        json.dump(
            settings,
            file,
            indent=4
        )

# ============================================================
# SAVE CCTV EVENT
# ============================================================

def save_event(event_type, confidence):

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO events
        (event_type, object_name, confidence)
        VALUES (?, ?, ?)
        """,
        (
            event_type,
            "person",
            confidence
        )
    )

    connection.commit()
    connection.close()


# ============================================================
# CREATE CCTV EVENT
# ============================================================

def create_event(event_type, confidence):

    save_event(
        event_type,
        confidence
    )

    print()
    print("================================")
    print("AI VISIONGUARD CCTV EVENT")
    print("================================")
    print(f"Event: {event_type}")
    print("Object: person")
    print(f"Confidence: {confidence:.2f}")
    print(f"Time: {datetime.now()}")
    print()

# ============================================================
# START PERSON SESSION
# ============================================================

def start_person_session(track_id, confidence):

    current_time = time.time()

    active_sessions[track_id] = {
        "start_timestamp": current_time,
        "last_seen": current_time,
        "confidence": confidence
    }

    print(
        f"[SESSION START] Person ID {track_id}"
    )


# ============================================================
# UPDATE PERSON SESSION
# ============================================================

def update_person_session(track_id, confidence):

    current_time = time.time()

    if track_id not in active_sessions:

        start_person_session(
            track_id,
            confidence
        )

        return

    active_sessions[track_id]["last_seen"] = current_time

    active_sessions[track_id]["confidence"] = confidence


# ============================================================
# CLOSE PERSON SESSION
# ============================================================

def close_person_session(track_id):

    if track_id not in active_sessions:
        return

    session = active_sessions[track_id]

    current_time = time.time()

    start_timestamp = session[
        "start_timestamp"
    ]

    duration = current_time - start_timestamp

    # Convert timestamps into readable local time
    start_time = datetime.fromtimestamp(
        start_timestamp
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    end_time = datetime.fromtimestamp(
        current_time
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    connection = sqlite3.connect(DATABASE)

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO person_sessions
        (
            track_id,
            start_time,
            end_time,
            duration_seconds
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            track_id,
            start_time,
            end_time,
            duration
        )
    )

    connection.commit()
    connection.close()

    print(
        f"[SESSION END] Person ID {track_id} "
        f"| Duration: {duration:.1f} seconds"
    )

    del active_sessions[track_id]


# ============================================================
# CLOSE LOST PERSON SESSIONS
# ============================================================

def close_expired_sessions(current_track_ids):

    current_time = time.time()

    tracked_ids = set(current_track_ids)

    expired_ids = []

    for track_id, session in list(
        active_sessions.items()
    ):

        if track_id in tracked_ids:

            continue

        time_since_seen = (
            current_time
            - session["last_seen"]
        )

        if (
            time_since_seen
            >= SESSION_GRACE_SECONDS
        ):

            expired_ids.append(
                track_id
            )

    for track_id in expired_ids:

        close_person_session(
            track_id
        )


# ============================================================
# CAMERA FRAME GENERATOR
# ============================================================

def generate_frames():

    global previous_person_count
    global cctv_data

    while True:

        success, frame = camera.read()

        # ----------------------------------------------------
        # CAMERA OFFLINE
        # ----------------------------------------------------

        if not success:

            cctv_data = {
                "people": 0,
                "status": "OFFLINE",
                "tracked_ids": []
            }

            time.sleep(0.1)

            continue

        cctv_data["status"] = "ONLINE"

        settings = load_settings()

        confidence_threshold = float(
            settings.get(
                "confidence_threshold",
                0.50
            )
        )

        # ----------------------------------------------------
        # RUN YOLO + BYTETRACK
        # ----------------------------------------------------

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            conf=confidence_threshold,
            classes=[0]
        )

        person_count = 0
        highest_confidence = 0.0
        current_track_ids = []

        # ----------------------------------------------------
        # PERSON DETECTION
        #
        # COCO class 0 = person
        # ----------------------------------------------------

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes

            for index in range(
                len(boxes)
            ):

                confidence = float(
                    boxes.conf[index]
                )

                if confidence < confidence_threshold:
                    continue

                class_id = int(
                    boxes.cls[index]
                )

                # Extra safety:
                # Only process person class.
                if class_id != 0:
                    continue

                person_count += 1

                if confidence > highest_confidence:

                    highest_confidence = confidence

                # ------------------------------------------------
                # GET BYTE TRACK TRACK ID
                # ------------------------------------------------

                if boxes.id is not None:

                    track_id = int(
                        boxes.id[index]
                    )

                    current_track_ids.append(
                        track_id
                    )

                    update_person_session(
                        track_id,
                        confidence
                    )

        # Remove duplicate IDs
        current_track_ids = sorted(
            list(
                set(current_track_ids)
            )
        )

        # ----------------------------------------------------
        # CLOSE OLD TRACKING SESSIONS
        # ----------------------------------------------------

        close_expired_sessions(
            current_track_ids
        )

        # ----------------------------------------------------
        # UPDATE LIVE CCTV DATA
        # ----------------------------------------------------

        cctv_data = {
            "people": person_count,
            "status": "ONLINE",
            "tracked_ids": current_track_ids
        }

        # ----------------------------------------------------
        # PERSON ENTERED
        # ----------------------------------------------------

        if (
            person_count > 0
            and previous_person_count == 0
            and settings.get(
                "person_detection",
                True
            )
        ):

            create_event(
                "PERSON_DETECTED",
                highest_confidence
            )

        # ----------------------------------------------------
        # MULTIPLE PEOPLE
        # ----------------------------------------------------

        if (
            person_count >= 2
            and previous_person_count < 2
            and settings.get(
                "multiple_people_detection",
                True
            )
        ):

            create_event(
                "MULTIPLE_PEOPLE",
                highest_confidence
            )

        # ----------------------------------------------------
        # PERSON LEFT
        # ----------------------------------------------------

        if (
            person_count == 0
            and previous_person_count > 0
        ):

            create_event(
                "PERSON_LEFT",
                0.0
            )

        previous_person_count = person_count

        # ----------------------------------------------------
        # DRAW DETECTIONS + TRACKING IDS
        # ----------------------------------------------------

        output = results[0].plot()

        cv2.putText(
            output,
            f"PEOPLE DETECTED: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            "AI CCTV - PERSON TRACKING",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if current_track_ids:

            ids_text = (
                "TRACK IDs: "
                + ", ".join(
                    str(track_id)
                    for track_id
                    in current_track_ids
                )
            )

            cv2.putText(
                output,
                ids_text,
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # ----------------------------------------------------
        # ENCODE FRAME
        # ----------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            output
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# LIVE CCTV DATA
# ============================================================

@app.route("/api/counts")
def counts():

    return jsonify(
        cctv_data
    )


# ============================================================
# EVENTS API
# ============================================================

@app.route("/api/events")
def events():

    event_type = request.args.get(
        "event_type",
        ""
    )

    search = request.args.get(
        "search",
        ""
    )

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    conditions = [
        "object_name = 'person'"
    ]

    parameters = []

    # Event filter
    if event_type:

        conditions.append(
            "event_type = ?"
        )

        parameters.append(
            event_type
        )

    # Search
    if search:

        conditions.append(
            """
            (
                event_type LIKE ?
                OR object_name LIKE ?
            )
            """
        )

        search_value = (
            "%"
            + search
            + "%"
        )

        parameters.append(
            search_value
        )

        parameters.append(
            search_value
        )

    query = """
        SELECT *
        FROM events
        WHERE
    """

    query += " AND ".join(
        conditions
    )

    query += """
        ORDER BY id DESC
        LIMIT 50
    """

    cursor.execute(
        query,
        parameters
    )

    event_rows = cursor.fetchall()

    connection.close()

    return jsonify(
        [
            dict(event)
            for event in event_rows
        ]
    )


# ============================================================
# CLEAR EVENT HISTORY
# ============================================================

@app.route(
    "/api/events/clear",
    methods=["DELETE"]
)
def clear_events():

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM events
        WHERE object_name = 'person'
        """
    )

    connection.commit()

    deleted_count = cursor.rowcount

    connection.close()

    print(
        f"Cleared {deleted_count} CCTV events."
    )

    return jsonify({
        "success": True,
        "deleted": deleted_count
    })


# ============================================================
# PERSON SESSIONS API
# ============================================================

@app.route("/api/sessions")
def sessions():

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            track_id,
            start_time,
            end_time,
            duration_seconds
        FROM person_sessions
        ORDER BY id DESC
        LIMIT 50
        """
    )

    session_rows = cursor.fetchall()

    connection.close()

    session_data = [
        dict(row)
        for row in session_rows
    ]

    # Add currently active sessions
    current_time = time.time()

    for track_id, session in active_sessions.items():

        duration = (
            current_time
            - session["start_timestamp"]
        )

        start_time = datetime.fromtimestamp(
            session["start_timestamp"]
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        session_data.insert(
            0,
            {
                "id": f"active-{track_id}",
                "track_id": track_id,
                "start_time": start_time,
                "end_time": None,
                "duration_seconds": duration,
                "active": True
            }
        )

    return jsonify(
        session_data[:50]
    )


# ============================================================
# CLEAR SESSION HISTORY
# ============================================================

@app.route(
    "/api/sessions/clear",
    methods=["DELETE"]
)
def clear_sessions():

    connection = sqlite3.connect(
        DATABASE
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM person_sessions
        """
    )

    connection.commit()

    deleted_count = cursor.rowcount

    connection.close()

    return jsonify({
        "success": True,
        "deleted": deleted_count
    })


# ============================================================
# SETTINGS API
# ============================================================

@app.route("/api/settings")
def get_settings():

    return jsonify(
        load_settings()
    )


@app.route(
    "/api/settings",
    methods=["POST"]
)
def update_settings():

    new_settings = request.get_json()

    if not isinstance(
        new_settings,
        dict
    ):

        return jsonify({
            "success": False,
            "error": "Invalid settings."
        }), 400

    current_settings = load_settings()

    for key in current_settings:

        if key in new_settings:

            current_settings[key] = (
                new_settings[key]
            )

    # Keep confidence within a safe range
    try:

        threshold = float(
            current_settings.get(
                "confidence_threshold",
                0.50
            )
        )

        threshold = max(
            0.05,
            min(
                threshold,
                0.95
            )
        )

        current_settings[
            "confidence_threshold"
        ] = threshold

    except:

        current_settings[
            "confidence_threshold"
        ] = 0.50

    save_settings(
        current_settings
    )

    return jsonify({
        "success": True,
        "settings": current_settings
    })


# ============================================================
# ANALYTICS API
# ============================================================

@app.route("/api/analytics")
def analytics():

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    # --------------------------------------------------------
    # TOTAL EVENTS
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM events
        WHERE object_name = 'person'
        """
    )

    total_events = (
        cursor.fetchone()["total"]
    )

    # --------------------------------------------------------
    # PERSON DETECTED EVENTS
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM events
        WHERE
            event_type = 'PERSON_DETECTED'
            AND object_name = 'person'
        """
    )

    person_events = (
        cursor.fetchone()["total"]
    )

    # --------------------------------------------------------
    # MULTIPLE PEOPLE EVENTS
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM events
        WHERE
            event_type = 'MULTIPLE_PEOPLE'
            AND object_name = 'person'
        """
    )

    multiple_people_events = (
        cursor.fetchone()["total"]
    )

    # --------------------------------------------------------
    # PERSON LEFT EVENTS
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM events
        WHERE
            event_type = 'PERSON_LEFT'
            AND object_name = 'person'
        """
    )

    person_left_events = (
        cursor.fetchone()["total"]
    )

    # --------------------------------------------------------
    # TOTAL COMPLETED SESSIONS
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM person_sessions
        """
    )

    completed_sessions = (
        cursor.fetchone()["total"]
    )

    # --------------------------------------------------------
    # AVERAGE PRESENCE DURATION
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT AVG(duration_seconds) AS average
        FROM person_sessions
        WHERE duration_seconds IS NOT NULL
        """
    )

    average_result = cursor.fetchone()

    average_duration = (
        average_result["average"]
        if average_result["average"] is not None
        else 0
    )

    # --------------------------------------------------------
    # EVENT BREAKDOWN
    # --------------------------------------------------------

    cursor.execute(
        """
        SELECT
            event_type,
            COUNT(*) AS count
        FROM events
        WHERE object_name = 'person'
        GROUP BY event_type
        ORDER BY count DESC
        """
    )

    breakdown = cursor.fetchall()

    connection.close()

    # Include active sessions in session count
    total_sessions = (
        completed_sessions
        + len(active_sessions)
    )

    return jsonify({

        "total_events":
            total_events,

        "person_events":
            person_events,

        "multiple_people_events":
            multiple_people_events,

        "person_left_events":
            person_left_events,

        "total_sessions":
            total_sessions,

        "completed_sessions":
            completed_sessions,

        "active_sessions":
            len(active_sessions),

        "average_duration":
            average_duration,

        "breakdown": [
            dict(row)
            for row in breakdown
        ]

    })


# ============================================================
# SHUTDOWN
# ============================================================

def shutdown_camera():

    if camera.isOpened():

        camera.release()

    print()
    print("Camera released.")


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("================================")
    print("       AI VISIONGUARD")
    print("================================")
    print("       AI CCTV SYSTEM")
    print("================================")
    print()
    print("Person detection: ENABLED")
    print("Person tracking: ENABLED")
    print("Multiple people: ENABLED")
    print("Tracking system: ByteTrack")
    print()
    print("Open:")
    print("http://127.0.0.1:5000")
    print()

    # debug=False prevents Flask's automatic reloader
    # from opening the webcam twice.
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )