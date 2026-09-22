
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import cv2
import sqlite3
import json

app = Flask(__name__)

# ==========================================
# LOAD AI MODEL
# ==========================================

print("Loading AI model...")
model = YOLO("yolo11n.pt")
print("AI model loaded.")

# ==========================================
# OPEN CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

# ==========================================
# LIVE COUNTS
# ==========================================

object_counts = {
    "person": 0,
    "laptop": 0,
    "cell phone": 0
}

# ==========================================
# EVENT TRACKING
# ==========================================

previous_objects = set()
previous_person_count = 0

# ==========================================
# SETTINGS
# ==========================================

def load_settings():

    try:
        with open("settings.json", "r") as file:
            return json.load(file)

    except:

        return {
            "person_detection": True,
            "phone_detection": True,
            "laptop_detection": True,
            "multiple_people_detection": True,
            "confidence_threshold": 0.50
        }


def save_settings(settings):

    with open("settings.json", "w") as file:
        json.dump(settings, file, indent=4)

# ==========================================
# DATABASE
# ==========================================

def save_event(event_type, object_name, confidence):

    connection = sqlite3.connect("detections.db")
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO events
        (event_type, object_name, confidence)
        VALUES (?, ?, ?)
        """,
        (
            event_type,
            object_name,
            confidence
        )
    )

    connection.commit()
    connection.close()


# ==========================================
# CREATE EVENT
# ==========================================

def create_event(event_type, object_name, confidence):

    save_event(
        event_type,
        object_name,
        confidence
    )

    print()
    print("================================")
    print("NEW VISIONGUARD EVENT")
    print("================================")
    print(f"Event: {event_type}")
    print(f"Object: {object_name}")
    print(f"Confidence: {confidence:.2f}")
    print()


# ==========================================
# CAMERA FRAME GENERATOR
# ==========================================

def generate_frames():

    global object_counts
    global previous_objects
    global previous_person_count

    while True:

        success, frame = camera.read()

        if not success:
            break

        settings = load_settings()

        confidence_threshold = settings[
            "confidence_threshold"
        ]

        # ==================================
        # RUN YOLO
        # ==================================

        results = model(
            frame,
            verbose=False
        )

        person_count = 0
        laptop_count = 0
        phone_count = 0

        current_objects = set()
        confidence_values = {}

        # ==================================
        # PROCESS DETECTIONS
        # ==================================

        for result in results:

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                if confidence < confidence_threshold:
                    continue

                class_id = int(
                    box.cls[0]
                )

                object_name = model.names[
                    class_id
                ]

                current_objects.add(
                    object_name
                )

                if (
                    object_name not in confidence_values
                    or confidence >
                    confidence_values[object_name]
                ):

                    confidence_values[
                        object_name
                    ] = confidence

                if object_name == "person":
                    person_count += 1

                elif object_name == "laptop":
                    laptop_count += 1

                elif object_name == "cell phone":
                    phone_count += 1

        # ==================================
        # UPDATE LIVE COUNTS
        # ==================================

        object_counts = {
            "person": person_count,
            "laptop": laptop_count,
            "cell phone": phone_count
        }

        # ==================================
        # FIND NEW OBJECTS
        # ==================================

        new_objects = (
            current_objects
            - previous_objects
        )

        # ==================================
        # PERSON EVENT
        # ==================================

        if (
            "person" in new_objects
            and settings["person_detection"]
        ):

            create_event(
                "PERSON_ENTERED",
                "person",
                confidence_values.get(
                    "person",
                    0.0
                )
            )

        # ==================================
        # PHONE EVENT
        # ==================================

        if (
            "cell phone" in new_objects
            and settings["phone_detection"]
        ):

            create_event(
                "PHONE_DETECTED",
                "cell phone",
                confidence_values.get(
                    "cell phone",
                    0.0
                )
            )

        # ==================================
        # LAPTOP EVENT
        # ==================================

        if (
            "laptop" in new_objects
            and settings["laptop_detection"]
        ):

            create_event(
                "LAPTOP_DETECTED",
                "laptop",
                confidence_values.get(
                    "laptop",
                    0.0
                )
            )

        # ==================================
        # MULTIPLE PEOPLE
        # ==========================================

        if (
            person_count >= 2
            and previous_person_count < 2
            and settings["multiple_people_detection"]
        ):

            create_event(
                "MULTIPLE_PEOPLE",
                "person",
                0.0
            )

        previous_objects = current_objects
        previous_person_count = person_count

        # ==================================
        # DRAW DETECTIONS
        # ==================================

        output = results[0].plot()

        cv2.putText(
            output,
            f"People: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Laptops: {laptop_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Phones: {phone_count}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # ==================================
        # ENCODE FRAME
        # ==================================

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


# ==========================================
# DASHBOARD
# ==========================================

@app.route("/")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# ==========================================
# VIDEO FEED
# ==========================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ==========================================
# LIVE COUNTS
# ==========================================

@app.route("/api/counts")
def counts():

    return jsonify(
        object_counts
    )


# ==========================================
# EVENTS API
# ==========================================

@app.route("/api/events")
def events():

    event_type = request.args.get(
        "event_type",
        ""
    )

    object_name = request.args.get(
        "object_name",
        ""
    )

    search = request.args.get(
        "search",
        ""
    )

    connection = sqlite3.connect(
        "detections.db"
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    conditions = []
    parameters = []

    if event_type:

        conditions.append(
            "event_type = ?"
        )

        parameters.append(
            event_type
        )

    if object_name:

        conditions.append(
            "object_name = ?"
        )

        parameters.append(
            object_name
        )

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
    """

    if conditions:

        query += (
            " WHERE "
            + " AND ".join(
                conditions
            )
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


# ==========================================
# CLEAR EVENT HISTORY
# ==========================================

@app.route(
    "/api/events/clear",
    methods=["DELETE"]
)
def clear_events():

    connection = sqlite3.connect(
        "detections.db"
    )

    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM events"
    )

    connection.commit()

    deleted_count = cursor.rowcount

    connection.close()

    print(
        f"Cleared {deleted_count} events."
    )

    return jsonify({
        "success": True,
        "deleted": deleted_count
    })


# ==========================================
# SETTINGS
# ==========================================

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

    current_settings = load_settings()

    for key in current_settings:

        if key in new_settings:

            current_settings[key] = (
                new_settings[key]
            )

    save_settings(
        current_settings
    )

    return jsonify({
        "success": True,
        "settings": current_settings
    })


# ==========================================
# ANALYTICS
# ==========================================

@app.route("/api/analytics")
def analytics():

    connection = sqlite3.connect(
        "detections.db"
    )

    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM events
        """
    )

    total_events = (
        cursor.fetchone()["total"]
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM events
        WHERE event_type = 'PERSON_ENTERED'
        """
    )

    people_events = (
        cursor.fetchone()["total"]
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM events
        WHERE event_type = 'PHONE_DETECTED'
        """
    )

    phone_events = (
        cursor.fetchone()["total"]
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM events
        WHERE event_type = 'LAPTOP_DETECTED'
        """
    )

    laptop_events = (
        cursor.fetchone()["total"]
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM events
        WHERE event_type = 'MULTIPLE_PEOPLE'
        """
    )

    multiple_people_events = (
        cursor.fetchone()["total"]
    )

    cursor.execute(
        """
        SELECT
            event_type,
            COUNT(*) AS count
        FROM events
        GROUP BY event_type
        ORDER BY count DESC
        """
    )

    breakdown = cursor.fetchall()

    connection.close()

    return jsonify({

        "total_events":
            total_events,

        "people_events":
            people_events,

        "phone_events":
            phone_events,

        "laptop_events":
            laptop_events,

        "multiple_people_events":
            multiple_people_events,

        "breakdown": [
            dict(row)
            for row in breakdown
        ]

    })


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    print()
    print("================================")
    print("       AI VISIONGUARD")
    print("================================")
    print("Dashboard starting...")
    print()
    print("Open:")
    print("http://127.0.0.1:5000")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
