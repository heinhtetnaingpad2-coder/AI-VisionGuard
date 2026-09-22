from ultralytics import YOLO
import cv2
import sqlite3
import os
from datetime import datetime


# ==========================================
# 1. LOAD AI MODEL
# ==========================================

print("Loading AI model...")

model = YOLO("yolo11n.pt")

print("AI model loaded.")


# ==========================================
# 2. CONNECT TO DATABASE
# ==========================================

connection = sqlite3.connect("detections.db")

cursor = connection.cursor()


# ==========================================
# 3. CREATE SCREENSHOT FOLDER
# ==========================================

os.makedirs("screenshots", exist_ok=True)


# ==========================================
# 4. OPEN LAPTOP CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Could not open camera.")
    exit()


# ==========================================
# 5. EVENT TRACKING
# ==========================================

previous_objects = set()


print()
print("================================")
print("       AI VISIONGUARD")
print("================================")
print("Camera started.")
print("Press Q to stop.")
print()


# ==========================================
# 6. MAIN PROGRAM
# ==========================================

while True:

    success, frame = camera.read()

    if not success:

        print("Could not read camera.")
        break


    # --------------------------------------
    # Run YOLO
    # --------------------------------------

    results = model(frame, verbose=False)


    # --------------------------------------
    # Object counters
    # --------------------------------------

    person_count = 0
    laptop_count = 0
    phone_count = 0


    # Objects detected in this frame
    current_objects = set()


    # --------------------------------------
    # Process AI detections
    # --------------------------------------

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(box.conf[0])

            object_name = model.names[class_id]


            # Add object to current frame
            current_objects.add(object_name)


            # Count objects

            if object_name == "person":

                person_count += 1

            elif object_name == "laptop":

                laptop_count += 1

            elif object_name == "cell phone":

                phone_count += 1


            # ----------------------------------
            # Save detection
            # ----------------------------------

            cursor.execute(
                """
                INSERT INTO detections
                (object_name, confidence)
                VALUES (?, ?)
                """,
                (object_name, confidence)
            )


    # --------------------------------------
    # Find NEW objects
    # --------------------------------------

    new_objects = current_objects - previous_objects


    for object_name in new_objects:

        # Current time
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )


        # Screenshot filename
        filename = (
            f"screenshots/"
            f"{object_name}_{timestamp}.jpg"
        )


        # Save screenshot
        cv2.imwrite(filename, frame)


        # Print event
        print(
            f"NEW EVENT: {object_name}"
        )

        print(
            f"Screenshot saved: {filename}"
        )


        # ----------------------------------
        # Save event to database
        # ----------------------------------

        cursor.execute(
            """
            INSERT INTO events
            (event_type, object_name, confidence)
            VALUES (?, ?, ?)
            """,
            (
                "Object Detected",
                object_name,
                0.0
            )
        )


    # --------------------------------------
    # Save database
    # --------------------------------------

    connection.commit()


    # Remember current objects
    previous_objects = current_objects


    # --------------------------------------
    # Draw detection boxes
    # --------------------------------------

    output = results[0].plot()


    # --------------------------------------
    # Display counts
    # --------------------------------------

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


    # --------------------------------------
    # Show camera
    # --------------------------------------

    cv2.imshow(
        "AI VisionGuard",
        output
    )


    # --------------------------------------
    # Press Q to quit
    # --------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# 7. CLEAN UP
# ==========================================

camera.release()

connection.close()

cv2.destroyAllWindows()


print()
print("AI VisionGuard stopped.")