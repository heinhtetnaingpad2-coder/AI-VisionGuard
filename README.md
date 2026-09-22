````markdown
# AI VisionGuard

AI VisionGuard is an AI-powered computer vision monitoring system that uses a laptop webcam to detect objects, generate meaningful events, store event history, and display the results through a web dashboard.

The system was designed as a laptop-only prototype, so no external hardware or sensors are required.

## Features

- Real-time webcam monitoring
- YOLO-based object detection
- Person detection
- Laptop detection
- Mobile phone detection
- Multiple-person detection
- Configurable AI confidence threshold
- Enable/disable individual detection rules
- Automatic event generation
- SQLite event database
- Event history and filtering
- Event search
- Event analytics
- Live object counters
- Web-based monitoring dashboard
- Event history cleanup

## System Architecture

```text
Laptop Webcam
      |
      v
YOLO Object Detection
      |
      v
Detection Processing
      |
      v
Event / Rule Engine
      |
      v
SQLite Database
      |
      v
Flask Backend / REST API
      |
      v
Web Dashboard
````

## Technologies Used

| Technology | Purpose                     |
| ---------- | --------------------------- |
| Python     | Main programming language   |
| YOLO       | Object detection            |
| OpenCV     | Webcam and image processing |
| Flask      | Web server and API          |
| SQLite     | Event storage               |
| HTML / CSS | Dashboard interface         |
| JavaScript | Dashboard interaction       |
| JSON       | Configuration storage       |

## How It Works

### 1. Camera Input

The laptop webcam provides a continuous video stream.

### 2. AI Detection

Each frame is processed by the YOLO object detection model.

The system identifies supported objects such as:

* person
* laptop
* cell phone

### 3. Event Processing

The system compares the current detections with previous detections.

Instead of storing every video frame as a database record, meaningful changes can generate events.

Examples:

```text
PERSON_ENTERED
PHONE_DETECTED
LAPTOP_DETECTED
MULTIPLE_PEOPLE
```

### 4. Database

Important events are stored in an SQLite database.

Each event contains:

* Event ID
* Event type
* Object name
* Detection confidence
* Event timestamp

### 5. Dashboard

The Flask web application provides:

* Live camera feed
* Current detection counts
* Event analytics
* Detection settings
* Event history
* Event filtering
* Event search

## Project Structure

```text
AI-VisionGuard
│
├── app.py
├── database.py
├── first_detection.py
├── settings.json
├── requirements.txt
├── .gitignore
├── README.md
├── yolo11n.pt
│
└── templates
    └── dashboard.html
```

## Installation

### 1. Install Python

Install Python 3.10 or newer.

### 2. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd AI-VisionGuard
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv venv
```

### 4. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Start the application

```bash
python app.py
```

### 7. Open the dashboard

Open:

```text
http://127.0.0.1:5000
```

## Configuration

Detection settings can be controlled from the dashboard.

The configuration includes:

```json
{
    "person_detection": true,
    "phone_detection": true,
    "laptop_detection": true,
    "multiple_people_detection": true,
    "confidence_threshold": 0.50
}
```

The confidence threshold controls how confident the AI model must be before a detection is accepted.

## Design Decisions

### Why SQLite?

SQLite provides a lightweight relational database without requiring a separate database server.

This makes it suitable for a laptop-based monitoring prototype.

### Why Flask?

Flask provides a simple backend for:

* serving the dashboard
* providing REST API endpoints
* connecting the AI system with the web interface

### Why Event-Based Logging?

Saving every detected video frame would create unnecessary database records.

The system therefore focuses on meaningful detection events instead of continuously storing every frame.

## Limitations

This project is currently a local computer vision prototype.

It is not intended to replace professional security or surveillance systems.

Detection accuracy depends on factors such as:

* lighting
* camera quality
* object visibility
* camera angle
* AI model limitations

The current event system is based on object-class presence and rule changes rather than persistent identity tracking.

## Future Improvements

Possible future improvements include:

* Object tracking
* More advanced event rules
* Email or notification alerts
* Historical charts
* Date and time filtering
* User authentication
* Exporting event reports
* Custom-trained detection models
* Improved dashboard visualizations

## Project Objective

The main objective of AI VisionGuard is to demonstrate how an AI detection model can be integrated into a complete software system rather than being used only for object detection.

The project combines:

```text
Computer Vision
+
AI Detection
+
Backend Development
+
Database Management
+
REST APIs
+
Web Development
```

## Author

Computer Engineering student project.

Built as a portfolio project to demonstrate practical software and AI engineering skills.

```

### One important thing

Don't put your actual name, student ID, email, or school email into the README yet.

Once you've saved it, tell me **“done”** and we'll do the next step: **create the GitHub repository and upload the project safely**.
```
