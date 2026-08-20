# Hand Gesture Drawing Application

A real-time computer vision drawing application that allows users to draw, erase, select colors, and control the brush using hand gestures.

The application uses a webcam to track hand movements with MediaPipe and processes the video stream using OpenCV.

## Features

- Real-time hand tracking
- Index-finger drawing
- Two-finger gesture eraser
- Gesture-controlled color selection
- Green, red, blue, and yellow drawing colors
- Adjustable brush size
- Canvas clearing
- Stroke-based undo and redo
- Save drawings as PNG images
- Smoothed hand movement for stable drawing
- Automated unit tests
- Clean GitHub-ready project structure

## Tech Stack

- Python
- OpenCV
- MediaPipe
- NumPy
- Pytest

## How It Works

The application captures video from the webcam and uses MediaPipe hand landmarks to track the user's hand.

The index fingertip acts as the drawing cursor.

### Controls

| Gesture / Key | Action |
|---|---|
| Index finger | Draw |
| Index + middle fingers | Erase |
| Move index finger to a color | Select color |
| `1` | Green |
| `2` | Red |
| `3` | Blue |
| `4` | Yellow |
| `+` / `-` | Increase / decrease brush size |
| `Z` | Undo |
| `Y` | Redo |
| `S` | Save drawing |
| `C` | Clear canvas |
| `Q` | Quit |

## Project Structure

```text
hand-gesture-drawing/
│
├── models/
│   └── hand_landmarker.task
│
├── src/
│   └── main.py
│
├── tests/
│   └── test_basic.py
│
├── drawings/
│   └── drawing.png
│
├── .gitignore
├── requirements.txt
└── README.md