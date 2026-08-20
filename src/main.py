import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/hand_landmarker.task"

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720

COLORS = {
    "GREEN": (0, 255, 0),
    "RED": (0, 0, 255),
    "BLUE": (255, 0, 0),
    "YELLOW": (0, 255, 255),
}

current_color_name = "GREEN"
brush_size = 8

MIN_BRUSH_SIZE = 2
MAX_BRUSH_SIZE = 30

ERASER_SIZE = 40


def create_hand_landmarker():
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    return vision.HandLandmarker.create_from_options(options)


def is_finger_up(hand_landmarks, tip, pip):
    return hand_landmarks[tip].y < hand_landmarks[pip].y


def get_gesture(hand_landmarks):
    index_up = is_finger_up(hand_landmarks, 8, 6)
    middle_up = is_finger_up(hand_landmarks, 12, 10)

    if index_up and middle_up:
        return "ERASER"

    if index_up and not middle_up:
        return "DRAW"

    return "READY"


def draw_toolbar(frame, color_name, size):
    toolbar_height = 70

    cv2.rectangle(
        frame,
        (0, 0),
        (CANVAS_WIDTH, toolbar_height),
        (40, 40, 40),
        -1,
    )

    x_positions = {
        "GREEN": 40,
        "RED": 130,
        "BLUE": 220,
        "YELLOW": 310,
    }

    for name, x in x_positions.items():
        color = COLORS[name]

        cv2.circle(
            frame,
            (x, 35),
            20,
            color,
            -1,
        )

        if name == color_name:
            cv2.circle(
                frame,
                (x, 35),
                26,
                (255, 255, 255),
                2,
            )

    cv2.putText(
        frame,
        f"Brush: {size}",
        (400, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "1 Green  2 Red  3 Blue  4 Yellow",
        (540, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "+ / - Size   C Clear   Q Quit",
        (540, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def main():
    global current_color_name
    global brush_size

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CANVAS_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CANVAS_HEIGHT)

    canvas = None
    previous_point = None

    hand_landmarker = create_hand_landmarker()

    timestamp_ms = 0

    print("Hand Gesture Drawing started.")
    print()
    print("Index finger       = Draw")
    print("Index + Middle     = Eraser")
    print("1                  = Green")
    print("2                  = Red")
    print("3                  = Blue")
    print("4                  = Yellow")
    print("+                  = Increase brush")
    print("-                  = Decrease brush")
    print("C                  = Clear canvas")
    print("Q                  = Quit")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = frame.copy()
            canvas[:] = 0

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms += 33

        result = hand_landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        gesture = "READY"
        current_point = None

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]

            gesture = get_gesture(hand_landmarks)

            index_finger = hand_landmarks[8]

            x = int(
                index_finger.x * frame.shape[1]
            )

            y = int(
                index_finger.y * frame.shape[0]
            )

            current_point = (x, y)

            if gesture == "DRAW":

                color = COLORS[current_color_name]

                cv2.circle(
                    frame,
                    current_point,
                    brush_size,
                    color,
                    -1,
                )

                if previous_point is not None:
                    cv2.line(
                        canvas,
                        previous_point,
                        current_point,
                        color,
                        brush_size,
                    )

            elif gesture == "ERASER":

                cv2.circle(
                    frame,
                    current_point,
                    ERASER_SIZE // 2,
                    (255, 255, 255),
                    2,
                )

                cv2.circle(
                    canvas,
                    current_point,
                    ERASER_SIZE // 2,
                    (0, 0, 0),
                    -1,
                )

            previous_point = current_point

        else:
            previous_point = None

        combined = cv2.addWeighted(
            frame,
            1.0,
            canvas,
            1.0,
            0,
        )

        draw_toolbar(
            combined,
            current_color_name,
            brush_size,
        )

        cv2.putText(
            combined,
            gesture,
            (30, CANVAS_HEIGHT - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            COLORS[current_color_name],
            2,
        )

        cv2.imshow(
            "Hand Gesture Drawing",
            combined,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("1"):
            current_color_name = "GREEN"

        elif key == ord("2"):
            current_color_name = "RED"

        elif key == ord("3"):
            current_color_name = "BLUE"

        elif key == ord("4"):
            current_color_name = "YELLOW"

        elif key == ord("+") or key == ord("="):
            brush_size = min(
                brush_size + 2,
                MAX_BRUSH_SIZE,
            )

        elif key == ord("-"):
            brush_size = max(
                brush_size - 2,
                MIN_BRUSH_SIZE,
            )

        elif key == ord("c"):
            canvas[:] = 0
            previous_point = None

    hand_landmarker.close()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()