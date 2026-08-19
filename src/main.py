import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/hand_landmarker.task"

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720

BRUSH_COLOR = (0, 255, 0)
BRUSH_SIZE = 8


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


def main():
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
    print("Move your index finger to draw.")
    print("Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = frame.copy()
            canvas[:] = 0

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms += 33

        result = hand_landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        current_point = None

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]

            index_finger = hand_landmarks[8]

            x = int(index_finger.x * frame.shape[1])
            y = int(index_finger.y * frame.shape[0])

            current_point = (x, y)

            cv2.circle(
                frame,
                current_point,
                10,
                BRUSH_COLOR,
                -1,
            )

            if previous_point is not None:
                cv2.line(
                    canvas,
                    previous_point,
                    current_point,
                    BRUSH_COLOR,
                    BRUSH_SIZE,
                )

        else:
            previous_point = None

        if current_point is not None:
            previous_point = current_point

        combined = cv2.addWeighted(
            frame,
            1.0,
            canvas,
            1.0,
            0,
        )

        cv2.imshow("Hand Gesture Drawing", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hand_landmarker.close()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()