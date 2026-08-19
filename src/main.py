import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/hand_landmarker.task"


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

    hand_landmarker = create_hand_landmarker()

    timestamp_ms = 0

    print("Hand tracking started. Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

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

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                for landmark in hand_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1,
                    )

        cv2.imshow("Hand Gesture Drawing", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hand_landmarker.close()
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()