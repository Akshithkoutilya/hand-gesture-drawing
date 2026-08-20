import cv2
import mediapipe as mp
import os
import math

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

TOOLBAR_HEIGHT = 85

undo_stack = []
redo_stack = []

SMOOTHING_FACTOR = 0.65


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


def smooth_point(previous, current):
    if previous is None:
        return current

    x = int(
        previous[0] * SMOOTHING_FACTOR
        + current[0] * (1 - SMOOTHING_FACTOR)
    )

    y = int(
        previous[1] * SMOOTHING_FACTOR
        + current[1] * (1 - SMOOTHING_FACTOR)
    )

    return (x, y)


def get_toolbar_color(point):
    if point is None:
        return None

    x, y = point

    if y > TOOLBAR_HEIGHT:
        return None

    color_buttons = {
        "GREEN": (45, 40),
        "RED": (135, 40),
        "BLUE": (225, 40),
        "YELLOW": (315, 40),
    }

    for color_name, (button_x, button_y) in color_buttons.items():

        distance = math.sqrt(
            (x - button_x) ** 2
            + (y - button_y) ** 2
        )

        if distance <= 30:
            return color_name

    return None


def draw_toolbar(frame, color_name, size, gesture):

    cv2.rectangle(
        frame,
        (0, 0),
        (CANVAS_WIDTH, TOOLBAR_HEIGHT),
        (35, 35, 35),
        -1,
    )

    color_buttons = {
        "GREEN": (45, 40),
        "RED": (135, 40),
        "BLUE": (225, 40),
        "YELLOW": (315, 40),
    }

    for name, position in color_buttons.items():

        cv2.circle(
            frame,
            position,
            20,
            COLORS[name],
            -1,
        )

        if name == color_name:

            cv2.circle(
                frame,
                position,
                28,
                (255, 255, 255),
                3,
            )

    cv2.putText(
        frame,
        f"COLOR: {color_name}",
        (370, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"BRUSH: {size}px",
        (370, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"TOOL: {gesture}",
        (560, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        COLORS[color_name],
        2,
    )

    cv2.putText(
        frame,
        "1-4 Colors | +/- Size | Z Undo | Y Redo | S Save | C Clear | Q Quit",
        (560, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
    )


def save_canvas(canvas):

    os.makedirs("drawings", exist_ok=True)

    filename = "drawings/drawing.png"

    success = cv2.imwrite(
        filename,
        canvas,
    )

    if success:
        print(f"Drawing saved: {filename}")
    else:
        print("Error: Could not save drawing.")


def main():

    global current_color_name
    global brush_size

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CANVAS_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CANVAS_HEIGHT,
    )

    canvas = None

    previous_point = None
    smooth_previous_point = None

    hand_landmarker = create_hand_landmarker()

    timestamp_ms = 0

    drawing_stroke = False
    erasing_stroke = False

    stroke_start_canvas = None

    previous_gesture = "READY"

    print()
    print("====================================")
    print(" Hand Gesture Drawing Application")
    print("====================================")
    print()
    print("INDEX FINGER        -> DRAW")
    print("INDEX + MIDDLE      -> ERASER")
    print("MOVE INDEX TO COLOR -> SELECT COLOR")
    print()
    print("1 -> GREEN")
    print("2 -> RED")
    print("3 -> BLUE")
    print("4 -> YELLOW")
    print("+ -> Increase brush")
    print("- -> Decrease brush")
    print("Z -> Undo")
    print("Y -> Redo")
    print("S -> Save drawing")
    print("C -> Clear canvas")
    print("Q -> Quit")
    print()

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

            gesture = get_gesture(
                hand_landmarks
            )

            index_finger = hand_landmarks[8]

            raw_point = (
                int(index_finger.x * frame.shape[1]),
                int(index_finger.y * frame.shape[0]),
            )

            current_point = smooth_point(
                smooth_previous_point,
                raw_point,
            )

            smooth_previous_point = current_point

            selected_color = get_toolbar_color(
                current_point
            )

            if selected_color is not None:

                current_color_name = selected_color

                previous_point = None
                smooth_previous_point = None

                drawing_stroke = False
                erasing_stroke = False

                stroke_start_canvas = None

                gesture = "COLOR"

            elif gesture != previous_gesture:

                if drawing_stroke or erasing_stroke:

                    if stroke_start_canvas is not None:

                        undo_stack.append(
                            stroke_start_canvas
                        )

                    drawing_stroke = False
                    erasing_stroke = False

                    stroke_start_canvas = None

                previous_point = None

            if gesture == "DRAW":

                if not drawing_stroke:

                    stroke_start_canvas = canvas.copy()

                    drawing_stroke = True
                    erasing_stroke = False

                    redo_stack.clear()

                color = COLORS[
                    current_color_name
                ]

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

                previous_point = current_point

            elif gesture == "ERASER":

                if not erasing_stroke:

                    stroke_start_canvas = canvas.copy()

                    erasing_stroke = True
                    drawing_stroke = False

                    redo_stack.clear()

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

        else:

            if drawing_stroke or erasing_stroke:

                if stroke_start_canvas is not None:

                    undo_stack.append(
                        stroke_start_canvas
                    )

                drawing_stroke = False
                erasing_stroke = False

                stroke_start_canvas = None

            previous_point = None
            smooth_previous_point = None

        previous_gesture = gesture

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
            gesture,
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

        elif key == ord("z"):

            if undo_stack:

                redo_stack.append(
                    canvas.copy()
                )

                canvas = undo_stack.pop()

                previous_point = None

        elif key == ord("y"):

            if redo_stack:

                undo_stack.append(
                    canvas.copy()
                )

                canvas = redo_stack.pop()

                previous_point = None

        elif key == ord("s"):

            save_canvas(canvas)

        elif key == ord("c"):

            if cv2.countNonZero(
                cv2.cvtColor(
                    canvas,
                    cv2.COLOR_BGR2GRAY,
                )
            ) > 0:

                undo_stack.append(
                    canvas.copy()
                )

                canvas[:] = 0

                redo_stack.clear()

            previous_point = None

    hand_landmarker.close()

    camera.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()