import os
import cv2
import numpy as np

from src.main import (
    is_finger_up,
    smooth_point,
    get_toolbar_color,
    save_canvas,
)


def test_smooth_point_first_value():
    point = (100, 200)

    result = smooth_point(None, point)

    assert result == point


def test_smooth_point_moves_toward_new_point():
    previous = (100, 100)
    current = (200, 200)

    result = smooth_point(previous, current)

    assert 100 < result[0] < 200
    assert 100 < result[1] < 200


def test_toolbar_green():
    assert get_toolbar_color((45, 40)) == "GREEN"


def test_toolbar_red():
    assert get_toolbar_color((135, 40)) == "RED"


def test_toolbar_blue():
    assert get_toolbar_color((225, 40)) == "BLUE"


def test_toolbar_yellow():
    assert get_toolbar_color((315, 40)) == "YELLOW"


def test_toolbar_outside():
    assert get_toolbar_color((500, 500)) is None


def test_finger_up():
    landmarks = [None] * 21

    landmarks[8] = type("Point", (), {"y": 0.2})()
    landmarks[6] = type("Point", (), {"y": 0.5})()

    assert is_finger_up(landmarks, 8, 6) is True


def test_finger_down():
    landmarks = [None] * 21

    landmarks[8] = type("Point", (), {"y": 0.7})()
    landmarks[6] = type("Point", (), {"y": 0.5})()

    assert is_finger_up(landmarks, 8, 6) is False


def test_save_canvas(tmp_path):
    canvas = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    original_directory = os.getcwd()

    os.chdir(tmp_path)

    try:
        save_canvas(canvas)

        saved_file = tmp_path / "drawings" / "drawing.png"

        assert saved_file.exists()

        image = cv2.imread(str(saved_file))

        assert image is not None
        assert image.shape == (100, 100, 3)

    finally:
        os.chdir(original_directory)