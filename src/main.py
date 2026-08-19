import cv2


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam started. Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame.")
            break

        cv2.imshow("Hand Gesture Drawing", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()