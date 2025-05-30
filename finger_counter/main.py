import cv2, argparse
from skin import detect_skin, mask_out_faces
from hand import find_hand_contour, count_fingers


def run(debug=False):
    """Run the finger counting application using webcam feed.
    This function initializes the webcam, processes each frame to detect skin, 
    masks out faces, finds hand contours, and counts extended fingers in real-time.
    The output is displayed in a window showing the original frame with detected
    hand contours and finger count overlaid.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("No webcam found")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        mask = detect_skin(frame)
        mask_out_faces(mask, frame)

        hand = find_hand_contour(mask)
        if hand is not None:
            hull = cv2.convexHull(hand)
            cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)

            fingers, defect_pts = count_fingers(hand)
            for pt in defect_pts:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Fingers: {fingers}", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)

        cv2.imshow("Finger Counter", frame)
        if debug:
            cv2.imshow("Skin Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true",
                        help="toon extra vensters met skin‑mask")
    args = parser.parse_args()
    run(debug=args.debug)