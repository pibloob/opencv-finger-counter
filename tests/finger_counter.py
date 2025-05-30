"""
DEPRECATED VERSION – use main.py instead (modular structure).
Kept for legacy/testing purposes.
"""
import cv2
import numpy as np
import warnings
warnings.simplefilter('always', DeprecationWarning)

def detect_skin(frame):
    """Return binary skin mask (255 = skin)."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77],  dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask  = cv2.inRange(ycrcb, lower, upper)

    # open-close om ruis te verwijderen & gaatjes te vullen
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    mask  = cv2.inRange(ycrcb, lower, upper)
    mask  = cv2.medianBlur(mask, 5)                 # 1a
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, 2)  # 1b
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)  # 1c
    mask  = cv2.GaussianBlur(mask, (5,5), 0)        # lichte blur
    return mask

def count_fingers(hand_contour):
    """Returns counted fingers on screen"""
    hull_idx = cv2.convexHull(hand_contour, returnPoints=False)
    defects  = cv2.convexityDefects(hand_contour, hull_idx)

    finger_cnt, defect_pts = 0, []
    if defects is not None:
        for s, e, f, d in defects[:, 0]:
            start = hand_contour[s][0]
            end   = hand_contour[e][0]
            far   = hand_contour[f][0]
            depth = d / 256.0

            a = np.linalg.norm(end  - start)
            b = np.linalg.norm(far  - start)
            c = np.linalg.norm(far  - end)
            angle = np.degrees(np.arccos((b**2 + c**2 - a**2) /
                                         (2*b*c + 1e-5)))

            if depth > 20 and angle < 90:
                finger_cnt += 1
                defect_pts.append(tuple(far))
    return finger_cnt + (1 if finger_cnt else 0), defect_pts

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error");  return

    while True:
        ok, frame = cap.read()
        if not ok: break

        mask  = detect_skin(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # grootste contour = hand
            hand = max(contours, key=cv2.contourArea)
            if cv2.contourArea(hand) > 8000:          # grain?
                hull = cv2.convexHull(hand)
                cv2.drawContours(frame, [hull], -1, (0,255,0), 2)

                fingers, defects = count_fingers(hand)
                for p in defects:
                    cv2.circle(frame, p, 6, (0,0,255), -1)

                cv2.putText(frame, f'Fingers: {fingers}', (40,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,255), 4)

        # debug-vensters
        cv2.imshow('Finger Counter', frame)
        cv2.imshow('Skin Mask',     mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release();  cv2.destroyAllWindows()

if __name__ == '__main__':
    warnings.simplefilter('always', DeprecationWarning)
    warnings.warn(
        "You are using a deprecated script. Use 'finger_counter/main.py' instead.",
        DeprecationWarning, stacklevel=2
    )
    main()
