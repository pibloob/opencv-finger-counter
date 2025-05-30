import cv2, numpy as np
import constants as C


def find_hand_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # grootste gebied nemen en area‑filter toepassen
    hand = max(contours, key=cv2.contourArea)
    if cv2.contourArea(hand) < C.MIN_HAND_AREA:
        return None
    # aspect‑ratio filter optioneel
    x, y, w, h = cv2.boundingRect(hand)
    ratio = h / float(w)
    if not (C.ASPECT_RANGE[0] <= ratio <= C.ASPECT_RANGE[1]):
        return None
    return hand


def count_fingers(contour):
    """
    Counts the number of fingers in a hand contour using convexity defects.
    This function analyzes the convexity defects of a hand contour to count extended fingers.
    It uses the depth of defects and angles between points to determine valid finger separations.
    """
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    defects      = cv2.convexityDefects(contour, hull_indices)
    if defects is None:
        return 0, []

    count, points = 0, []
    for s, e, f, d in defects[:, 0]:
        start, end, far = contour[s][0], contour[e][0], contour[f][0]
        depth = d / 256.0

        a = np.linalg.norm(end - start)
        b = np.linalg.norm(far - start)
        c = np.linalg.norm(far - end)
        angle = np.degrees(np.arccos((b**2 + c**2 - a**2) / (2*b*c + 1e-5)))

        if depth > C.MIN_DEFECT_DEPTH and angle < C.MAX_DEFECT_ANGLE:
            count  += 1
            points.append(tuple(far))

    return (count + 1) if count else 0, points