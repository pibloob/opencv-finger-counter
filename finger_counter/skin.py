import cv2, numpy as np
import constants as C

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                     (C.KERNEL_SIZE, C.KERNEL_SIZE))

def detect_skin(frame):
    """Return binary mask (uint8) in white"""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask  = cv2.inRange(ycrcb, C.YCRCB_LOWER, C.YCRCB_UPPER)

    # ruis & gaatjes
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _KERNEL, C.OPEN_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL, C.CLOSE_ITER)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return mask


def mask_out_faces(mask, frame):
    """Face-box in mask coloured."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 0, -1)
    return mask  # gemodificeerd in‑place, maar return is handig