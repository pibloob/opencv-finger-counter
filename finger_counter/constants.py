# ---------- Skin detectie (YCrCb‑range)
YCRCB_LOWER = (0, 133, 77)
YCRCB_UPPER = (255, 173, 127)

# Morphology
KERNEL_SIZE    = 5        # ellips‑kernel (px)
OPEN_ITER      = 2
CLOSE_ITER     = 2

# Contour filtering
MIN_HAND_AREA  = 8_000    
ASPECT_RANGE   = (1.0, 2.5)

# Finger‑defect
MIN_DEFECT_DEPTH = 20     # approx. 8 px in 256‑fixed‑pt
MAX_DEFECT_ANGLE = 90     