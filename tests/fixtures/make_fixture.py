"""Generate a synthetic black-on-white '4+3' PNG for smoke testing the endpoint.

Not a correctness oracle for recognition (the CNN may read it differently);
it exists to prove the decode -> segment -> predict -> eval path runs without
crashing inside the container.
"""
import cv2
import numpy as np

img = np.full((256, 800), 255, np.uint8)
cv2.putText(img, "4+3", (250, 180), cv2.FONT_HERSHEY_SIMPLEX, 6, (0,), 12)
cv2.imwrite("tests/fixtures/sample.png", img)
print("wrote tests/fixtures/sample.png")
