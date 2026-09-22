"""
Quick Spray Bottle Image Capture Tool
Auto-stops after 30 captures. Move the spray bottle around slowly.
"""
import cv2
import os
import time

SAVE_DIR = "data/spray_bottle_captures"
CAPTURE_INTERVAL = 2.0   # seconds between saves
MAX_CAPTURES = 30         # auto-stops when reached

os.makedirs(SAVE_DIR, exist_ok=True)

# Try camera indices 0 and 1 with DirectShow (fixes Windows MSMF error)
cap = None
for cam_idx in [0, 1, 2]:
    test = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if test.isOpened():
        cap = test
        print(f"  Camera opened on index {cam_idx}")
        break
    test.release()

if cap is None:
    print("ERROR: Could not open any camera. Make sure GUI is closed first.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("=" * 50)
print("SPRAY BOTTLE CAPTURE — auto-stops at 30 images")
print("=" * 50)
print("Hold spray bottle in front of WHITE wall.")
print("Move it slowly: vertical, tilted, horizontal,")
print("close, far. Keep moving until done!")
print("=" * 50)

count = 0
last_capture = time.time()

while count < MAX_CAPTURES:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    countdown = max(0.0, CAPTURE_INTERVAL - (now - last_capture))
    remaining = MAX_CAPTURES - count

    display = frame.copy()
    cv2.putText(display, f"Saved: {count}/{MAX_CAPTURES}  Next in: {countdown:.1f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"Remaining: {remaining} images",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(display, "Keep moving the spray bottle!",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow("Spray Capture — auto-stops at 30", display)
    key = cv2.waitKey(1) & 0xFF

    if (now - last_capture) >= CAPTURE_INTERVAL or key == ord(' '):
        filename = os.path.join(SAVE_DIR, f"spray_{count:04d}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
        last_capture = now
        print(f"  [{count}/{MAX_CAPTURES}] Saved {filename}")

    if key in (ord('q'), ord('Q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nDone! {count} images saved to '{SAVE_DIR}'")
print("Now upload these to CVAT, annotate as spray_bottle, and retrain.")

