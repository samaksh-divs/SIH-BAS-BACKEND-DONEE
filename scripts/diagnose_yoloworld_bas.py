from pathlib import Path

from ultralytics import YOLOWorld
import torch
import cv2


ROOT = Path(__file__).resolve().parents[1]

IMAGE_ROOT = ROOT / "data" / "bas_images"
OUT = ROOT / "logs" / "yoloworld_diagnostics"

OUT.mkdir(parents=True, exist_ok=True)


CLASSES = [
    "person",
    "white container",
    "yellow box",
    "red box",
    "plant",
    "spray bottle",
]


MODEL_PATH = ROOT / "models" / "yolov8s-world.pt"


print("=" * 70)
print("YOLO-WORLD BAS VISUAL DIAGNOSTIC")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"

print()
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()
print("Model:", MODEL_PATH)

print()
print("Loading YOLO-World...")

model = YOLOWorld(str(MODEL_PATH))

model.set_classes(CLASSES)

print("Classes:")
for index, name in enumerate(CLASSES):
    print(f"  {index}: {name}")


FOLDERS = [
    "front_C11",
    "top_C01",
    "front_E07",
    "top_E01",
]


for folder_name in FOLDERS:

    folder = IMAGE_ROOT / folder_name

    images = sorted(folder.glob("*.jpg"))

    if not images:
        print()
        print("WARNING: no images found:", folder)
        continue

    # Ten representative frames from the 44-frame sequence.
    indices = [
        0,
        4,
        9,
        14,
        19,
        24,
        29,
        34,
        39,
        43,
    ]

    selected_images = [
        images[i]
        for i in indices
        if i < len(images)
    ]

    print()
    print("=" * 70)
    print(folder_name)
    print("=" * 70)

    for image_path in selected_images:

        print()
        print("-" * 70)
        print(image_path.name)

        results = model.predict(
            str(image_path),
            device=device,
            conf=0.01,
            iou=0.70,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            print("  NO DETECTIONS")
            continue

        boxes = result.boxes

        detection_count = len(boxes)

        print("  Detections:", detection_count)

        for box, cls_id, confidence in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.cls.cpu().tolist(),
            boxes.conf.cpu().tolist(),
        ):

            cls_id = int(cls_id)

            if cls_id < 0 or cls_id >= len(CLASSES):
                continue

            confidence = float(confidence)

            rounded_box = [
                round(float(value), 1)
                for value in box
            ]

            print(
                f"  {CLASSES[cls_id]:<18} "
                f"confidence={confidence:.3f} "
                f"box={rounded_box}"
            )

        annotated = result.plot()

        output_path = OUT / f"{folder_name}_{image_path.name}"

        cv2.imwrite(
            str(output_path),
            annotated
        )


print()
print("=" * 70)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)

print()
print("Annotated images saved to:")

print(OUT)