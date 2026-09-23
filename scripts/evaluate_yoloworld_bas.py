from pathlib import Path
from collections import defaultdict
from ultralytics import YOLOWorld
import torch
import json
import yaml
import math

ROOT = Path(__file__).resolve().parents[1]

IMAGE_ROOT = ROOT / "data" / "bas_images"
ANN_ROOT = ROOT / "data" / "annotations"

CLASSES = [
    "person",
    "white_container",
    "red_box",
    "yellow_box",
    "plant",
    "spray_bottle",
]

WORLD_CLASSES = [
    "person",
    "white container",
    "yellow box",
    "red box",
    "plant",
    "spray bottle",
]

# Annotation mappings -> canonical mapping
CANONICAL = {
    "person": 0,
    "white_container": 1,
    "red_box": 2,
    "yellow_box": 3,
    "plant": 4,
    "spray_bottle": 5,
}

DATASET_MAPPINGS = {
    "correct_01_02_03": {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
    },
    "correct_13_14_15": {
        0: 0,
        1: 1,
        2: 3,  # yellow_box
        3: 4,  # plant
        4: 5,  # spray_bottle
        5: 2,  # red_box
    },
    "error": {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
    },
}

# Exact image -> annotation dataset mapping.
DATASETS = {
    "front_C11": ("correct_13_14_15", "procedure"),
    "top_C01": ("correct_01_02_03", "procedure_first"),
    "front_E07": ("correct_01_02_03", "candidate"),
    "top_E01": ("error", "candidate_top_E01"),
}


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def yolo_to_xyxy(cx, cy, w, h, image_w, image_h):
    x1 = (cx - w / 2.0) * image_w
    y1 = (cy - h / 2.0) * image_h
    x2 = (cx + w / 2.0) * image_w
    y2 = (cy + h / 2.0) * image_h

    return [x1, y1, x2, y2]


def read_labels(label_path, dataset_name, image_w, image_h):
    mapping = DATASET_MAPPINGS[dataset_name]
    result = []

    if not label_path.exists():
        return result

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        cls_id = int(parts[0])

        if cls_id not in mapping:
            continue

        canonical_id = mapping[cls_id]

        cx, cy, w, h = map(float, parts[1:])

        box = yolo_to_xyxy(
            cx, cy, w, h,
            image_w,
            image_h
        )

        result.append({
            "class_id": canonical_id,
            "box": box,
        })

    return result


def find_label(dataset_name, mode, image_name, frame_number):
    base = ANN_ROOT / dataset_name / "labels" / "train"

    if mode == "candidate":
        return base / f"candidate_{frame_number:02d}.txt"

    if mode == "candidate_top_E01":
        return base / "top_E01" / f"candidate_{frame_number:02d}.txt"

    if mode == "procedure":
        return base / image_name.replace(".jpg", ".txt")

    if mode == "procedure_first":
        # First 44 frames of correct_01_02_03
        # Match by exact filename.
        return base / image_name.replace(".jpg", ".txt")

    raise ValueError(f"Unknown mode: {mode}")


def collect_images():
    items = []

    for folder_name, (dataset_name, mode) in DATASETS.items():
        folder = IMAGE_ROOT / folder_name

        images = sorted(folder.glob("*.jpg"))

        if len(images) != 44:
            print(
                f"WARNING: {folder_name} contains {len(images)} images, "
                f"expected 44"
            )

        for index, image_path in enumerate(images, start=1):
            label_path = find_label(
                dataset_name,
                mode,
                image_path.name,
                index
            )

            items.append({
                "camera": folder_name,
                "image": image_path,
                "label": label_path,
                "dataset": dataset_name,
            })

    return items


def match_predictions(gt, predictions, iou_threshold):
    matched_gt = set()
    matched_pred = set()

    pairs = []

    candidates = []

    for pi, pred in enumerate(predictions):
        for gi, truth in enumerate(gt):
            if pred["class_id"] != truth["class_id"]:
                continue

            overlap = iou(pred["box"], truth["box"])

            if overlap >= iou_threshold:
                candidates.append(
                    (overlap, pi, gi)
                )

    candidates.sort(reverse=True)

    for overlap, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue

        matched_pred.add(pi)
        matched_gt.add(gi)

        pairs.append((pi, gi, overlap))

    tp = len(pairs)
    fp = len(predictions) - tp
    fn = len(gt) - len(matched_gt)

    return tp, fp, fn


def evaluate_threshold(model, items, confidence, iou_threshold=0.50):
    totals = {
        cls: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for cls in CLASSES
    }

    image_count = 0
    missing_labels = []

    for item in items:
        image_path = item["image"]
        label_path = item["label"]
        dataset_name = item["dataset"]

        # Read image dimensions through PIL.
        from PIL import Image

        with Image.open(image_path) as im:
            image_w, image_h = im.size

        gt = read_labels(
            label_path,
            dataset_name,
            image_w,
            image_h
        )

        if not label_path.exists():
            missing_labels.append(
                str(label_path.relative_to(ROOT))
            )

        result = model.predict(
            str(image_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
            conf=confidence,
            iou=0.70,
            verbose=False
        )[0]

        predictions = []

        if result.boxes is not None:
            for box, cls_id, conf_score in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                result.boxes.conf.cpu().tolist()
            ):
                cls_id = int(cls_id)

                if cls_id < 0 or cls_id >= len(CLASSES):
                    continue

                predictions.append({
                    "class_id": cls_id,
                    "box": box,
                    "confidence": float(conf_score),
                })

        for cls_id, cls_name in enumerate(CLASSES):
            gt_cls = [
                x for x in gt
                if x["class_id"] == cls_id
            ]

            pred_cls = [
                x for x in predictions
                if x["class_id"] == cls_id
            ]

            tp, fp, fn = match_predictions(
                gt_cls,
                pred_cls,
                iou_threshold
            )

            totals[cls_name]["tp"] += tp
            totals[cls_name]["fp"] += fp
            totals[cls_name]["fn"] += fn

        image_count += 1

    metrics = {}

    for cls_name, values in totals.items():
        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        metrics[cls_name] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    total_tp = sum(x["tp"] for x in totals.values())
    total_fp = sum(x["fp"] for x in totals.values())
    total_fn = sum(x["fn"] for x in totals.values())

    overall_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    overall_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    overall_f1 = (
        2 * overall_precision * overall_recall /
        (overall_precision + overall_recall)
        if overall_precision + overall_recall > 0
        else 0.0
    )

    return {
        "confidence": confidence,
        "iou": iou_threshold,
        "images": image_count,
        "metrics": metrics,
        "overall": {
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
        },
        "missing_labels": missing_labels,
    }


def main():
    print("=" * 70)
    print("BAS YOLO-WORLD REAL DATASET EVALUATION")
    print("=" * 70)

    print("\nModel:")
    print("  models/yolov8s-world.pt")

    print("\nClasses:")
    for i, cls in enumerate(CLASSES):
        print(f"  {i}: {cls}")

    print(
        "\nDevice:",
        "CUDA" if torch.cuda.is_available() else "CPU"
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    items = collect_images()

    print("\nImages:")
    for folder in DATASETS:
        count = sum(
            1 for x in items
            if x["camera"] == folder
        )
        print(f"  {folder}: {count}")

    print(f"\nTotal images: {len(items)}")

    print("\nLoading YOLO-World...")

    model = YOLOWorld(
        str(ROOT / "models" / "yolov8s-world.pt")
    )

    model.set_classes(WORLD_CLASSES)

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
    ]

    all_results = []

    for threshold in thresholds:
        print(
            f"\n{'=' * 70}\n"
            f"CONFIDENCE THRESHOLD: {threshold:.2f}\n"
            f"{'=' * 70}"
        )

        result = evaluate_threshold(
            model,
            items,
            confidence=threshold,
            iou_threshold=0.50
        )

        all_results.append(result)

        print(
            "\nClass                 "
            "Precision   Recall      F1"
        )
        print("-" * 55)

        for cls_name in CLASSES:
            m = result["metrics"][cls_name]

            print(
                f"{cls_name:<20} "
                f"{m['precision'] * 100:7.2f}%   "
                f"{m['recall'] * 100:7.2f}%   "
                f"{m['f1'] * 100:7.2f}%"
            )

        o = result["overall"]

        print("-" * 55)
        print(
            f"{'OVERALL':<20} "
            f"{o['precision'] * 100:7.2f}%   "
            f"{o['recall'] * 100:7.2f}%   "
            f"{o['f1'] * 100:7.2f}%"
        )

    output_path = ROOT / "logs" / "yoloworld_bas_evaluation.json"

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        json.dumps(
            all_results,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)
    print(output_path)

    print("\nIMPORTANT:")
    print(
        "These are IoU@0.50 detection metrics, not classification "
        "accuracy."
    )
    print(
        "Do not change the production confidence threshold based "
        "on this test alone."
    )


if __name__ == "__main__":
    main()