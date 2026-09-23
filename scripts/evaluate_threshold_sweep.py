from pathlib import Path
from collections import defaultdict
import json
import re

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# BAS YOLO-WORLD CORRECTED EVALUATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "best.pt"

IMAGE_DIRS = {
    "front_C11": ROOT / "data" / "bas_images" / "front_C11",
    "top_C01": ROOT / "data" / "bas_images" / "top_C01",
    "front_E07": ROOT / "data" / "bas_images" / "front_E07",
    "top_E01": ROOT / "data" / "bas_images" / "top_E01",
}

LABEL_DIRS = {
    "front_C11": (
        ROOT / "data" / "annotations"
        / "correct_13_14_15"
        / "labels" / "train"
    ),
    "top_C01": (
        ROOT / "data" / "annotations"
        / "correct_01_02_03"
        / "labels" / "train"
    ),
    "front_E07": (
        ROOT / "data" / "annotations"
        / "correct_01_02_03"
        / "labels" / "train"
    ),
    "top_E01": (
        ROOT / "data" / "annotations"
        / "error"
        / "labels" / "train" / "top_E01"
    ),
}

CLASSES = [
    "person",
    "white_container",
    "red_box",
    "yellow_box",
    "plant",
    "spray_bottle",
]

# correct_01_02_03:
# 0 person
# 1 white_container
# 2 yellow_box
# 3 plant
# 4 spray_bottle
# 5 red_box
REMAP_CORRECT_01_02_03 = {
    0: 0,
    1: 1,
    2: 3,
    3: 4,
    4: 5,
    5: 2,
}

# correct_13_14_15 and error already use canonical mapping
REMAP_CORRECT_13_14_15 = None
REMAP_ERROR = None

CONF_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]
IOU_THRESHOLD = 0.25
DEVICE = "cuda"


# ============================================================
# HELPERS
# ============================================================

def extract_procedure_index(stem):
    """
    Extract the leading procedure number.

    Examples:
        001_open_white_box_frame13 -> 1
        044_close_white_box_frame1826 -> 44
        candidate_01 -> None
    """
    match = re.match(r"^(\d{3})_", stem)

    if not match:
        return None

    return int(match.group(1))


def extract_frame_number(stem):
    """
    Extract the frame number from names such as:

        003_retrieve_red_box_frame89

    Returns:
        89
    """
    match = re.search(r"_frame(\d+)(?:_1)?$", stem)

    if not match:
        return None

    return int(match.group(1))


def get_front_c11_label(image_path, label_dir):
    """
    Match a front_C11 image to its annotation.

    front_C11 image frame numbers do not exactly match the annotation
    frame numbers.

    Therefore we match using:

        1. Same procedure index (001-044)
        2. Same action name
        3. Closest annotation frame number

    Example:

        image:
        003_retrieve_red_box_frame89.jpg

        available labels:
        003_retrieve_red_box_frame81.txt
        003_retrieve_red_box_frame88.txt
        003_retrieve_red_box_frame90.txt

        selected:
        003_retrieve_red_box_frame88.txt
    """

    image_stem = image_path.stem

    procedure_index = extract_procedure_index(image_stem)
    image_frame = extract_frame_number(image_stem)

    if procedure_index is None or image_frame is None:
        return None

    prefix = f"{procedure_index:03d}_"

    candidates = []

    for label_path in label_dir.glob(f"{prefix}*.txt"):
        label_stem = label_path.stem

        label_frame = extract_frame_number(label_stem)

        if label_frame is None:
            continue

        # Avoid suffix duplicates such as *_1 when a cleaner
        # matching annotation exists.
        candidates.append((abs(label_frame - image_frame), label_frame, label_path))

    if not candidates:
        return None

    # Closest frame number wins.
    candidates.sort(key=lambda item: (item[0], item[1]))

    return candidates[0][2]


def get_label_path(camera_name, image_path):
    """
    Return the annotation file corresponding to an image.
    """

    label_dir = LABEL_DIRS[camera_name]

    # --------------------------------------------------------
    # front_C11 requires special matching because its frame
    # numbers differ from the annotation frame numbers.
    # --------------------------------------------------------
    if camera_name == "front_C11":
        return get_front_c11_label(image_path, label_dir)

    # --------------------------------------------------------
    # All other cameras use exact stem matching.
    # --------------------------------------------------------
    exact = label_dir / f"{image_path.stem}.txt"

    if exact.exists():
        return exact

    return None


def load_yolo_labels(label_path, image_width, image_height, remap=None):
    """
    Load YOLO-format labels.

    Format:
        class_id x_center y_center width height

    Returns:
        list of dictionaries with:
            class_id
            bbox
    """

    labels = []

    if label_path is None or not label_path.exists():
        return labels

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        try:
            class_id = int(parts[0])
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue

        if remap is not None:
            if class_id not in remap:
                continue

            class_id = remap[class_id]

        x1 = (xc - w / 2) * image_width
        y1 = (yc - h / 2) * image_height
        x2 = (xc + w / 2) * image_width
        y2 = (yc + h / 2) * image_height

        labels.append(
            {
                "class_id": class_id,
                "bbox": np.array(
                    [x1, y1, x2, y2],
                    dtype=np.float32,
                ),
            }
        )

    return labels


def get_remap(camera_name):
    if camera_name in ["top_C01", "front_E07"]:
        return REMAP_CORRECT_01_02_03

    if camera_name == "front_C11":
        return REMAP_CORRECT_13_14_15

    if camera_name == "top_E01":
        return REMAP_ERROR

    return None


def bbox_iou(box_a, box_b):
    """
    Calculate IoU between two xyxy bounding boxes.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)

    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def get_predictions(model, image, confidence):
    """
    Run YOLO-World prediction using the six configured classes.
    """

    results = model.predict(
        source=image,
        conf=confidence,
        imgsz=640,
        device=DEVICE,
        verbose=False,
    )

    predictions = []

    if not results:
        return predictions

    result = results[0]

    if result.boxes is None:
        return predictions

    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())

        xyxy = box.xyxy[0].cpu().numpy().astype(np.float32)

        predictions.append(
            {
                "class_id": cls_id,
                "confidence": conf,
                "bbox": xyxy,
            }
        )

    return predictions


def match_predictions(gt_labels, predictions, iou_threshold):
    """
    Class-aware matching.

    A prediction can only match a ground-truth object of the
    same class.
    """

    matches = []

    used_gt = set()
    used_pred = set()

    pairs = []

    for gt_index, gt in enumerate(gt_labels):
        for pred_index, pred in enumerate(predictions):

            if gt["class_id"] != pred["class_id"]:
                continue

            iou = bbox_iou(gt["bbox"], pred["bbox"])

            if iou >= iou_threshold:
                pairs.append(
                    (
                        iou,
                        gt_index,
                        pred_index,
                    )
                )

    pairs.sort(reverse=True)

    for iou, gt_index, pred_index in pairs:

        if gt_index in used_gt:
            continue

        if pred_index in used_pred:
            continue

        used_gt.add(gt_index)
        used_pred.add(pred_index)

        matches.append(
            (
                gt_index,
                pred_index,
                iou,
            )
        )

    return matches, used_gt, used_pred


def calculate_metrics(gt_labels, predictions, iou_threshold):
    """
    Calculate class-aware TP/FP/FN.
    """

    matches, used_gt, used_pred = match_predictions(
        gt_labels,
        predictions,
        iou_threshold,
    )

    metrics = {
        class_id: {
            "TP": 0,
            "FP": 0,
            "FN": 0,
        }
        for class_id in range(len(CLASSES))
    }

    for gt_index, pred_index, _ in matches:
        class_id = gt_labels[gt_index]["class_id"]
        metrics[class_id]["TP"] += 1

    for gt_index, gt in enumerate(gt_labels):
        if gt_index not in used_gt:
            metrics[gt["class_id"]]["FN"] += 1

    for pred_index, pred in enumerate(predictions):
        if pred_index not in used_pred:
            metrics[pred["class_id"]]["FP"] += 1

    return metrics


def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1


# ============================================================
# CONFUSION ANALYSIS
# ============================================================

def build_confusion_matrix(
    gt_labels,
    predictions,
    iou_threshold,
):
    """
    Build a confusion matrix.

    Rows:
        Ground truth

    Columns:
        Predicted class

    Only spatially overlapping objects are considered.
    """

    matrix = np.zeros(
        (len(CLASSES), len(CLASSES)),
        dtype=int,
    )

    used_gt = set()
    used_pred = set()

    candidate_pairs = []

    for gt_index, gt in enumerate(gt_labels):

        for pred_index, pred in enumerate(predictions):

            iou = bbox_iou(
                gt["bbox"],
                pred["bbox"],
            )

            if iou >= iou_threshold:
                candidate_pairs.append(
                    (
                        iou,
                        gt_index,
                        pred_index,
                    )
                )

    candidate_pairs.sort(reverse=True)

    for iou, gt_index, pred_index in candidate_pairs:

        if gt_index in used_gt:
            continue

        if pred_index in used_pred:
            continue

        gt_class = gt_labels[gt_index]["class_id"]
        pred_class = predictions[pred_index]["class_id"]

        matrix[gt_class][pred_class] += 1

        used_gt.add(gt_index)
        used_pred.add(pred_index)

    return matrix


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 78)
    print("BAS YOLO-WORLD CORRECTED EVALUATION")
    print("=" * 78)

    print()
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print()

    model = YOLO(str(MODEL_PATH))

    # --------------------------------------------------------
    # IMAGE / LABEL PAIRING
    # --------------------------------------------------------

    print("Checking image/annotation pairing...")
    print()

    all_images = []

    for camera_name, image_dir in IMAGE_DIRS.items():

        images = sorted(
            image_dir.glob("*.jpg")
        )

        print(
            f"{camera_name:12s}: {len(images)} images"
        )

        for image_path in images:

            label_path = get_label_path(
                camera_name,
                image_path,
            )

            if label_path is None:

                print(
                    f"WARNING: Missing annotation for "
                    f"{camera_name}/{image_path.name}"
                )

                continue

            all_images.append(
                (
                    camera_name,
                    image_path,
                    label_path,
                )
            )

    total_images = sum(
        len(list(path.glob("*.jpg")))
        for path in IMAGE_DIRS.values()
    )

    valid_images = len(all_images)

    print()
    print(
        f"Total images evaluated: {total_images}"
    )

    print(
        f"Images with valid annotations: {valid_images}"
    )

    if valid_images != total_images:

        print()
        print(
            f"WARNING: Expected {total_images} "
            f"evaluated images but found {valid_images}."
        )

        print(
            "DO NOT treat the metrics as final."
        )

    else:

        print()
        print(
            "SUCCESS: All images have valid annotation mappings."
        )

    # --------------------------------------------------------
    # VALIDATE FRONT_C11 MAPPING
    # --------------------------------------------------------

    print()
    print("-" * 78)
    print("FRONT_C11 ANNOTATION MAPPING CHECK")
    print("-" * 78)

    front_c11_pairs = [
        item
        for item in all_images
        if item[0] == "front_C11"
    ]

    for camera_name, image_path, label_path in front_c11_pairs:

        print(
            f"{image_path.name:55s} -> {label_path.name}"
        )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    results_json = {
        "model": str(MODEL_PATH),
        "device": DEVICE,
        "total_images": total_images,
        "valid_images": valid_images,
        "iou_threshold": IOU_THRESHOLD,
        "confidence_thresholds": CONF_THRESHOLDS,
        "metrics": {},
        "confusion": {},
    }

    for confidence in CONF_THRESHOLDS:

        print()
        print("=" * 78)
        print(
            f"CLASS-AWARE METRICS "
            f"CONF={confidence:.2f} "
            f"IoU={IOU_THRESHOLD:.2f}"
        )
        print("=" * 78)

        totals = {
            class_id: {
                "TP": 0,
                "FP": 0,
                "FN": 0,
            }
            for class_id in range(len(CLASSES))
        }

        for camera_name, image_path, label_path in all_images:

            image = cv2.imread(str(image_path))

            if image is None:
                continue

            height, width = image.shape[:2]

            remap = get_remap(camera_name)

            gt_labels = load_yolo_labels(
                label_path,
                width,
                height,
                remap,
            )

            predictions = get_predictions(
                model,
                image,
                confidence,
            )

            image_metrics = calculate_metrics(
                gt_labels,
                predictions,
                IOU_THRESHOLD,
            )

            for class_id in totals:

                totals[class_id]["TP"] += image_metrics[class_id]["TP"]
                totals[class_id]["FP"] += image_metrics[class_id]["FP"]
                totals[class_id]["FN"] += image_metrics[class_id]["FN"]

        print(
            f"{'Class':20s}"
            f"{'TP':>8s}"
            f"{'FP':>8s}"
            f"{'FN':>8s}"
            f"{'Precision':>14s}"
            f"{'Recall':>12s}"
            f"{'F1':>10s}"
        )

        print("-" * 78)

        overall_tp = 0
        overall_fp = 0
        overall_fn = 0

        confidence_results = {}

        for class_id, class_name in enumerate(CLASSES):

            tp = totals[class_id]["TP"]
            fp = totals[class_id]["FP"]
            fn = totals[class_id]["FN"]

            precision, recall, f1 = precision_recall_f1(
                tp,
                fp,
                fn,
            )

            overall_tp += tp
            overall_fp += fp
            overall_fn += fn

            print(
                f"{class_name:20s}"
                f"{tp:8d}"
                f"{fp:8d}"
                f"{fn:8d}"
                f"{precision * 100:13.2f}%"
                f"{recall * 100:11.2f}%"
                f"{f1 * 100:9.2f}%"
            )

            confidence_results[class_name] = {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        overall_precision, overall_recall, overall_f1 = (
            precision_recall_f1(
                overall_tp,
                overall_fp,
                overall_fn,
            )
        )

        print("-" * 78)

        print(
            f"{'OVERALL':20s}"
            f"{overall_tp:8d}"
            f"{overall_fp:8d}"
            f"{overall_fn:8d}"
            f"{overall_precision * 100:13.2f}%"
            f"{overall_recall * 100:11.2f}%"
            f"{overall_f1 * 100:9.2f}%"
        )

        results_json["metrics"][
            f"conf_{confidence:.2f}"
        ] = {
            "classes": confidence_results,
            "overall": {
                "TP": overall_tp,
                "FP": overall_fp,
                "FN": overall_fn,
                "precision": overall_precision,
                "recall": overall_recall,
                "f1": overall_f1,
            },
        }

    # --------------------------------------------------------
    # CONFUSION ANALYSIS
    # --------------------------------------------------------

    confusion_confidence = 0.05

    print()
    print("=" * 90)
    print(
        f"CLASS CONFUSION ANALYSIS "
        f"CONF={confusion_confidence:.2f} "
        f"IoU={IOU_THRESHOLD:.2f}"
    )
    print("=" * 90)

    confusion = np.zeros(
        (len(CLASSES), len(CLASSES)),
        dtype=int,
    )

    for camera_name, image_path, label_path in all_images:

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        remap = get_remap(camera_name)

        gt_labels = load_yolo_labels(
            label_path,
            width,
            height,
            remap,
        )

        predictions = get_predictions(
            model,
            image,
            confusion_confidence,
        )

        image_confusion = build_confusion_matrix(
            gt_labels,
            predictions,
            IOU_THRESHOLD,
        )

        confusion += image_confusion

    header = "GT \\ Pred".ljust(20)

    for class_name in CLASSES:
        header += f"{class_name[:15]:>18s}"

    print(header)
    print("-" * len(header))

    for gt_index, gt_name in enumerate(CLASSES):

        row = f"{gt_name:20s}"

        for pred_index in range(len(CLASSES)):
            row += f"{confusion[gt_index][pred_index]:18d}"

        print(row)

    print()
    print("Non-zero confusions:")
    print("-" * 60)

    confusion_entries = []

    for gt_index, gt_name in enumerate(CLASSES):

        for pred_index, pred_name in enumerate(CLASSES):

            if (
                gt_index != pred_index
                and confusion[gt_index][pred_index] > 0
            ):

                count = int(
                    confusion[gt_index][pred_index]
                )

                print(
                    f"{gt_name:20s} -> "
                    f"{pred_name:20s} "
                    f"{count}"
                )

                confusion_entries.append(
                    {
                        "ground_truth": gt_name,
                        "predicted": pred_name,
                        "count": count,
                    }
                )

    results_json["confusion"] = {
        "confidence": confusion_confidence,
        "iou_threshold": IOU_THRESHOLD,
        "matrix": confusion.tolist(),
        "non_zero_confusions": confusion_entries,
    }

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = (
        ROOT
        / "logs"
        / "best_pt_corrected_evaluation.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results_json,
            f,
            indent=2,
        )

    print()
    print("=" * 78)
    print("RESULT SAVED")
    print("=" * 78)
    print(output_path)
    print("=" * 78)


if __name__ == "__main__":
    main()

