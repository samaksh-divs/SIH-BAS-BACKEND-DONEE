from pathlib import Path
from ultralytics import YOLOWorld
import torch
from PIL import Image


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


DATASET_MAPPINGS = {
    "correct_01_02_03": {
        0: 0,
        1: 1,
        2: 3,
        3: 4,
        4: 5,
        5: 2,
    },
    "correct_13_14_15": {
        0: 0,
        1: 1,
        2: 3,
        3: 4,
        4: 5,
        5: 2,
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


DATASETS = {
    "front_C11": ("correct_13_14_15", "procedure"),
    "top_C01": ("correct_01_02_03", "procedure"),
    "front_E07": ("correct_01_02_03", "candidate"),
    "top_E01": ("error", "candidate_top_E01"),
}


def iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def yolo_to_xyxy(cx, cy, w, h, image_w, image_h):

    return [
        (cx - w / 2) * image_w,
        (cy - h / 2) * image_h,
        (cx + w / 2) * image_w,
        (cy + h / 2) * image_h,
    ]


def read_labels(label_path, dataset_name, image_w, image_h):

    mapping = DATASET_MAPPINGS[dataset_name]

    labels = []

    if not label_path.exists():
        return labels

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        source_class = int(parts[0])

        if source_class not in mapping:
            continue

        canonical_class = mapping[source_class]

        cx, cy, w, h = map(
            float,
            parts[1:]
        )

        labels.append({
            "class_id": canonical_class,
            "box": yolo_to_xyxy(
                cx,
                cy,
                w,
                h,
                image_w,
                image_h,
            ),
        })

    return labels


def get_label_path(
    dataset_name,
    mode,
    image_name,
    frame_number,
):

    base = (
        ANN_ROOT /
        dataset_name /
        "labels" /
        "train"
    )

    if mode == "candidate":
        return (
            base /
            f"candidate_{frame_number:02d}.txt"
        )

    if mode == "candidate_top_E01":
        return (
            base /
            "top_E01" /
            f"candidate_{frame_number:02d}.txt"
        )

    return (
        base /
        image_name.replace(
            ".jpg",
            ".txt"
        )
    )


def collect_images():

    items = []

    for camera, (dataset, mode) in DATASETS.items():

        folder = IMAGE_ROOT / camera

        images = sorted(
            folder.glob("*.jpg")
        )

        for index, image in enumerate(
            images,
            start=1
        ):

            label = get_label_path(
                dataset,
                mode,
                image.name,
                index,
            )

            items.append({
                "camera": camera,
                "dataset": dataset,
                "image": image,
                "label": label,
            })

    return items


def evaluate(
    model,
    items,
    confidence,
    iou_threshold,
):

    totals = {
        cls: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for cls in CLASSES
    }

    for item in items:

        image = item["image"]
        label = item["label"]
        dataset = item["dataset"]

        with Image.open(image) as im:
            width, height = im.size

        ground_truth = read_labels(
            label,
            dataset,
            width,
            height,
        )

        result = model.predict(
            str(image),
            device=(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            ),
            conf=confidence,
            iou=0.70,
            verbose=False,
        )[0]

        predictions = []

        if result.boxes is not None:

            for box, cls_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):

                cls_id = int(cls_id)

                if 0 <= cls_id < len(CLASSES):

                    predictions.append({
                        "class_id": cls_id,
                        "box": box,
                    })

        for class_id, class_name in enumerate(
            CLASSES
        ):

            gt = [
                x for x in ground_truth
                if x["class_id"] == class_id
            ]

            pred = [
                x for x in predictions
                if x["class_id"] == class_id
            ]

            matched_gt = set()
            matched_pred = set()

            candidates = []

            for pi, prediction in enumerate(pred):

                for gi, truth in enumerate(gt):

                    overlap = iou(
                        prediction["box"],
                        truth["box"],
                    )

                    if overlap >= iou_threshold:

                        candidates.append(
                            (
                                overlap,
                                pi,
                                gi,
                            )
                        )

            candidates.sort(
                reverse=True
            )

            for overlap, pi, gi in candidates:

                if pi in matched_pred:
                    continue

                if gi in matched_gt:
                    continue

                matched_pred.add(pi)
                matched_gt.add(gi)

            totals[class_name]["tp"] += len(
                matched_pred
            )

            totals[class_name]["fp"] += (
                len(pred) -
                len(matched_pred)
            )

            totals[class_name]["fn"] += (
                len(gt) -
                len(matched_gt)
            )

    return totals


def print_results(
    totals,
    confidence,
    iou_threshold,
):

    print()
    print(
        f"CONF={confidence:.2f}  "
        f"IoU={iou_threshold:.2f}"
    )

    print(
        f"{'Class':<20}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    print("-" * 58)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for class_name in CLASSES:

        values = totals[class_name]

        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0
        )

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if precision + recall
            else 0
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        print(
            f"{class_name:<20}"
            f"{precision * 100:>11.2f}%"
            f"{recall * 100:>11.2f}%"
            f"{f1 * 100:>11.2f}%"
        )

    precision = (
        total_tp /
        (total_tp + total_fp)
        if total_tp + total_fp
        else 0
    )

    recall = (
        total_tp /
        (total_tp + total_fn)
        if total_tp + total_fn
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall
        else 0
    )

    print("-" * 58)

    print(
        f"{'OVERALL':<20}"
        f"{precision * 100:>11.2f}%"
        f"{recall * 100:>11.2f}%"
        f"{f1 * 100:>11.2f}%"
    )


def main():

    print("=" * 70)
    print("BAS YOLO-WORLD IoU SENSITIVITY TEST")
    print("=" * 70)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    model = YOLOWorld(
        str(
            ROOT /
            "models" /
            "yolov8s-world.pt"
        )
    )

    model.set_classes(
        WORLD_CLASSES
    )

    items = collect_images()

    print(
        "Images evaluated:",
        len(items)
    )

    # Keep confidence fixed so we isolate
    # the effect of IoU evaluation.
    confidence = 0.05

    for iou_threshold in [
        0.25,
        0.50,
        0.75,
    ]:

        totals = evaluate(
            model,
            items,
            confidence,
            iou_threshold,
        )

        print_results(
            totals,
            confidence,
            iou_threshold,
        )


if __name__ == "__main__":
    main()