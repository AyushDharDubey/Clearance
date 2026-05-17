import cv2
import csv
import torch
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

# ============================================================
#  CONFIGURATION
# ============================================================

# Your trained model
MODEL_PATH     = r"D:\erickshaw_training_v2\train\weights\best.pt"

# Input video — change this to any of your 30 videos
INPUT_VIDEO    = r"D:\video_data_analysis\vbox_videos\d14.mp4"

# Output folder
OUTPUT_DIR     = r"D:\inference_results"

# VideoVBOX HD crop settings (1920x1080)
# Top half only — discard green telemetry bottom half
CROP_TOP       = 0
CROP_BOTTOM    = 540   # discard everything below this
SPLIT_X        = 960   # left camera: 0-960, right camera: 960-1920

# Detection settings
CONF_THRESHOLD = 0.40  # minimum confidence to show a detection
IMAGE_SIZE     = 640

# Cross-class IoU suppression
# If a 'person' box overlaps a vehicle box by more than this threshold,
# the person is suppressed (assumed to be a rider, not a standalone pedestrian)
# 0.25 = suppress if 25% or more of the person box overlaps a vehicle box
PERSON_VEHICLE_IOU_THRESHOLD = 0.25
VEHICLE_CLASSES = {'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'E-rickshaw'}

# COCO class IDs we care about (from standard YOLOv8 COCO classes)
# E-rickshaw is handled by our fine-tuned model (class 0 in our model)
# For COCO classes, we filter to only relevant ones:
COCO_CLASSES_KEEP = {
    0:  'person',
    1:  'bicycle',
    2:  'car',
    3:  'motorcycle',
    5:  'bus',
    7:  'truck',
}

# Our fine-tuned model only has 1 class: E-rickshaw
# So all detections from our model are class 0 = E-rickshaw

# Box colors per class (BGR format)
CLASS_COLORS = {
    'E-rickshaw' : (0,   255, 0),    # green
    'person'     : (255, 178, 50),   # blue
    'bicycle'    : (0,   165, 255),  # orange
    'car'        : (0,   0,   255),  # red
    'motorcycle' : (255, 0,   255),  # magenta
    'bus'        : (255, 255, 0),    # cyan
    'truck'      : (128, 0,   128),  # purple
}

# ============================================================
#  SETUP
# ============================================================

if __name__ == '__main__':

    print("=" * 65)
    print("VIDEO INFERENCE — 7-Class Traffic Detection")
    print("=" * 65)

    # Verify GPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"  Device    : {'GPU - ' + torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")
    print(f"  Model     : {MODEL_PATH}")
    print(f"  Video     : {INPUT_VIDEO}")

    if not Path(INPUT_VIDEO).exists():
        print(f"\n[ERROR] Video not found: {INPUT_VIDEO}")
        print("  Update INPUT_VIDEO path at the top of this script.")
        exit(1)

    # Load our fine-tuned model (detects E-rickshaw)
    print("\n  Loading fine-tuned e-rickshaw model...")
    erick_model = YOLO(MODEL_PATH)

    # Load standard COCO model (detects all other classes)
    print("  Loading COCO model for other vehicle classes...")
    coco_model  = YOLO("yolov8m.pt")
    print("  Both models loaded!")

    # Output paths
    output_dir   = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_stem   = Path(INPUT_VIDEO).stem
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_left_path  = output_dir / f"{video_stem}_left_{timestamp}.mp4"
    out_right_path = output_dir / f"{video_stem}_right_{timestamp}.mp4"
    csv_path       = output_dir / f"{video_stem}_detections_{timestamp}.csv"

    # ── Open video ───────────────────────────────────────────
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {INPUT_VIDEO}")
        exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / video_fps

    print(f"\n  Resolution  : {width}x{height}")
    print(f"  FPS         : {video_fps:.1f}")
    print(f"  Duration    : {duration_sec/60:.1f} min ({total_frames:,} frames)")
    print(f"  Crop        : top {CROP_BOTTOM}px only (discarding telemetry)")
    print(f"  Output size : {SPLIT_X}x{CROP_BOTTOM} per camera")
    print("=" * 65)

    # ── Video writers ────────────────────────────────────────
    fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
    out_left   = cv2.VideoWriter(str(out_left_path),  fourcc, video_fps,
                                  (SPLIT_X, CROP_BOTTOM))
    out_right  = cv2.VideoWriter(str(out_right_path), fourcc, video_fps,
                                  (SPLIT_X, CROP_BOTTOM))

    # ── CSV writer ───────────────────────────────────────────
    csv_file   = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'frame', 'timestamp_sec', 'camera',
        'class', 'confidence',
        'x_center', 'y_center', 'width', 'height',
        'x1', 'y1', 'x2', 'y2'
    ])

    # ── Stats ────────────────────────────────────────────────
    from collections import defaultdict
    total_detections = defaultdict(int)
    frame_idx        = 0

    print(f"\n  Processing video... (press Ctrl+C to stop early)\n")

    def compute_iou(boxA, boxB):
        """
        Compute Intersection over Union between two boxes.
        Each box is (x1, y1, x2, y2).
        Returns IoU value between 0 and 1.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter   = inter_w * inter_h

        if inter == 0:
            return 0.0

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - inter

        return inter / union if union > 0 else 0.0


    def suppress_rider_persons(all_dets):
        """
        Given a list of detection dicts, remove any 'person' detection
        whose bounding box overlaps significantly with a vehicle box.
        This prevents drawing a separate person box on every rider.

        Input:  list of dicts with keys: label, conf, x1, y1, x2, y2
        Output: filtered list with rider-persons removed
        """
        vehicle_boxes = [
            (d['x1'], d['y1'], d['x2'], d['y2'])
            for d in all_dets
            if d['label'] in VEHICLE_CLASSES
        ]

        if not vehicle_boxes:
            return all_dets  # no vehicles → keep all persons

        filtered = []
        suppressed_count = 0

        for det in all_dets:
            if det['label'] == 'person':
                person_box = (det['x1'], det['y1'], det['x2'], det['y2'])
                # Check overlap with every vehicle box
                is_rider = any(
                    compute_iou(person_box, vbox) >= PERSON_VEHICLE_IOU_THRESHOLD
                    for vbox in vehicle_boxes
                )
                if is_rider:
                    suppressed_count += 1
                    continue  # skip this person — they're a rider
            filtered.append(det)

        return filtered, suppressed_count


    def draw_box(frame, x1, y1, x2, y2, label, conf, color):
        """Draw a bounding box with label on frame."""
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text     = f"{label} {conf:.2f}"
        txt_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        # Background rectangle for text
        cv2.rectangle(frame,
                      (x1, y1 - txt_size[1] - 6),
                      (x1 + txt_size[0] + 2, y1),
                      color, -1)
        cv2.putText(frame, text,
                    (x1 + 1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        return frame


    def run_detection(crop, frame_idx, timestamp, camera_name):
        """
        Run both models on a crop.
        1. Collect all detections from both models as dicts
        2. Suppress person boxes that overlap with vehicle boxes (riders)
        3. Draw remaining detections and build CSV rows
        Returns (annotated_frame, csv_rows, suppressed_count)
        """
        annotated  = crop.copy()
        all_dets   = []   # list of dicts — collected before drawing

        # ── E-rickshaw model ──────────────────────────────
        er_results = erick_model(crop, imgsz=IMAGE_SIZE,
                                  conf=CONF_THRESHOLD, verbose=False)
        for result in er_results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                all_dets.append({
                    'label' : 'E-rickshaw',
                    'conf'  : float(box.conf[0]),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                })

        # ── COCO model (other vehicles + persons) ─────────
        co_results = coco_model(crop, imgsz=IMAGE_SIZE,
                                 conf=CONF_THRESHOLD, verbose=False)
        for result in co_results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in COCO_CLASSES_KEEP:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                all_dets.append({
                    'label' : COCO_CLASSES_KEEP[cls_id],
                    'conf'  : float(box.conf[0]),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                })

        # ── Suppress rider-persons ────────────────────────
        filtered_dets, n_suppressed = suppress_rider_persons(all_dets)
        total_detections['_suppressed'] += n_suppressed

        # ── Draw boxes and build CSV rows ─────────────────
        csv_rows = []
        for det in filtered_dets:
            label = det['label']
            conf  = det['conf']
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            color = CLASS_COLORS.get(label, (200, 200, 200))
            annotated = draw_box(annotated, x1, y1, x2, y2, label, conf, color)
            cx  = (x1 + x2) / 2 / crop.shape[1]
            cy  = (y1 + y2) / 2 / crop.shape[0]
            bw  = (x2 - x1) / crop.shape[1]
            bh  = (y2 - y1) / crop.shape[0]
            csv_rows.append([frame_idx, timestamp, camera_name,
                             label, f"{conf:.3f}",
                             f"{cx:.4f}", f"{cy:.4f}",
                             f"{bw:.4f}", f"{bh:.4f}",
                             x1, y1, x2, y2])
            total_detections[label] += 1

        return annotated, csv_rows


    # ── Main loop ────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx  += 1
            timestamp   = frame_idx / video_fps

            # Progress every 100 frames
            if frame_idx % 100 == 0:
                pct = frame_idx / total_frames * 100
                print(f"  Frame {frame_idx:>6,}/{total_frames:,} "
                      f"({pct:.1f}%)  |  "
                      f"E-rick: {total_detections['E-rickshaw']:,}  "
                      f"Car: {total_detections['car']:,}  "
                      f"Moto: {total_detections['motorcycle']:,}")

            # Crop top half only (discard green telemetry)
            top_half = frame[CROP_TOP:CROP_BOTTOM, 0:width]

            # Split into left and right camera
            left_crop  = top_half[0:CROP_BOTTOM, 0:SPLIT_X]
            right_crop = top_half[0:CROP_BOTTOM, SPLIT_X:width]

            # Run detection on both cameras
            left_annotated,  left_rows  = run_detection(
                left_crop,  frame_idx, timestamp, 'left')
            right_annotated, right_rows = run_detection(
                right_crop, frame_idx, timestamp, 'right')

            # Write to CSV
            for row in left_rows + right_rows:
                csv_writer.writerow(row)

            # Add frame number overlay
            for ann_frame in [left_annotated, right_annotated]:
                cv2.putText(ann_frame,
                            f"Frame: {frame_idx}  T: {timestamp:.1f}s",
                            (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

            # Write to output videos
            out_left.write(left_annotated)
            out_right.write(right_annotated)

    except KeyboardInterrupt:
        print("\n  [!] Stopped early by user")

    finally:
        cap.release()
        out_left.release()
        out_right.release()
        csv_file.close()

    # ── Final summary ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("INFERENCE COMPLETE")
    print("=" * 65)
    print(f"  Frames processed : {frame_idx:,} / {total_frames:,}")
    print(f"  Duration covered : {frame_idx/video_fps/60:.1f} minutes")
    print("\n  Total detections per class (after rider suppression):")
    print(f"  {'─'*40}")
    for cls_name, count in sorted(total_detections.items(),
                                   key=lambda x: -x[1]):
        if cls_name == '_suppressed':
            continue
        print(f"  {cls_name:<15} : {count:>8,}")
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':<15} : {sum(v for k,v in total_detections.items() if k != '_suppressed'):>8,}")
    print(f"  {'Riders suppressed':<15} : {total_detections['_suppressed']:>8,}  (person-on-vehicle boxes removed)")
    print(f"\n  Output files:")
    print(f"  Left video  : {out_left_path}")
    print(f"  Right video : {out_right_path}")
    print(f"  CSV log     : {csv_path}")
    print("=" * 65)