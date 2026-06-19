import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class VideoProcessor:
    VEHICLE_CLASSES = {'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'erickshaw'}
    COCO_CLASSES_KEEP = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    
    def __init__(self, input_video, output_dir="detection_output", erick_model_path="best.pt", conf_threshold=0.40):
        self.input_path = Path(input_video)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.conf_threshold = conf_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Heuristics thresholds
        self.iou_threshold = 0.20
        self.ioa_threshold = 0.80
        self.centroid_dist_threshold = 30
        
        # Core Models Setup
        self.coco_model = YOLO("yolo26n.pt").to(self.device)
        self.erick_model = YOLO(erick_model_path).to(self.device)
        
        # Depth Estimation Setup
        self.depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(self.device).eval()
        self.depth_processor = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        
        self.total_detections = defaultdict(int)

    @staticmethod
    def distance_levit(median_depth: float) -> float:
        """Converts raw DPT LeViT depth into absolute meters using rational function regression."""
        # a3 = -0.00000165, a2 = 0.00617685, a1 = -7.76795188, a0 = 3457.55560786
        # return (a3 * (median_depth ** 3)) + (a2 * (median_depth ** 2)) + (a1 * median_depth) + a0

        a, b, c = 157994.966563, -372.880209, -13.939203
        return (a / (median_depth + b)) + c

    @staticmethod
    def compute_iou(box_a, box_b):
        x_a, y_a = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
        x_b, y_b = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        inter = max(0, x_b - x_a) * max(0, y_b - y_a)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        return inter / (area_a + area_b - inter)

    @staticmethod
    def compute_ioa(box_a, box_b):
        x_a, y_a = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
        x_b, y_b = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        inter = max(0, x_b - x_a) * max(0, y_b - y_a)
        if inter == 0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        area_min = min(area_a, area_b)
        return inter / area_min if area_min > 0 else 0.0

    def is_bike_rider(self, pbox, vbox):
        if self.compute_iou(pbox, vbox) >= self.iou_threshold or self.compute_ioa(pbox, vbox) >= self.ioa_threshold:
            return True
        cx_p, cy_p = (pbox[0] + pbox[2]) / 2, (pbox[1] + pbox[3]) / 2
        cx_v, cy_v = (vbox[0] + vbox[2]) / 2, (vbox[1] + vbox[3]) / 2
        return ((cx_p - cx_v) ** 2 + (cy_p - cy_v) ** 2) ** 0.5 < self.centroid_dist_threshold

    def suppress_detections(self, all_dets):
        vehicle_boxes = [(d['x1'], d['y1'], d['x2'], d['y2']) for d in all_dets if d['label'] in self.VEHICLE_CLASSES]
        erick_boxes = [(d['x1'], d['y1'], d['x2'], d['y2']) for d in all_dets if d['label'] == 'erickshaw']

        if not vehicle_boxes:
            return all_dets, 0

        filtered = []
        suppressed_count = 0

        for det in all_dets:
            box = (det['x1'], det['y1'], det['x2'], det['y2'])
            if det['label'] == 'person':
                if any(self.is_bike_rider(box, vbox) for vbox in vehicle_boxes):
                    suppressed_count += 1
                    continue
            elif det['label'] in self.VEHICLE_CLASSES and det['label'] != 'erickshaw':
                if any(self.compute_iou(box, ebox) >= self.iou_threshold for ebox in erick_boxes):
                    suppressed_count += 1
                    continue
            filtered.append(det)

        return filtered, suppressed_count

    def _process_yolo_results(self, results, class_mapping, depth_map, is_custom_erick=False):
        detections = []
        if results.boxes.id is None:
            return detections

        boxes = results.boxes
        track_ids = results.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            cls_id = int(box.cls[0])
            if not is_custom_erick and cls_id not in class_mapping:
                continue

            label = 'erickshaw' if is_custom_erick else class_mapping[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            box_depth = depth_map[y1:y2, x1:x2]
            if box_depth.size == 0:
                continue
                
            median_depth_val = np.median(box_depth)
            dist = self.distance_levit(median_depth_val)

            detections.append({
                'tracking_id': track_id,
                'object': f"{label}{track_id}",
                'label': label,
                'conf': float(box.conf[0]),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'distance': dist
            })
        return detections

    def draw_box(self, frame, det, color):
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{det['object']} {det['conf']:.2f} {det['distance']:.1f}cm"
        txt_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        cv2.rectangle(frame, (x1, y1 - txt_size[1] - 6), (x1 + txt_size[0] + 2, y1), color, -1)
        cv2.putText(frame, text, (x1 + 1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def run_tracking(self, frame, frame_idx, timestamp_sec, camera_name):
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Depth Map Generation
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.depth_processor(rgb_frame).to(self.device)
        with torch.no_grad():
            predicted_depth = self.depth_model(inputs)
            depth_map = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()

        # Models Predictions Unified Pipeline
        er_res = self.erick_model.track(frame, conf=self.conf_threshold, verbose=False, persist=True, agnostic_nms=True)[0]
        co_res = self.coco_model.track(frame, conf=self.conf_threshold, verbose=False, persist=True, agnostic_nms=True)[0]

        all_dets = []
        all_dets.extend(self._process_yolo_results(er_res, None, depth_map, is_custom_erick=True))
        all_dets.extend(self._process_yolo_results(co_res, self.COCO_CLASSES_KEEP, depth_map))

        filtered_dets, n_suppressed = self.suppress_detections(all_dets)
        self.total_detections['_suppressed'] += n_suppressed

        csv_rows = []
        for det in filtered_dets:
            dist = det['distance']
            if not (0 <= dist <= 500):
                continue

            random.seed(det['tracking_id'])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            annotated = self.draw_box(annotated, det, color)

            # Slicing & normalization calculations
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            cx, cy = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            
            csv_rows.append([
                frame_idx, timestamp_sec, camera_name, det['label'], det['object'],
                f"{det['conf']:.3f}", f"{cx:.4f}", f"{cy:.4f}", f"{bw:.4f}", f"{bh:.4f}",
                x1, y1, x2, y2, dist
            ])
            self.total_detections[det['label']] += 1

        return annotated, csv_rows

    def process_video(self):
        cap = cv2.VideoCapture(str(self.input_path))
        if not cap.isOpened():
            print(f"Error: Could not open video file {self.input_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"{self.input_path.stem}_{timestamp}.mp4"
        csv_path = self.output_dir / f"{self.input_path.stem}_{timestamp}.csv"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Managed stream execution pipeline
        with open(csv_path, 'w', newline='') as csv_file, \
             VideoWriterContext(str(out_path), fourcc, video_fps, (width, height)) as out:
            
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                'frame', 'timestamp_sec', 'camera', 'class', 'object', 'confidence',
                'x_center', 'y_center', 'width', 'height', 'x1', 'y1', 'x2', 'y2', 'distance'
            ])

            frame_idx = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    timestamp_sec = frame_idx / video_fps

                    ann_frame, rows = self.run_tracking(frame, frame_idx, timestamp_sec, 'camera')
                    
                    for row in rows:
                        csv_writer.writerow(row)

                    cv2.putText(ann_frame, f"Frame: {frame_idx} T: {timestamp_sec:.1f}s", (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    out.write(ann_frame)

                    pct = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                    print(f"\rProgress: {pct:.1f}%", end="")
                    
                print("\nProcessing complete!")
            except KeyboardInterrupt:
                print("\nProcessing interrupted by user.")
            finally:
                cap.release()
                cv2.destroyAllWindows()


class VideoWriterContext:
    """Helper context manager safely releasing OpenCV's VideoWriter component."""
    def __init__(self, *args, **kwargs):
        self.writer = cv2.VideoWriter(*args, **kwargs)
    def __enter__(self):
        return self.writer
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track objects and measure depth using YOLO and MiDaS")
    parser.add_argument("input", type=str, nargs="?", default="./videos/d1_right.mp4",
                        help="Path to the input video file")
    args = parser.parse_args()

    
    processor = VideoProcessor(input_video=args.input)
    processor.process_video()