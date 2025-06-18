import argparse
import cv2
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO

if __name__ == '__main__':
    # Load YOLOv8 (person detection)
    yolo_model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy

    parser = argparse.ArgumentParser(description='Live Depth Detection with Webcam')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print("DEVICE: ", DEVICE)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    cap = cv2.VideoCapture(0)  # Use default webcam
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    print("Starting webcam feed. Press 'q' to quit.")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_height, frame_width = raw_frame.shape[:2]
        output_width = frame_width if args.pred_only else frame_width * 2 + margin_width

        # YOLO detection
        results = yolo_model(raw_frame)[0]
        human_boxes = []
        if results.boxes is not None:
            human_boxes = [box for box in results.boxes if int(box.cls[0]) == 0]  # class 0 = person

        with torch.no_grad():
            depth_tensor = depth_anything.infer_image(raw_frame, args.input_size)
        depth = depth_tensor.squeeze()

        norm_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        norm_depth = norm_depth.astype(np.uint8)

        if args.grayscale:
            depth_vis = np.repeat(norm_depth[..., np.newaxis], 3, axis=-1)
        else:
            depth_vis = (cmap(norm_depth)[:, :, :3] * 255).astype(np.uint8)[:, :, ::-1]

        if depth_vis.shape[:2] != raw_frame.shape[:2]:
            depth_vis = cv2.resize(depth_vis, (frame_width, frame_height))

        for box in human_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, depth.shape[1] - 1), min(y2, depth.shape[0] - 1)

            box_depth = depth[y1:y2, x1:x2]
            avg_depth = np.mean(box_depth)

            cv2.rectangle(raw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{avg_depth:.2f} {'CAUTION!!!!!' if avg_depth > 3 else ''}"
            cv2.putText(raw_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if args.pred_only:
            display_frame = depth_vis
        else:
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            display_frame = cv2.hconcat([raw_frame, split_region, depth_vis])

        cv2.imshow("Live Depth + YOLO", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()