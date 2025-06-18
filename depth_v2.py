import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2
from ultralytics import YOLO

if __name__ == '__main__':

    # Load YOLOv8 (person detection)
    yolo_model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy

    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
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

    # Collect video files
    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)

    if not filenames:
        raise ValueError("No valid video files found.")

    os.makedirs(args.outdir, exist_ok=True)

    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for k, filename in enumerate(filenames):
        print(f'\nVideo {k+1}/{len(filenames)}: {filename}')
        raw_video = cv2.VideoCapture(filename)

        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        frame_count = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))

        if args.pred_only:
            output_width = frame_width
        else:
            output_width = frame_width * 2 + margin_width

        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))

        with tqdm(total=frame_count, desc="Processing frames") as pbar:
            while raw_video.isOpened():
                ret, raw_frame = raw_video.read()
                if not ret:
                    break

                # YOLO detection
                results = yolo_model(raw_frame)[0]
                human_boxes = []
                if results.boxes is not None:
                    human_boxes = [box for box in results.boxes if int(box.cls[0]) == 0]  # class 0 = person, 2 = car

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
                    label = f"{avg_depth:.2f} {'CAUTION!!!!!' if avg_depth > 4 else ''}"
                    cv2.putText(raw_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if args.pred_only:
                    out.write(depth_vis)
                else:
                    split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                    combined_frame = cv2.hconcat([raw_frame, split_region, depth_vis])
                    out.write(combined_frame)

                pbar.update(1)

        raw_video.release()
        out.release()
