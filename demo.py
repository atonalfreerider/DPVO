import os
import json
from multiprocessing import Process, Queue
from tqdm import tqdm

import cv2
import torch

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.stream import image_stream, video_stream
from dpvo.utils import Timer

SKIP = 0

@torch.no_grad()
def run(cfg, video_path, calib, stride=1, skip=0, timeit=False):
    visual_odometry = None
    queue = Queue(maxsize=8)

    # Start the reader process first to get total frame count
    if os.path.isdir(video_path):
        # For image directory, count number of images
        n_frames = len([f for f in os.listdir(video_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        reader = Process(target=image_stream, args=(queue, video_path, calib, stride, skip))
    else:
        # For video file, get frame count using cv2
        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        reader = Process(target=video_stream, args=(queue, video_path, calib, stride, skip))

    reader.start()

    # Calculate actual number of frames after stride and skip
    n_frames = (n_frames - skip) // stride

    # Initialize progress bar
    pbar = tqdm(total=n_frames, desc="Processing Visual Odometry", unit="frames")

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0:
            pbar.close()
            break

        image = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if visual_odometry is None:
            _, H, W = image.shape
            visual_odometry = DPVO(cfg, 'models/dpvo.pth', ht=H, wd=W)

        with Timer("VO", enabled=timeit):
            visual_odometry(t, image, intrinsics)

        # Update progress bar
        pbar.update(1)

    reader.join()
    return visual_odometry.terminate(), (*intrinsics, H, W)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--calib', type=str, default='calib/iphone.txt')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--skip', type=int, default=0)
    parser.add_argument('--config', default="config/default.yaml")
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--opts', nargs='+', default=[])

    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    print("Running with config...")
    print(cfg)

    (poses, tstamps), calib = run(cfg, args.video_path, args.calib, args.stride, args.skip, args.timeit)

    result = []

    # Iterate through all timestamps and collect the data
    # The assumption is that the camera starts by pointing forward. Every frame is an absolute position and rotation
    for i, timestamp in enumerate(tstamps):
        # Append absolute position and quaternion data to the result list with 10 decimal precision
        result.append([
            round(float(poses[i][0]), 10),  # x
            round(float(poses[i][1]), 10),  # y
            round(float(poses[i][2]), 10),  # z
            round(float(poses[i][3]), 10),  # q_x
            round(float(poses[i][4]), 10),  # q_y
            round(float(poses[i][5]), 10),  # q_z
            round(float(poses[i][6]), 10)  # q_w
        ])

    # Create output JSON path by replacing .mp4 with _vo.json
    json_path = args.video_path.rsplit('.', 1)[0] + '.vo.json'

    # Convert the parsed data to JSON format
    with open(json_path, 'w') as outfile:
        json.dump(result, outfile, indent=4)
    print(f"Results saved to {json_path}")
