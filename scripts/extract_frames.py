import cv2
import os
from pathlib import Path
from tqdm import tqdm


def extract_frames(video_path, output_dir, frame_skip=5):

    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_id = 0
    saved_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % frame_skip == 0:

            filename = output_dir / f"{video_path.stem}_{saved_id}.jpg"

            cv2.imwrite(str(filename), frame)

            saved_id += 1

        frame_id += 1

    cap.release()

    return saved_id


def process_folder(video_folder, output_folder, frame_skip=5):

    video_folder = Path(video_folder)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    videos = list(video_folder.glob("*.mp4"))

    for video in tqdm(videos):

        extract_frames(video, output_folder, frame_skip)