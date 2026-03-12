import random
import shutil
from pathlib import Path
from collections import defaultdict

SOURCE = Path("data/faces")
DEST = Path("data")

SPLIT = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

def split_class(cls):

    images = list((SOURCE / cls).glob("*.jpg"))

    # group frames by video id
    videos = defaultdict(list)

    for img in images:
        video_id = img.stem.split("_")[0]
        videos[video_id].append(img)

    video_ids = list(videos.keys())
    random.shuffle(video_ids)

    n = len(video_ids)

    train_end = int(n * SPLIT["train"])
    val_end = train_end + int(n * SPLIT["val"])

    split_videos = {
        "train": video_ids[:train_end],
        "val": video_ids[train_end:val_end],
        "test": video_ids[val_end:]
    }

    for split, vids in split_videos.items():

        folder = DEST / split / cls
        folder.mkdir(parents=True, exist_ok=True)

        for vid in vids:
            for img in videos[vid]:
                shutil.copy(img, folder / img.name)

for cls in ["real", "fake"]:
    split_class(cls)
