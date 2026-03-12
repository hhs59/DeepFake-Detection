import os
from extract_frames import process_folder


VIDEO_ROOT = "data/videos"

OUTPUT_ROOT = "data/frames"


def main():

    classes = ["real", "fake"]

    for cls in classes:

        video_folder = os.path.join(VIDEO_ROOT, cls)

        output_folder = os.path.join(OUTPUT_ROOT, cls)

        process_folder(video_folder, output_folder)


if __name__ == "__main__":
    main()