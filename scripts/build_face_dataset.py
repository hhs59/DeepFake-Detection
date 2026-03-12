import os
from detect_faces import process_images


FRAME_ROOT = "data/frames"
FACE_ROOT = "data/faces"


def main():

    classes = ["real", "fake"]

    for cls in classes:

        input_dir = os.path.join(FRAME_ROOT, cls)

        output_dir = os.path.join(FACE_ROOT, cls)

        process_images(input_dir, output_dir)


if __name__ == "__main__":
    main()