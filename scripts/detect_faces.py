import cv2
from pathlib import Path
from facenet_pytorch import MTCNN
from tqdm import tqdm


mtcnn = MTCNN(image_size=224)


def process_images(input_dir, output_dir):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(input_dir.glob("*.jpg"))

    for img_path in tqdm(images):

        img = cv2.imread(str(img_path))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face = mtcnn(img)

        if face is None:
            continue

        output_path = output_dir / img_path.name

        face = face.permute(1,2,0).numpy()

        face = (face * 255).astype("uint8")

        cv2.imwrite(str(output_path), face)