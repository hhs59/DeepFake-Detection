import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from src.models.deepfake_model import DeepfakeModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


model = DeepfakeModel().to(DEVICE)

model.load_state_dict(
    torch.load("checkpoints/best_model.pth")
)

model.eval()


def predict_frame(frame):

    img = Image.fromarray(frame)

    img = transform(img)

    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(img)

        prob = torch.softmax(output, dim=1)[0,1]

    return prob.item()


def detect_video(video_path):

    cap = cv2.VideoCapture(video_path)

    scores = []

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % 10 == 0:   # sample frames

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            score = predict_frame(frame)

            scores.append(score)

        frame_id += 1


    cap.release()

    video_score = np.mean(scores)

    if video_score > 0.5:
        label = "FAKE"
    else:
        label = "REAL"

    print("Deepfake probability:", video_score)
    print("Prediction:", label)


if __name__ == "__main__":

    detect_video("sample_video.mp4")