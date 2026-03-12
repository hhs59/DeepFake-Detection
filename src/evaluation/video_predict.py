import torch
from pathlib import Path
from collections import defaultdict
from PIL import Image
from torchvision import transforms
from src.models.deepfake_model import DeepfakeModel

device = "cuda" if torch.cuda.is_available() else "mps"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

model = DeepfakeModel().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.eval()

faces_dir = Path("data/test")

videos = defaultdict(list)

for cls in ["real", "fake"]:
    for img in (faces_dir / cls).glob("*.jpg"):
        vid = "_".join(img.stem.split("_")[:2])
        videos[vid].append(img)

correct = 0
total = 0
MAX_VIDEOS = 1000

for vid, frames in list(videos.items())[:MAX_VIDEOS]:

    scores = []

    for frame in frames:

        img = Image.open(frame).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img)
            prob_fake = torch.softmax(logits, dim=1)[0,1].item()

        scores.append(prob_fake)

    video_score = sum(scores) / len(scores)

    pred = 1 if video_score > 0.5 else 0
    true = 1 if "fake" in str(frames[0]) else 0

    if pred == true:
        correct += 1

    total += 1

print("Total videos:", total)
print("Video Accuracy:", correct / total)
