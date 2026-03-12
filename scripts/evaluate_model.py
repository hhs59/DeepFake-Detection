import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from src.datasets.build_dataloader import build_dataloader
from src.models.deepfake_model import DeepfakeModel
from src.config import *

# load dataloader
_, _, test_loader = build_dataloader(
    DATA_DIR,
    batch_size=BATCH_SIZE
)

# load model
model = DeepfakeModel().to(DEVICE)

model.load_state_dict(
    torch.load("checkpoints/best_model.pth")
)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images, labels in tqdm(test_loader):

        images = images.to(DEVICE)

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())


# classification report
print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=["real","fake"]
    )
)

# confusion matrix
print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        all_labels,
        all_preds
    )
)