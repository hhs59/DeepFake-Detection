import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from src.datasets.build_dataloader import build_dataloader
from src.models.deepfake_model import DeepfakeModel
from src.utils.metrics import compute_metrics


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = "data"
BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-4

SAVE_DIR = Path("checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

writer = SummaryWriter("runs/deepfake")


def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total

    return total_loss / len(loader), acc


def validate(model, loader, criterion):

    model.eval()

    total_loss = 0

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():

        for images, labels in tqdm(loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            y_true.append(labels)
            y_pred.append(preds)
            y_prob.append(probs)

    metrics = compute_metrics(y_true, y_pred, y_prob)

    return total_loss / len(loader), metrics


def main():

    train_loader, val_loader, test_loader = build_dataloader(DATA_DIR, BATCH_SIZE)

    model = DeepfakeModel().to(DEVICE)

    print(
        "Trainable params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    patience = 5
    patience_counter = 0

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        # fine-tuning
        if epoch == 5:

            print("Unfreezing backbone")

            model.unfreeze_backbone()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, metrics = validate(
            model,
            val_loader,
            criterion
        )

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f}")

        print(f"\nVal Loss: {val_loss:.4f}")

        print("Validation metrics:")

        for k, v in metrics.items():

            print(f"{k}: {v:.4f}")

        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", metrics["accuracy"], epoch)

        writer.add_scalar("Val/F1", metrics["f1"], epoch)
        writer.add_scalar("Val/AUC", metrics["auc"], epoch)

        # save best model
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                SAVE_DIR / "best_model.pth"
            )

            print("\nBest model saved")

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print("\nEarly stopping triggered")

                break


if __name__ == "__main__":

    main()