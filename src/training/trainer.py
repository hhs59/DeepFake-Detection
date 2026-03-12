import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.utils.metrics import accuracy


class Trainer:

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = SummaryWriter("runs/deepfake")
        self.best_acc = 0
    def train_epoch(self, epoch):

        self.model.train()

        total_loss = 0

        for step, (images, labels) in enumerate(self.train_loader):

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

            self.writer.add_scalar(
                "Train/Loss",
                loss.item(),
                epoch * len(self.train_loader) + step
            )

        return total_loss / len(self.train_loader)


    def validate(self, epoch):

        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in self.val_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                preds = torch.argmax(outputs, dim=1)

                correct += (preds == labels).sum().item()

                total += labels.size(0)

        acc = correct / total

        self.writer.add_scalar("Val/Accuracy", acc, epoch)

        return acc