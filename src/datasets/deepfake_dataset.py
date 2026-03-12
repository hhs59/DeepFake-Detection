from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random

class DeepfakeDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.images = []
        self.labels = []

        for label, cls in enumerate(["real", "fake"]):

            for img_path in (self.root_dir / cls).glob("*.jpg"):

                self.images.append(img_path)
                self.labels.append(label)

        # 🔥 SHUFFLE DATASET
        data = list(zip(self.images, self.labels))
        random.shuffle(data)

        self.images, self.labels = zip(*data)

        self.images = list(self.images)
        self.labels = list(self.labels)

        #self.images = self.images[:10000]
        #self.labels = self.labels[:10000]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label