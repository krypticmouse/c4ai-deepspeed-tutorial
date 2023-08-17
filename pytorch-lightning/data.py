import pytorch_lightning as pl
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

class Data(pl.LightningDataModule):
    def prepare_data(self):
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
      
        self.train_data = datasets.MNIST('', train=True, download=True, transform=transform)
        self.val_data = datasets.MNIST('', train=False, download=True, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size= 32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size= 32, shuffle=True)