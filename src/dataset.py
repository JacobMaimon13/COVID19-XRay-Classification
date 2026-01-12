import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# Transforms
basic_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

augmented_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor()
])

class ChestXrayDataset(Dataset):
    def __init__(self, df, transform=None, augment_covid=False):
        self.df = df
        self.transform = transform if transform else basic_transform
        self.augment_covid = augment_covid

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['filepath']).convert('RGB')
        label = int(row['label'])
        
        # Apply specific augmentation only for COVID class if requested
        if self.augment_covid and label == 2:
            image = augmented_transform(image)
        elif self.transform:
            image = self.transform(image)
            
        return image, label
