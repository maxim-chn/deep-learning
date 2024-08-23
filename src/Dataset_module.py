import os
import cv2
import torch
from torchvision.transforms import transforms, InterpolationMode
from torch.utils.data import Dataset


class SRDataset(Dataset):
    """
    A dataset class for loading and transforming images for super-resolution tasks.
    """

    def __init__(self, dataset_path, limit=-1, _transforms=None, hr_sz=128, lr_sz=32):
        """
        Initializes the dataset object.
        :param dataset_path: Path to the directory containing images.
        :param limit: Maximum number of images to load (default is -1, which means all images).
        :param _transforms: Optional custom transformations; if None, defaults are applied.
        :param hr_sz: Target size for high-resolution images.
        :param lr_sz: Target size for low-resolution images.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.limit = limit if limit > 0 else None  # None means no limit

        # Define default transformations if none are provided
        if not _transforms:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),  # Converts images to PyTorch tensors
                transforms.RandomHorizontalFlip(0.5),  # Randomly flips images horizontally
                transforms.ColorJitter(brightness=0.5, contrast=1),  # Randomly alters brightness and contrast
                transforms.RandomAdjustSharpness(1.1, p=0.4),  # Randomly adjusts sharpness
                transforms.Normalize([0.5], [0.5])  # Normalizes the images
            ])
        else:
            self.transforms = _transforms

        self.hr_sz = transforms.Resize((hr_sz, hr_sz), interpolation=InterpolationMode.BICUBIC)
        self.lr_sz = transforms.Resize((lr_sz, lr_sz), interpolation=InterpolationMode.BICUBIC)

        # Load valid image file names
        self.images = [f for f in os.listdir(dataset_path) if f.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']]
        if self.limit:
            self.images = self.images[:self.limit]

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Retrieves an image by its index, applies transformations, and returns both its low-resolution and high-resolution versions.
        :param index: Index of the image to retrieve.
        :return: A tuple of (low-resolution image, high-resolution image).
        """
        img_path = os.path.join(self.dataset_path, self.images[index])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = self.transforms(image)  # Apply transformations
        lr_image = self.lr_sz(image)  # Resize to low-resolution
        hr_image = self.hr_sz(lr_image)  # Resize back to high-resolution
        return lr_image, hr_image
