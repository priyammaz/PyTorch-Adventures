import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

### Set Transforms for Training and Testing ###
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

### Build Image Transforms ###
class ImageDataLoader:
    """
    Prepare transformation pipelines for Image Classification Datasets
    """
    def __init__(self, 
                 path_to_imagefolder,
                 train_folder = "train", 
                 test_folder = "validation", 
                 img_size: int = 224,
                 hflip: float = 0.5, 
                 vflip: float = 0.5,
                 scale: tuple = (0.8, 1),
                 color_jitter = 0.3,
                 num_workers: int = 16,
                 pin_memory: bool = True,
                 mean: tuple = IMAGENET_MEAN,
                 std: tuple = IMAGENET_STD, 
                 train_batch_size: int = None,
                 eval_batch_size: int = None,
                 grad_accum_steps: int = None):
        
        """
        Args:
            path_to_imagefolder: Path to image folder root
            train_folder: Folder with all training images
            test_folder: Folder with all testing images
            train_perc: If set, assume no pre-train/test split has occured and will split dataset randomly
            hflip: Probability of horizontal flip
            vflip: Probability of vertical flip
            scale: Range of values for rescaling in RandomResizedCrop
            color_jitter:
                - If single scalar given, duplicated across brightness, contrast, saturation
                - If tuple is passed, must be of length 3 indicating transform on brightness, contrast, saturation
            num_workers: Number of CPU cores to dedicate to dataloader
            pin_memory: Pre-load data to pass to GPU
            mean: Mean pixel value per channel (default 3 Channel Imagenet)
            batch_size: Number of samples grabbed in every iteration (can be sub-batch_size as well depending on what is passed)
            std: Standard Deviation of pixel values per channel (default 3 Channel Imagenet)
        """
        
        self.path_to_image_folder = path_to_imagefolder
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.img_size = img_size
        self.hflip = hflip
        self.vflip = vflip
        self.scale = scale
        self.color_jitter = color_jitter
        self.num_workers = num_workers
        self.mean = mean
        self.std = std
        self.pin_memory = pin_memory
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.grad_accum_steps = grad_accum_steps
        
        if self.grad_accum_steps is not None:
            assert self.train_batch_size % self.grad_accum_steps == 0, "Make sure the batch size is divisible by the number of gradient accumulation steps"
            self.train_batch_size = int(self.train_batch_size//self.grad_accum_steps)

    def prep_transforms(self):
        train_transforms_pipeline = []
        test_transforms_pipeline = []
        
        ### TRAIN TRANSFORMS ###
        ### Random Resize Crop ###
        if self.scale is not None:
            train_transforms_pipeline.append(transforms.RandomResizedCrop(size=(self.img_size, self.img_size),
                                                                          scale=self.scale))
        ### Random Horizontal Flip ###
        if self.hflip is not None:
            train_transforms_pipeline.append(transforms.RandomHorizontalFlip(p=self.hflip))
        ### Random Vertical Flip ###
        if self.vflip is not None:
            train_transforms_pipeline.append(transforms.RandomVerticalFlip(p=self.vflip))
        ### Random Color Jitter ###
        if self.color_jitter is not None:
            if isinstance(self.color_jitter, (list, tuple)):
                assert len(self.color_jitter) in (3,4)
            else:
                self.color_jitter = tuple([self.color_jitter for _ in range(3)])
            train_transforms_pipeline.append(transforms.ColorJitter(*self.color_jitter))
        ### Convert to Tensor ###
        train_transforms_pipeline.append(transforms.ToTensor())
        ### Normalize Image ###
        train_transforms_pipeline.append(transforms.Normalize(mean=torch.tensor(self.mean),
                                                              std=torch.tensor(self.std)))
        

        ### TEST TRANSFORMS ###
        test_transforms_pipeline.append(transforms.Resize((self.img_size, self.img_size)))
        test_transforms_pipeline.append(transforms.ToTensor())
        test_transforms_pipeline.append(transforms.Normalize(mean=torch.tensor(self.mean),
                                                              std=torch.tensor(self.std)))
        
        train_transforms_pipeline = transforms.Compose(train_transforms_pipeline)
        test_transforms_pipeline = transforms.Compose(test_transforms_pipeline)

        return train_transforms_pipeline, test_transforms_pipeline
    
    def prep_datasets(self):

        path_to_train_folder = os.path.join(self.path_to_image_folder, self.train_folder)
        path_to_test_folder = os.path.join(self.path_to_image_folder, self.test_folder)

        train_dataset = ImageFolder(path_to_train_folder)
        test_dataset = ImageFolder(path_to_test_folder)
            
        ### Add the Transformation Pipelines to Datasets ###
        train_transforms, test_transforms = self.prep_transforms()
        train_dataset.transform = train_transforms
        test_dataset.transform = test_transforms

        return train_dataset, test_dataset
    
    def prep_dataloaders(self):
       train_dataset, test_dataset = self.prep_datasets()
       train_dataloader = DataLoader(
           train_dataset, shuffle=True, batch_size=self.train_batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True
       )
       test_dataloader = DataLoader(
           test_dataset, shuffle=False, batch_size=self.eval_batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True
       )
       return train_dataloader, test_dataloader

if __name__ == "__main__":
    loader_pred = ImageDataLoader(path_to_imagefolder="/mnt/datadrive/data/ImageNet",
                                  train_batch_size=256, 
                                  eval_batch_size=256, 
                                  grad_accum_steps=1)
    train_loader, test_loader = loader_pred.prep_dataloaders()