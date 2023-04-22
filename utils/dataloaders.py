import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile

### Some Images are Truncated so we will allow them! ###
ImageFile.LOAD_TRUNCATED_IMAGES = True 

class DogsVsCats(Dataset):
    """
    Basic Dataloader made in the PyTorch DataLoaders Tutorial. We will access this throughout!!
    """
    def __init__(self, path_to_folder, img_transforms=None):
        path_to_cats = os.path.join(path_to_folder, "Cat")
        path_to_dogs = os.path.join(path_to_folder, "Dog")
        dog_files = os.listdir(path_to_dogs)
        cat_files = os.listdir(path_to_cats)
        path_to_dog_files = [os.path.join(path_to_dogs, file) for file in dog_files]
        path_to_cat_files = [os.path.join(path_to_cats, file) for file in cat_files]
        self.training_files = path_to_dog_files + path_to_cat_files
        self.dog_label, self.cat_label = 0, 1

        if img_transforms is not None:
            self.transform = img_transforms

        else:
            self.transform = transforms.Compose(
                                [
                                    transforms.Resize((224,224)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                                ]
                            )

    def __len__(self):
        return len(self.training_files)

    def __getitem__(self, idx):
        path_to_image = self.training_files[idx]
        if "Dog" in path_to_image:
            label = self.dog_label
        else:
            label = self.cat_label
        image = Image.open(path_to_image)
        image = self.transform(image)
        return image, label



