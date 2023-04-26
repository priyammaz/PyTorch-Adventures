## PyTorch Datasets and DataLoaders &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nurV-kJmoPYlXP-qNAGGLsFXuS3lpNil?usp=sharing)

There is no way to train a model without first knowing how to work with different data types and load them in an 
efficient manner. For this we will explore the incredible PyTorch Dataset and DataLoader that handles most of the heavy lifting for you!!
We will be focusing on two domains in this part:

1) **Computer Vision** 
   1) How to Build a custom PyTorch Dataset for the Dogs Vs Cats Data 
   2) Some introduction to the Transforms module in Torchvision 
2) **Natural Language** 
   1) How to load sequence data for the IMBD dataset
   2) Custom data collator to manage sequences of different lengths

### Typical Dataset/DataLoader Setup
```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class DogsVsCats(Dataset):
    def __init__(self, path_to_folder, transforms):
        self.training_files = PATH_TO_FILES 
        self.dog_label, self.cat_label = 0, 1 

        self.transform = transforms 
        
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


dataset = DogsVsCats(path_to_folder, transforms)
dataloader = DataLoader(dataset, batch_size=16)

for images, labels in dataloader:
    print(images.shape)
    print(labels)
    break
```
We will be breaking all of this down in detail to best understand how to manipulate and pass data
as efficiently as possible to our model!
