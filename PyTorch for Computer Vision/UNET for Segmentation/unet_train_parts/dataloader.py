import os
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

class ADE20KDataset(Dataset):
    def __init__(self, path_to_data, 
                 train=True, 
                 image_size=128, 
                 random_crop_ratio=(0.2, 1),
                 inference_mode=False):
        
        self.path_to_data = path_to_data
        self.inference_mode = inference_mode
        self.train = train
        self.image_size = image_size
        self.min_ratio, self.max_ratio = random_crop_ratio

        if train:
            split = "training"
        else:
            split = "validation"

        ### Get Path to Images and Segmentations ###
        self.path_to_images = os.path.join(self.path_to_data, "images", split)
        self.path_to_annotations = os.path.join(self.path_to_data, "annotations", split)

        ### Get All Unique File Roots ###
        self.file_roots = [path.split(".")[0] for path in os.listdir(self.path_to_images)]

        ### Store all Transforms we want ###
        self.resize = transforms.Resize(size=(self.image_size, self.image_size))
        self.normalize = transforms.Normalize(mean=(0.48897059, 0.46548275, 0.4294), 
                                              std=(0.22861765, 0.22948039, 0.24054667))
        self.random_resize = transforms.RandomResizedCrop(size=(self.image_size, self.image_size))
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1)
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.file_roots)

    def __getitem__(self, idx):

        ### Grab File Root ###
        file_root = self.file_roots[idx]

        ### Get Paths to Image and Annotation ###
        image = os.path.join(self.path_to_images, f"{file_root}.jpg")
        annot = os.path.join(self.path_to_annotations, f"{file_root}.png")

        ### Load Image and Annotation ###
        image = Image.open(image).convert("RGB")
        annot = Image.open(annot)

        ### Train Image Transforms ###
        if self.train and (not self.inference_mode):

            ### Resize Image and Annotation ###
            if random.random() < 0.5:
                
                image = self.resize(image)
                annot = self.resize(annot)

            ### Random Resized Crop ###
            else:

                ### Get Smaller Side ###
                min_side = min(image.size)
    
                ### Get a Random Crop Size with Ratio ###
                random_ratio = random.uniform(self.min_ratio, self.max_ratio)

                ### Compute Crop Size ###
                crop_size = int(random_ratio * min_side)

                ### Get Parameters of Random Crop ###
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(crop_size, crop_size))

                ### Crop Image and Annotation ###
                image = TF.crop(image, i, j, h, w)
                annot = TF.crop(annot, i, j, h, w)

                ### Resize Image to Desired Image Size ###
                image = self.resize(image)
                annot = self.resize(annot)
            

            ### Random Horizontal Flip ###
            if random.random() < 0.5:
                image = self.horizontal_flip(image)
                annot = self.horizontal_flip(annot)

        ### Validation Image Transforms ###
        else:

            image = self.resize(image)
            annot = self.resize(annot)
                
        ### Convert Everything to Tensors ###
        image = self.totensor(image)
        annot = torch.tensor(np.array(annot), dtype=torch.long)

        ### Update Annotations as class 0 is other and not needed ###
        annot = annot - 1 # Make it from [0-150] to [-1-149]

        ### Normalize Image ###
        image = self.normalize(image)

        return image, annot

class CarvanaDataset(Dataset):
    """
    Carvana Class: This will do exactly what the ADE20K class was doing, but also include a random sampling
    of data as Carvana doesn't automatically split the dataset into training and validation (with an included seed)
    """
    def __init__(self, path_to_data, train=True, image_size=128, random_crop_ratio=(0.5, 1), seed=0, test_pct=0.1):
        self.path_to_data = path_to_data
        self.train = train
        self.image_size = image_size
        self.min_ratio, self.max_ratio = random_crop_ratio

        ### Get Path to Images and Segmentations ###
        self.path_to_images = os.path.join(self.path_to_data, "train")
        self.path_to_annotations = os.path.join(self.path_to_data, "train_masks")

        ### Get All Unique File Roots ###
        file_roots = [path.split(".")[0] for path in os.listdir(self.path_to_images)]

        ### Random Split Dataset into Train/Test ###
        random.seed(0)
        testing_data = random.sample(file_roots, int(test_pct*len(file_roots)))
        training_data = [sample for sample in file_roots if sample not in testing_data]

        if self.train:
            self.file_roots = training_data
        else:
            self.file_roots = testing_data
            
        ### Store all Transforms we want ###
        self.resize = transforms.Resize(size=(self.image_size, self.image_size))
        self.normalize = transforms.Normalize(mean=(0.48897059, 0.46548275, 0.4294), 
                                              std=(0.22861765, 0.22948039, 0.24054667))
        self.random_resize = transforms.RandomResizedCrop(size=(self.image_size, self.image_size))
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1)
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.file_roots)

    def __getitem__(self, idx):

        ### Grab File Root ###
        file_root = self.file_roots[idx]

        ### Get Paths to Image and Annotation ###
        image = os.path.join(self.path_to_images, f"{file_root}.jpg")
        annot = os.path.join(self.path_to_annotations, f"{file_root}_mask.gif")

        ### Load Image and Annotation ###
        image = Image.open(image).convert("RGB")
        annot = Image.open(annot)

        ### Train Image Transforms ###
        if self.train:

            ### Resize Image and Annotation ###
            if random.random() < 0.5:
                
                image = self.resize(image)
                annot = self.resize(annot)

            ### Random Resized Crop ###
            else:

                ### Get Smaller Side ###
                min_side = min(image.size)
    
                ### Get a Random Crop Size with Ratio ###
                random_ratio = random.uniform(self.min_ratio, self.max_ratio)

                ### Compute Crop Size ###
                crop_size = int(random_ratio * min_side)

                ### Get Parameters of Random Crop ###
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(crop_size, crop_size))

                ### Crop Image and Annotation ###
                image = TF.crop(image, i, j, h, w)
                annot = TF.crop(annot, i, j, h, w)

                ### Resize Image to Desired Image Size ###
                image = self.resize(image)
                annot = self.resize(annot)
            

            ### Random Horizontal Flip ###
            if random.random() < 0.5:
                image = self.horizontal_flip(image)
                annot = self.horizontal_flip(annot)

        ### Validation Image Transforms ###
        else:

            image = self.resize(image)
            annot = self.resize(annot)
                
        ### Convert Everything to Tensors ###
        image = self.totensor(image)
        annot = torch.tensor(np.array(annot), dtype=torch.float) # BCEWithLogits needs float tensor

        ### Normalize Image ###
        image = self.normalize(image)

        return image, annot
        