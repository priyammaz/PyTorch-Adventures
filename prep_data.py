import zipfile
import torchvision
import os
from PIL import Image

### Unpack and Save CatsVsDogs ###
print("Unpacking CatsvsDogs")
with zipfile.ZipFile("data/kagglecatsanddogs_5340.zip", "r") as zip:
    zip.extractall("data")
    
### Clean Up CatsVDogs ###
path_to_cats = os.path.join("../data/PetImages/", "Cat") # Get Path to Cat folder
path_to_dogs = os.path.join("../data/PetImages/", "Dog") # Get Path to Dog folder 

dog_files = os.listdir(path_to_dogs) # Get list of all files inside of dog folder
cat_files = os.listdir(path_to_cats) # Get list of all files inside cat folder

path_to_dog_files = [os.path.join(path_to_dogs, file) for file in dog_files] # Get full path to each cat file
path_to_cat_files = [os.path.join(path_to_cats, file) for file in cat_files] # Get full path to each dog file

path_to_files = path_to_dog_files + path_to_cat_files

for file in path_to_files:
    try:    
        img = np.array(Image.open(file))
        if img.shape[-1] != 3: 
            os.remove(file) # Delete image if it doesnt have three channels
    except:
        os.remove(file) # Delete if not an image file, or broken image

### Download MNST Dataset ###
train = torchvision.datasets.MNIST('data', train=True, download=True)
test = torchvision.datasets.MNIST('data', train=False, download=True)


