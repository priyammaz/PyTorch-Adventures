import zipfile
import torchvision
import os
from PIL import Image
import numpy as np
import argparse

parser = argparse.ArgumentParser("Data Preperation Script")

parser.add_argument("-p", "--path", type=str, help="Path to Data Root")
parser.add_argument("--all", action="store_true")
parser.add_argument("--catsvdogs", action="store_true")
parser.add_argument("--mnist", action="store_true")

args = parser.parse_args()

if args.all is True:
    args.catsvdogs = True
    args.mnist = True

if args.path is None:
    args.path = ""

if args.catsvdogs:
    ### Unpack and Save CatsVsDogs ###
    print("Unpacking CatsvsDogs")
    path_to_catvdog_zip = os.path.join(args.path, "kagglecatsanddogs_5340.zip")
    with zipfile.ZipFile(path_to_catvdog_zip, "r") as zip:
        zip.extractall(args.path)

    ### Clean Up CatsVDogs ###
    path_to_catvdog = os.path.join(args.path, "PetImages")
    path_to_cats = os.path.join(path_to_catvdog, "Cat") # Get Path to Cat folder
    path_to_dogs = os.path.join(path_to_catvdog, "Dog") # Get Path to Dog folder

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

if args.mnist:
    ### Download MNST Dataset ###
    train = torchvision.datasets.MNIST(args.path, train=True, download=True)
    test = torchvision.datasets.MNIST(args.path, train=False, download=True)


