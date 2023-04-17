### PreTrained Models 

Most niche problems don't have large datasets so a typical strategy is to start with a Pre-Trained model 
on one task and transfer that knowledge to the new task of interest. To do so we will be attempting to 
classify images of Dogs vs Cats with a few different methods:
- Train entire ResNet18 Model from PyTorch from scratch
- Train only classification head of a Pre-Trained ResNet18 Model
- Leverage the widely popular HuggingFace 🤗 repository to complete the same task

### Dataset Setup
For this tutorial you will need to download the CatsVsDogs dataset. This can be 
easily downloaded from [Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats/data).

Once you have the data downloaded, you will see that all the images are split between train and test
and each folder has both cat and dog images. Create two folders, one for cats and another for dogs and 
update the paths below in *prep_data.ipynb*.
```
PATH_TO_DATA = "data/train/" # Path to original data (unzipped)
PATH_TO_CATS = "data/dogsvcats/cats" # Path to Cats Folder
PATH_TO_DOGS = "data/dogsvcats/dogs" # Path to Dogs Folder
```

Run the notebook and the data should now be in a format that will work with the 
PyTorch ImageFolder DataLoader