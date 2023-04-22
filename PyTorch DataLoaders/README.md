# PyTorch Datasets and DataLoaders

---
There is no way to train a model without first knowing how to work with different data types and load them in an 
efficient manner. For this we will explore the incredible PyTorch Dataset and DataLoader that handles most of the heavy lifting for you!!
We will be focusing on two domains in this part:

1) Computer Vision 
   1) How to Build a custom PyTorch Dataset for the Dogs Vs Cats Data 
   2) Some introduction to the Transforms module in Torchvision 
2) Natural Language 
   1) How to load sequence data for the IMBD dataset
   2) Custom data collator to manage sequences of different lengths

Once we have built custom datasets, we will then look how to wrap it in the DataLoader to enable us to grab minibatches!!
