import torch
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import default_collate

### Set Transforms for Training and Testing ###
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
    
def train_transformations(image_size=(224,224),
                          image_mean=IMAGENET_MEAN, 
                          image_std=IMAGENET_STD, 
                          hflip_probability=0.5,
                          interpolation=InterpolationMode.BILINEAR, 
                          random_aug_magnitude=9):
    
    """
    Dataloader with Random Augmentation

    Args: 
        image_size: What size image do we want to return?
        image_mean: Mean of the image channels
        image_std: Standard deviation of the image channels
        hflip_probability: Probability of random horizontal flipping
        interpolation: What interpolation model do you want to use?
        random_aug_magnitude: Strength of augmentations (Valid values between 0 and 30)
    """
                
    ### Create a List to Store Transformations ###
    transformation_chain = []

    ### We need to grab a Random Crop of the image size we want (We will just use RandomResizeCrop defauls) ###
    transformation_chain.append(v2.RandomResizedCrop(image_size, interpolation=interpolation, antialias=True))

    ### Random Horizontal Flipping ###
    if hflip_probability > 0:
        transformation_chain.append(v2.RandomHorizontalFlip(p=hflip_probability))
    
    ### Auto Augmentation ###
    if random_aug_magnitude > 0:
        print("Enabling Random Augmentation!")
        transformation_chain.append(v2.RandAugment(magnitude=random_aug_magnitude, interpolation=interpolation))

    ### We need to convert PIL Images to Tensor ###
    transformation_chain.append(v2.PILToTensor())

    ### Conver to float32 and scale to [0,1] ###
    transformation_chain.append(v2.ToDtype(torch.float32, scale=True))

    ### We need to normalize data ###
    transformation_chain.append(v2.Normalize(mean=(image_mean), std=image_std))

    return transforms.Compose(transformation_chain)

def eval_transformations(image_size=(224,224),
                         resize_size=(256,256),
                         image_mean=IMAGENET_MEAN, 
                         image_std=IMAGENET_STD, 
                         interpolation=InterpolationMode.BILINEAR):
    
    """
    Quick evaluation dataloader, no fancy transformations in this one
    
    Args:
        image_size: What size image do we want to pass to model, this is a center crop size?
        resize_size: Original size we resize image to before center crop 
        image_mean: Mean of the image channels
        image_std: Standard deviation of the image channels
        interpolation: What interpolation model do you want to use?
    """
    
    transformations = transforms.Compose(
        [
            v2.Resize(resize_size, interpolation=interpolation, antialias=True),
            v2.CenterCrop(image_size),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=image_mean, std=image_std)
        ]
    )

    return transformations

def mixup_cutmix_collate_fn(mixup_alpha=0.2, cutmix_alpha=1.0, num_classes=1000):

    """
    Wrapper function to add Mixup and Cutmix to our image processing pipelines. 

    Args: 
        mixup_alpha: Alpha parameter for Beta distribution from which mixup lambda is sampled
        cutmix_alpha: Alpha parameter for Beta distribution from which cutmix lambda is samples
        num_classes: How many classes are there to predict from


    Note!

    Normally, we have a single label for each image (and our dataloader returns an index representing
    what class its in). But now instead of returning a single tensor of size (Batch, ) that has these indexes, 
    we will instead return (Batch x Num Classes), as each image (after transformation) will be a mixture of two images
    so we return the proportion of pixels of each image represented in each image. 
    """

    mix_cut_transform = None

    mixup_cutmix = []
    if mixup_alpha > 0:
        print("Enabling MixUp!")
        mixup_cutmix.append(v2.MixUp(alpha=mixup_alpha, num_classes=num_classes))
    if cutmix_alpha > 0:
        print("Enabling CutMix!")
        mixup_cutmix.append(v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes))
    
    if len(mixup_cutmix) > 0:
        mix_cut_transform = v2.RandomChoice(mixup_cutmix)

    def collate_fn(batch):
        collated = default_collate(batch)

        if mix_cut_transform is not None:
            collated = mix_cut_transform(collated)
        return collated

    return collate_fn

def accuracy(output, target, topk=(1,5)):
    """
    Computes the accuracy over the k top predictions for the specified values of k

    This was mostly taken from https://github.com/pytorch/vision/blob/main/references/classification/utils.py

    """
    with torch.inference_mode():

        ### Get the maximum K we want to Compute ###
        maxk = max(topk)

        ### Grab the current batch size ###
        batch_size = target.size(0)

        ### Targets can be Batch x Num Classes due to Cutmix/Mixup so we grab the max class as our "true" label ###
        ### to compute accuracy, even though each image is a mixture of two images ###
        if target.ndim == 2:
            target = target.max(dim=1)[1]
        
        ### Gran the Topk Values ###
        values, pred = output.topk(maxk, dim=-1, largest=True, sorted=True)
        
        ### pred is currently (Batch x maxK), we want (maxK x Batch)
        pred = pred.transpose(0,1)
        
        ### Compare each row of our predictions (for each K value) to our correct targets ###
        correct = (pred == target)

        ### Loop through the K values we want and return the accuracy of the cumulative rows ###
        accs = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            accs.append((correct_k / batch_size))
   
        return accs