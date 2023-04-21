import zipfile
import torchvision

### Unpack and Save CatsVsDogs ###
print("Unpacking CatsvsDogs")
with zipfile.ZipFile("data/kagglecatsanddogs_5340.zip", "r") as zip:
    zip.extractall("data")

### Download MNST Dataset ###
train = torchvision.datasets.MNIST('data', train=True, download=True)
test = torchvision.datasets.MNIST('data', train=False, download=True)


