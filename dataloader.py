import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
import os

class FoodDataset(Dataset):
	def __init__(self, root_dir_path, transforms=None, train=True):
		self.root = root_dir_path
		self.transforms = transforms
		if train:
			self.labels = pd.read_csv(os.path.join(root_dir_path, ['meta', 'train.txt']))
		else:
			self.labels = pd.read_csv(os.path.join(root_dir_path, ['meta', 'test.txt']))

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		pass



class CifarLoader(object):
	"""docstring for CifarLoader"""
	def __init__(self, args):
		super(CifarLoader, self).__init__()
		transform = transforms.Compose(
		    [
			 transforms.RandomResizedCrop(32, scale=(0.6, 0.8)),
		     transforms.RandomHorizontalFlip(),
		     transforms.ToTensor(),
			# kept the normalize values as is as mentioned by the PyTorch dev in this thread
			#  https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/6
		     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		     ])

		transform_test = transforms.Compose([
            transforms.Resize(32),
		    transforms.ToTensor(),
		    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
		])

		'''trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
		                                        download=True, transform=transform)'''

		trainset = torchvision.datasets.ImageFolder(root="food-101/train_images/images", transform=transform)
		self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchSize,
		                                          shuffle=True, num_workers=2)

		testset = torchvision.datasets.ImageFolder(root="food-101/test_images/images", transform=transform_test)
		self.testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchSize,
		                                         shuffle=False, num_workers=2)

		self.classes = ("Baby back ribs","Baklava","Beef carpaccio","Beef tartare","Beet salad","Beignets","Bibimbap","Bread pudding","Breakfast burrito","Bruschetta","Caesar salad","Cannoli","Caprese salad","Carrot cake","Ceviche","Cheesecake","Cheese plate","Chicken curry","Chicken quesadilla","Chicken wings","Chocolate cake","Chocolate mousse","Churros","Clam chowder","Club sandwich","Crab cakes","Creme brulee","Croque madame","Cup cakes","Deviled eggs","Donuts","Dumplings","Edamame","Eggs benedict","Escargots","Falafel","Filet mignon","Fish and chips","Foie gras","French fries","French onion soup","French toast","Fried calamari","Fried rice","Frozen yogurt","Garlic bread","Gnocchi","Greek salad","Grilled cheese sandwich","Grilled salmon","Guacamole","Gyoza","Hamburger","Hot and sour soup","Hot dog","Huevos rancheros","Hummus","Ice cream","Lasagna","Lobster bisque","Lobster roll sandwich","Macaroni and cheese","Macarons","Miso soup","Mussels","Nachos","Omelette","Onion rings","Oysters","Pad thai","Paella","Pancakes","Panna cotta","Peking duck","Pho","Pizza","Pork chop","Poutine","Prime rib","Pulled pork sandwich","Ramen","Ravioli","Red velvet cake","Risotto","Samosa","Sashimi","Scallops","Seaweed salad","Shrimp and grits","Spaghetti bolognese","Spaghetti carbonara","Spring rolls","Steak","Strawberry shortcake","Sushi","Tacos","Takoyaki","Tiramisu","Tuna tartare","Waffles")
