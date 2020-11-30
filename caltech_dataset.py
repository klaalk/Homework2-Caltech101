from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class CaltechData:
    def __init__(self, image: Image, label: int):
        """
        Single element of Caltech dataset composed by an image and the relative label index

        :param image: data image
        :param label: data label index
        """
        self.image = image
        self.label = label

    def __iter__(self):
        return iter((self.image, self.label))


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        """
        Caltech class extend VisionDataset abstract class.
        All dataset are stored in memory and use indexes to access the image-label pair.
        Labels start from 0 to 100 (background class is excluded)

        :param root: path to data directory
        :param split: defines the split you are going to use (train or test)
        :param transform:
        :param target_transform:
        """
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split  # This defines the split you are going to use

        # Store all data
        dataset = []
        # Store all labels
        labels = []
        # Check if in correct range
        if split in ("train", "test"):
            # Open file
            with open(f"Caltech/{split}.txt", 'r') as f:
                for path in f.readlines():
                    path = path.strip("\n")
                    # Get the label from the path
                    label = path.split("/")[0]
                    # Avoid BACKGROUND label 
                    if not label.startswith("BACKGROUND"):
                        if label not in labels:
                            labels.append(label)
                        image = pil_loader(os.path.join(root, path))
                        dataset.append(CaltechData(image, labels.index(label)))
        else:
            print("Error: split={} not in range".format(split))

        self.total = len(dataset)
        self.dataset = dataset
        self.labels = labels
        print("Total labels={}\nTotal data={}".format(len(labels), len(dataset)))

    def __getitem__(self, index):
        """
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        image, label = None, None

        if len(self.dataset) > index >= 0:
            image, label = self.dataset[index]
        else:
            print("Error: input index={} out of range".format(index))

        # Applies preprocessing when accessing the image
        if image is not None and self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        """
        # Provide a way to get the length (number of elements) of the dataset
        return self.total
