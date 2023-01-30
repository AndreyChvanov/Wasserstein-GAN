import os
from torch.utils.data import Dataset
import gzip
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class MNISTDataset(Dataset):
    def __init__(self, data_path, dataset_type, transforms, class_to_use=0):
        self.imgs = None
        self.class_to_use = class_to_use
        self.labels = None
        self.data_path = data_path
        self.type = dataset_type
        self.transforms = transforms
        self.read_data()

    def read_data(self):
        filenames = os.listdir(self.data_path)
        print()
        if self.type == 'train':
            self.filenames = [file for file in filenames if 'train' in file]
        else:
            self.filenames = [filename for filename in filenames if 'train' not in filename]
        with gzip.open(os.path.join(self.data_path, self.filenames[0])) as f:
            self.imgs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        with gzip.open(os.path.join(self.data_path, self.filenames[1])) as f:
            self.labels = np.frombuffer(f.read(), np.uint8, offset=8).astype(int)

        self.imgs = self.imgs[self.labels == self.class_to_use]
        self.labels = self.labels[self.labels == self.class_to_use]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = Image.fromarray(img)
        if self.transforms is not None:
                img = self.transforms(img)
        return {'img': img, 'label': self.labels[idx]}


if __name__ == "__main__":
    dataset = MNISTDataset(data_path='../data/MNIST/raw', dataset_type='train',
                           transforms=None, class_to_use=1)
    for i in range(len(dataset)):
        plt.imshow(dataset[i]['img'])
        # plt.title('class {}'.format(dataset[i]['label']))
        # plt.show()
        assert dataset[i]['label'] == 1
