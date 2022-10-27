import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader,SubsetRandomSampler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data1000.csv', delimiter=',')
# STUDENTS: it may be helpful for the final part to balance the distribution of your collected data

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
# STUDENTS: __len__() returns the length of the dataset
        return (len(self.data))

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.
        return {'input':self.normalized_data[idx,:6],'label':self.normalized_data[idx,6]}


class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary

        #Creating data indices for training and validation splits:
        dataset_size = len(self.nav_dataset)
        indices = list(range(dataset_size))
        val_split=0.2
        val_split = int(np.floor(val_split * dataset_size))
        np.random.seed()
        np.random.shuffle(indices)
        train_indices, val_indices = indices[val_split:], indices[:val_split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = DataLoader(self.nav_dataset, batch_size=batch_size, sampler=train_sampler)
        self.test_loader = DataLoader(self.nav_dataset, batch_size=batch_size, sampler=valid_sampler)


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
        # print(idx, sample)
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']
if __name__ == '__main__':
    main()