import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform #None

        # Total number of datapoints
        # #FIXME：original
        self.dataset_size = int(len(self.captions) / 1)

        # FIXME:my
        # if self.split == 'TRAIN':
        #     self.dataset_size = int(len(self.captions)/1)
        # else:
        #     self.dataset_size = int(len(self.captions) / 5)


    def __getitem__(self, i):
        # FIXME：original
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            if img.shape == torch.Size([3,256,256]):
                img = self.transform(img)
            elif img.shape == torch.Size([2,3,256,256]):
                ori_img = img
                img[0] = self.transform(img[0])
                img[1] = self.transform(img[1])

        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

        # #FIXME：my; Now i is i-th image
        # if self.split is 'TRAIN':
        #     i = i * self.cpi
        #     img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        #     if self.transform is not None:
        #         img = self.transform(img)  # TODO:img[0] = self.transform(img[0])
        #
        #     caption = torch.LongTensor(self.captions[i])
        #     caplen = torch.LongTensor([self.caplens[i]])
        #     return img, caption, caplen
        # else:
        #     # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
        #     img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        #     if self.transform is not None:
        #         img = self.transform(img)  # TODO:img[0] = self.transform(img[0])
        #
        #     caption = torch.LongTensor(self.captions[i])
        #     caplen = torch.LongTensor([self.caplens[i]])
        #
        #     all_captions = torch.LongTensor(
        #         self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
        #     return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
