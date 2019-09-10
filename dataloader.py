import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
from PIL import Image
import glob

from torchvision import transforms

class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId:

            data_image = Image.open(self.data_files[current])
            label_image = Image.open(self.label_files[current])

            if self.mode == 'train':
                if random.random() > 0.5:
                    print("1")
                    gamma = random.randint(10, 12) / 10
                    data_image = transforms.functional.adjust_gamma(data_image, gamma, gain=1)
                    #label_image = transforms.functional.adjust_gamma(label_image, gamma, gain=1)
                    

                if random.random() > 0.5:
                    
                    print("2")
                    hue_factor = random.randint(1, 5) / 10
                    data_image = transforms.functional.adjust_hue(data_image, hue_factor)
                    #label_image = transforms.functional.adjust_hue(label_image, hue_factor)
                
                # if random.random() > 5:
                #     i = random.randint(0, data_image.shape[1] / 2)
                #     j = random.randint(0, data_image.shape[0] / 2)
                #     h = random.randint(20, data_image.shape[1] - i)
                #     w = random.randint(20, data_image.shape[0] - j)
                #     data_image = transforms.functional.crop(data_image, i, j, h, w)
                #     label_image = transforms.functional.crop(label_image, i, j, h, w)
                
                if random.random() > 0.5:
                    
                    print("3")
                    data_image = transforms.functional.hflip(data_image)
                    label_image = transforms.functional.hflip(label_image)

                if random.random() > 0.5:
                    
                    print("4")
                    data_image = transforms.functional.vflip(data_image)
                    label_image = transforms.functional.vflip(label_image)

            data_image = data_image.resize((388, 388))
            label_image = label_image.resize((388, 388))

            data_image = np.array(data_image)
            label_image = np.array(label_image)

            data_image = np.pad(data_image, (94, 94), 'symmetric')

            data_image = data_image / 255
            
            current += 1

            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))
        
