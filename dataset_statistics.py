"""
Title: Generative Adversarial Network - Video Frame Interpolation (GAN-VFI)
Author: Chris McDonald
Date: April 23, 2024

Description:
This script is used to calculate the mean and standard deviation of pixel values for each channel across every image in the dataset.
It contains a method for the user to decide how to obtain the normalized dataset.
Reloads the dataset if any transformations were altered from the initial dataload.

Usage:
This file should not be run on it's own.
Run the main script using `python main.py` in a terminal.

Notes:
This file is imported in the following scripts:
main.py

It imports the following scripts:
dataset.py
format_time.py
"""

# Importing external libraries

import torch                                # PyTorch library - Used for tensors and neural networks in the deep learning model.
import time                                 # Time module - Used to check how long specific sections of code take to execute.
from PIL import Image                       # Image module of Pillow library - Used for processing frames into images.
import numpy as np                          # NumPy library - Used for numerical computation in multi-dimensional arrays.
import os                                   # os module - Used for file and directory manipulation. 
import random                               # random module - Used to generate "random" numbers.
from torchvision import transforms          # transforms module of torchvision library - Used for transformation functions applied to images.
from torch.utils.data import DataLoader     # DataLoader class of torch.utils.data module - Used for loading dataset into batches.

# Importing local scripts

from dataset import VimeoDataset            # 
from format_time import format_duration     # 

class dataset_statistics():
    def get_stats():
        dir = 'src\\vimeo_septuplet'
        split_file = os.path.join(dir, f"sep_trainlist.txt")

        with open(split_file, 'r') as f:
            file_paths = [line.strip() for line in f.readlines()]

        mean = [0.0, 0.0, 0.0]
        std = [0.0, 0.0, 0.0]
        pixels = 0
        images = 0
        frame_list = ['im1.png', 'im2.png', 'im3.png', 'im4.png', 'im5.png', 'im6.png', 'im7.png']
        h, w, _ = np.array(Image.open('src\\vimeo_septuplet\\sequences\\00001\\0001\\im1.png')).shape
        print(h, w)
        start_duration = time.perf_counter()
        for idx in range(len(file_paths)):
            sequence_path = os.path.join(dir, 'sequences', file_paths[idx])

            for index in range(len(frame_list)):
                frame = np.array(Image.open(os.path.join(sequence_path, frame_list[index]))) / 255.0

                temp_mean = np.mean(frame, axis=(0, 1))
                temp_std = np.std(frame, axis=(0, 1))

                if images == 0:
                    mean = temp_mean
                    std = temp_std
                else:
                    mean = ((mean*images)+temp_mean)/(images+1)
                    std = ((std*images)+temp_std)/(images+1)

                images += 1
                pixels += (h*w)

                print('\r'+f'{images}/{len(file_paths)*7} processed ({(images / (len(file_paths)*7))*100:.3f}%) Mean: {mean} Std: {std}',end='')

        end_duration = time.perf_counter()
        duration_spent = format_duration(start_duration, end_duration)
        print(f'\nGenerating statistics took {duration_spent}')
        print(f"Final values:\nMean: {mean}\nStandard Deviation: {std}")

    def get_train_stats(dataloader, device, batch_size, training_batches, training_remainder, train_dataset):
        total_pixels = 0
        sample_count = 0
        mean = torch.zeros(3, device=device)
        standard_deviation = torch.zeros(3, device=device)
        start_duration = time.perf_counter()
        for idx, batch in enumerate(dataloader):
            frames, _ = batch
            temp = batch_size
            if idx == training_batches - 1:
                temp = training_remainder
            for i in range(temp):
                for j in range(len(frames)):
                    currentframe = frames[j][i]
                    currentframe = currentframe.to(device)
                    c, h, w = currentframe.shape
                    pixel_count = h * w
                    channel_sum = torch.sum(currentframe, dim=[1, 2])
                    channel_sum_squared = torch.sum(currentframe ** 2, dim=[1, 2])
                    mean = (total_pixels * mean + channel_sum) / (total_pixels + pixel_count)
                    standard_deviation = (total_pixels * standard_deviation + channel_sum_squared) / (total_pixels + pixel_count)
                    sample_count += 1
                    total_pixels += pixel_count
                    print('\r'+f'{sample_count}/{len(train_dataset)*7} processed ({(sample_count / (len(train_dataset)*7))*100:.3f}%) Mean: {mean} Std: {standard_deviation}',end='')
    
        end_duration = time.perf_counter()
        duration_spent = format_duration(start_duration, end_duration, device)
        print(f'\nGenerating statistics took {duration_spent}')
        return mean, standard_deviation

    def dataset_options(device, batch_size, training_batches, training_remainder, worker_threads):
        # Try calculating mean and std using dataset and txt file rather than class created
        normalize_method = input("Do you want to apply an additional (deprecated) normalize method? (y/n) ")
        if normalize_method == 'y':
            normalize_method = input("Using the whole dataset or the training dataset: (1/2) ")
            if normalize_method != 1 or 2:
                normalize_method = random.randint(1, 2)
                print(f"Invalid input, choosing at random: {normalize_method}")
            
            if normalize_method == 1:
                if dataset_mean == [0,0,0] and dataset_standard_deviation == [0,0,0]:
                    dataset_mean, dataset_standard_deviation = dataset_statistics.get_stats()
                else:
                    recheck_stats = input(f"The RGB mean {dataset_mean} and standard deviation {dataset_standard_deviation} have already been calculated. Retry? (y/n) ")
                    if recheck_stats == 'y':
                        dataset_mean, dataset_standard_deviation = dataset_statistics.get_stats()
                    elif recheck_stats != 'y':
                        pass
            elif normalize_method == 2:
                if train_dataset_mean == [0,0,0] and train_dataset_standard_deviation == [0,0,0]:
                    train_dataset_mean, train_dataset_standard_deviation = dataset_statistics.get_train_stats(train_data, device, batch_size, training_batches, training_remainder, train_dataset)
                else:
                    recheck_stats = input(f"The RGB mean {train_dataset_mean} and standard deviation {train_dataset_standard_deviation} have already been calculated. Retry? (y/n) ")
                    if recheck_stats == 'y':
                        dataset_mean, dataset_standard_deviation = dataset_statistics.get_train_stats(train_data, device, batch_size, training_batches, training_remainder, train_dataset)
                    elif recheck_stats != 'y':
                        pass
                    
            normalize_dataset = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=dataset_mean, std=dataset_standard_deviation)
            ])

            train_dataset = VimeoDataset(root_dir='src\\vimeo_septuplet', split='train', transform=normalize_dataset)
            #print("Training dataset reread")
            train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=worker_threads)
            #print("Training dataset reloaded")

            test_dataset = VimeoDataset(root_dir='src\\vimeo_septuplet', split='test', transform=normalize_dataset)
            #print("Testing dataset reread")
            test_data = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=worker_threads)
            #print("Testing dataset reloaded")
        else:
            pass
