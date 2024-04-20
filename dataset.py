from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
from PIL import Image

worker_threads = 2
batch_size = 1
class VimeoDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Load file paths for the specified split
        split_file = os.path.join(root_dir, f"sep_{split}list.txt")
        with open(split_file, 'r') as f:
            self.file_paths = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        sequence_path = os.path.join(self.root_dir, 'sequences', self.file_paths[idx])
        # Loads all frames from the path as pillow images
        frame1path = os.path.join(sequence_path, 'im1.png')
        frame2path = os.path.join(sequence_path, 'im2.png')
        frame3path = os.path.join(sequence_path, 'im3.png')
        frame4path = os.path.join(sequence_path, 'im4.png')
        frame5path = os.path.join(sequence_path, 'im5.png')
        frame6path = os.path.join(sequence_path, 'im6.png')
        frame7path = os.path.join(sequence_path, 'im7.png')

        frame1 = Image.open(frame1path)
        frame2 = Image.open(frame2path)
        frame3 = Image.open(frame3path)
        frame4 = Image.open(frame4path)
        frame5 = Image.open(frame5path)
        frame6 = Image.open(frame6path)
        frame7 = Image.open(frame7path)

        # Pads them to a square shape and normalizes the pixel values
        width, height = 448, 256
        left_pad = (512 - width) // 2
        right_pad = 512 - width - left_pad
        top_pad = (512 - height) // 2
        bottom_pad = 512 - height - top_pad
        frames = [frame1, frame2, frame3, frame4, frame5, frame6, frame7]
        for i in range(len(frames)):
            frames[i] = transforms.functional.pad(frames[i], (left_pad, top_pad, right_pad, bottom_pad))
            frames[i] = (np.array(frames[i]).astype(np.float32) - 127.5) / 127.5

            if self.transform:
                frames[i] = self.transform(frames[i])

        return frames, self.file_paths[idx]
    
class MSUDataset(Dataset):
    def __init__(self, root_dir, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])):
        self.root_dir = root_dir
        self.transform = transform

        # Load file paths for the specified split
        split_file = os.path.join(root_dir, f"train.txt")
        with open(split_file, 'r') as f:
            self.file_paths = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        sequence_path = os.path.join(self.root_dir, self.file_paths[idx])
        frames = []
        paths = os.listdir(sequence_path)
        for i in range(len(paths)):
            frames.append(Image.open(os.path.join(sequence_path, paths[i])))

        width, height = 1920, 1080
        left_pad = (2048 - width) // 2
        right_pad = 2048 - width - left_pad
        top_pad = (2048 - height) // 2
        bottom_pad = 2048 - height - top_pad
        for i in range(len(frames)):
            frames[i] = transforms.functional.pad(frames[i], (left_pad, top_pad, right_pad, bottom_pad))
            frames[i] = (np.array(frames[i]).astype(np.float32) - 127.5) / 127.5

            if self.transform:
                frames[i] = self.transform(frames[i])

        return frames, self.file_paths[idx]
    
