from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
from PIL import Image

class VimeoDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=transforms.ToTensor()):
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

        frame1path = os.path.join(sequence_path, 'im1.png')
        frame2path = os.path.join(sequence_path, 'im2.png')
        frame3path = os.path.join(sequence_path, 'im3.png')
        frame4path = os.path.join(sequence_path, 'im4.png')
        frame5path = os.path.join(sequence_path, 'im5.png')
        frame6path = os.path.join(sequence_path, 'im6.png')
        frame7path = os.path.join(sequence_path, 'im7.png')

        frame1 = (np.array(Image.open(frame1path)).astype(np.float32) - 127.5) / 127.5
        frame2 = (np.array(Image.open(frame2path)).astype(np.float32) - 127.5) / 127.5
        frame3 = (np.array(Image.open(frame3path)).astype(np.float32) - 127.5) / 127.5
        frame4 = (np.array(Image.open(frame4path)).astype(np.float32) - 127.5) / 127.5
        frame5 = (np.array(Image.open(frame5path)).astype(np.float32) - 127.5) / 127.5
        frame6 = (np.array(Image.open(frame6path)).astype(np.float32) - 127.5) / 127.5
        frame7 = (np.array(Image.open(frame7path)).astype(np.float32) - 127.5) / 127.5

        if self.transform:
            frame1 = self.transform(frame1)
            frame2 = self.transform(frame2)
            frame3 = self.transform(frame3)
            frame4 = self.transform(frame4)
            frame5 = self.transform(frame5)
            frame6 = self.transform(frame6)
            frame7 = self.transform(frame7)

        return (frame1, frame2, frame3, frame4, frame5, frame6, frame7), self.file_paths[idx]
