from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import lightning as L

worker_threads = 2
batch_size = 32
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
    
class VimeoDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = 'src\\vimeo_septuplet', batch_size: int = batch_size, transform=transforms.ToTensor()):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform

    def prepare_data(self):
        '''
        Empty method since the dataset is stored locally not downloaded
        More details here: https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
        '''
        pass

    def setup(self, stage: str):
        if stage == 'fit' or stage is None: # Training dataset
            train_set_full = VimeoDataset(root_dir=self.data_dir, split='train')
            total_samples = len(train_set_full)
            train_size = int(0.9 * total_samples)
            val_size = total_samples - train_size
            self.train, self.validate = random_split(train_set_full, [train_size, val_size])
        
        if stage == 'test' or stage is None:
            self.test = VimeoDataset(root_dir=self.data_dir, split='test')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=batch_size, num_workers=worker_threads)
    
    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=batch_size, num_workers=worker_threads)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=batch_size, num_workers=worker_threads)