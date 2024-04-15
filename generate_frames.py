import torch
import torchvision.transforms as transforms
import torch.nn as nn
import os
import random
from PIL import Image
import numpy as np
from dataset import VimeoDataset
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import cv2 as cv

def Generate(generator, device):
    train_dataset = VimeoDataset(root_dir='src\\vimeo_septuplet')
    train_data = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
        
    test_batch = next(iter(train_data))
    frames, label = test_batch
    print(label)
    frame_list = []
    try:
        for i in range(1):
            for j in range(len(frames)):
                frame_list.append(frames[j][i])
    except IndexError:
        print(f"IndexError, use remainder batch size at batch {1}")

    frame_initial = []
    frame_after = []
    
    for i in range(len(frame_list)):
        if i+1 < len(frame_list) and 0 < ((i+1) % 7):
            frame_initial.append(frame_list[i])
            frame_after.append(frame_list[i+1])
        else:
            pass

    with torch.no_grad():
        frame_initial = torch.stack(frame_initial).to(device).contiguous()
        frame_after = torch.stack(frame_after).to(device).contiguous()
        frame_list_tensor = torch.stack(frame_list).to(device)
        frame_pairs = torch.stack((frame_initial, frame_after)).to(device).contiguous()
        print(frame_initial.shape)
        print(frame_after.shape)
        print(frame_list_tensor.shape)

        fake_frames = (generator(frame_pairs[0], frame_pairs[1]))
        print(fake_frames.shape)
        print(fake_frames[0].shape)
        fake_list = [fake_frames[0], fake_frames[1], fake_frames[2], fake_frames[3], fake_frames[4], fake_frames[5]]
        
        for i in range(len(fake_list)):
            fake_list[i] = fake_list[i].cpu().detach().numpy().transpose(1, 2, 0)
        print(len(fake_list))
        #frame_list_np = [frame.cpu().detach().numpy().transpose(1, 2, 0) for frame in frame_list]

        for i in range(len(fake_list)):
            try:
                mse = mean_squared_error(fake_list[i].flatten(), fake_list[i+1].flatten())
                print(mse)
            except:
                pass

        sequence_path = os.path.join('src\\vimeo_septuplet', 'sequences', label[0])

        frame1 = Image.open(os.path.join(sequence_path, 'im1.png'))
        frame2 = Image.open(os.path.join(sequence_path, 'im2.png'))
        frame3 = Image.open(os.path.join(sequence_path, 'im3.png'))
        frame4 = Image.open(os.path.join(sequence_path, 'im4.png'))
        frame5 = Image.open(os.path.join(sequence_path, 'im5.png'))
        frame6 = Image.open(os.path.join(sequence_path, 'im6.png'))
        frame7 = Image.open(os.path.join(sequence_path, 'im7.png'))

        f = [frame1, frame2, frame3, frame4, frame5, frame6, frame7]
        for i in range(len(fake_list)):
            fake_list[i] = Image.fromarray(fake_list[i].astype(np.uint8))
        ff = [frame1, fake_list[0], frame2, fake_list[1], frame3, fake_list[2], frame4, fake_list[3], frame5, fake_list[4], frame6, fake_list[5], frame7]

        return f, ff

