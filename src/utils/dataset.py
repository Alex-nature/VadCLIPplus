import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import os
import utils.tools as tools

class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, normal: bool = False):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # Convert path to match current environment
        original_path = self.df.loc[index]['path']
        if original_path.startswith('/home/xbgydx/Desktop/UCFClipFeatures/'):
            # Replace with the correct local path relative to project root
            relative_path = original_path.replace('/home/xbgydx/Desktop/UCFClipFeatures/', 'data/CLIP_Features/UCF-Crime/UCFClipFeatures/')
            # Use os.path.join to handle different OS path separators
            clip_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), relative_path)
        else:
            clip_path = original_path

        clip_feature = np.load(clip_path)
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length

class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # Convert path to match current environment
        original_path = self.df.loc[index]['path']
        if original_path.startswith('/home/xbgydx/Desktop/XDTrainClipFeatures/'):
            # Replace with the correct local path relative to project root
            relative_path = original_path.replace('/home/xbgydx/Desktop/XDTrainClipFeatures/', 'data/CLIP_Features/XD-Violence/XDTrainClipFeatures/')
            # Use os.path.join to handle different OS path separators
            clip_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), relative_path)
        elif original_path.startswith('/home/xbgydx/Desktop/XDTestClipFeatures/'):
            # Handle test data path
            relative_path = original_path.replace('/home/xbgydx/Desktop/XDTestClipFeatures/', 'data/CLIP_Features/XD-Violence/XDTestClipFeatures/')
            # Use os.path.join to handle different OS path separators
            clip_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), relative_path)
        else:
            clip_path = original_path

        clip_feature = np.load(clip_path)
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length