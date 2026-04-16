import os
import json
from torch.utils.data import Dataset, DataLoader
import random
from glob import glob


def split_size(data:list, s_idx=None, e_idx=None):
    if s_idx is not None and e_idx is not None:
        data = data[s_idx:e_idx]
    elif s_idx is not None:
        data = data[s_idx:]
    elif e_idx is not None:
        data = data[:e_idx]
    return data

def read_jsonl(path: str):
    if not path.endswith(".jsonl"):
        raise ValueError(f"Unexpected File: {path}")
    
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty line
            data.append(json.loads(line))
    return data

class GenEval(Dataset):
    def __init__(self, data_path, s_idx=None, e_idx=None): 
        with open(data_path, 'r') as f:
            self.dataset = [json.loads(line) for line in f]

        self.s_idx = s_idx if s_idx is not None else 0
        self.dataset = split_size(self.dataset, s_idx, e_idx)
        self.total_length = len(self.dataset)
        print("Total length of eval dataset: ", self.total_length)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        original_idx = self.s_idx + idx 
        return self.dataset[idx], original_idx 

class DPGEval(Dataset):
    def __init__(self, 
                 file_path,
                 s_idx=None,
                 e_idx=None):
        
        self.dataset = []

        file_list = sorted(glob(os.path.join(file_path,'*.txt')))

        for data_idx, file in enumerate(file_list):
            with open(file, 'r') as f:
                text = f.read()
            data = {'idx': data_idx, 
                    'prompt': text, 
                    'file_name': file.split('/')[-1].split('.')[0]}
            self.dataset.append(data)

        self.dataset = split_size(self.dataset, s_idx, e_idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # return self.dataset[idx], idx
        sample = self.dataset[idx]
        return sample, sample['idx']

class T2ICompBench(Dataset):
    def __init__(self, 
                data_dir, 
                category_list: list = None, 
                split=None,
                s_idx=None,
                e_idx=None):

        self.full_data = []
        if category_list is None:
            raise ValueError("Category list is not given.")
        else:
            print(f"Category list: {category_list}")

        split = "" if split is None else "_" + split        
        print(f"Load {split} split.")

        for category in category_list:
            idx = 0
            self.data = []
            with open(os.path.join(data_dir, f"{category}{split}.txt"), 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    self.data.append({'category': category, 'caption': line, 'idx': idx})
                    idx += 1
            self.data = split_size(self.data, s_idx, e_idx)
            self.full_data.extend(self.data)
        
        
        self.total_length = len(self.full_data)
        print("Total length of eval dataset: ", self.total_length)
        print()
        
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        return self.full_data[idx]