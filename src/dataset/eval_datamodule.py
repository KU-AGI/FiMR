import os
from glob import glob

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset

import pytorch_lightning as pl

from src.dataset.eval_dataset import GenEval, T2ICompBench, DPGEval, TIIFEval


class EvalDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = None
    
    def setup(self, stage=None):
        task = self.config.task_name
        path = self.config.dataset.get(task, None)

        if path is None:
            raise ValueError(f"Invalid task name: {task}")

        if task == "geneval":
            self.dataset = GenEval(data_path=path,
                                  s_idx=self.config.dataset.s_idx,
                                  e_idx=self.config.dataset.e_idx)

        elif task == "t2icompbench":
            self.dataset = T2ICompBench(data_dir=path,
                                       category_list=self.config.dataset.category,
                                       split=self.config.dataset.split,
                                       s_idx=self.config.dataset.s_idx,
                                       e_idx=self.config.dataset.e_idx)
        elif task == "dpgbench":
            self.dataset = DPGEval(file_path=path,
                                   s_idx=self.config.dataset.s_idx,
                                   e_idx=self.config.dataset.e_idx)

        elif task == "tiif": 
            self.dataset = TIIFEval(data_dir=path, # directory (not file)
                                   s_idx=self.config.dataset.s_idx,
                                   e_idx=self.config.dataset.e_idx,
                                   mode=self.config.prompt_mode) 

        else:
            raise ValueError(f"Invalid task name: {task}")
            
    def test_dataloader(self):
        return DataLoader(self.dataset,
                            collate_fn = lambda batch: batch, 
                            batch_size=self.config.dataset.batch_size,
                            num_workers=self.config.dataset.num_workers,
                            pin_memory=True,
                            drop_last=False) # ddp
    