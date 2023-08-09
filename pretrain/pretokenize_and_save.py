import os
import pandas as pd
from pyvi import ViTokenizer
import datasets
from sklearn.model_selection import train_test_split
from config.config import DatasetConf

conf = DatasetConf()

def tokenize(example):
    example[conf.column] = ViTokenizer.tokenize(example['vi'])

    return example
    
data_files = {'train': conf.train_data, 'val': conf.val_data, 'test': conf.test_data}
dataset = datasets.load_dataset(conf.split_dir, data_files=data_files, cache_dir=conf.cache_dir)

dataset = dataset.map(tokenize, batch_size=True, num_proc=4)
dataset.save_to_disk(conf.cache_dir)


