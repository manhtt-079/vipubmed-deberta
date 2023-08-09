import torch
import random
import os
import json
import copy
import re
import pandas as pd
from loguru import logger
from datasets import Dataset
from pyvi import ViTokenizer
from transformers import AutoTokenizer
from utils.utils import path_exists
from config.config import VIMedNLIConfig


class Processor(object):
    def __init__(self, config: VIMedNLIConfig) -> None:
        super().__init__()
        self.config = config

    @path_exists
    def read_tsv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep='\t', header=False)
        df.columns = ['text', 'label']
        
        def separate_premise_hypothesis(text: str):
            text = self.clean(text)
            text = text.removeprefix('sent1: ')
            idx = text.index('sent2: ')
            p, h = text[:idx].strip(), text[idx:].strip()

            return p, h
        
        temp = df['text'].map(lambda x: separate_premise_hypothesis(x))
        df['premise'] = [e[0] for e in temp]
        df['hypothesis'] = [e[1] for e in temp]
        
        return df[['premise', 'hypothesis', 'label']]
    
    def clean(self, text):
        text = re.sub(r'\[\*\s?\*?|\[(?=[0-9]{2,4})|\*?\s?\*\]|[\u001d\u0008*]|(?<=-)\/|\*\s\*\s-\s\[', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()
    
    def get_examples(self, mode: str):
        file_name = self.config.train_file_name if mode=='train' else self.config.dev_file_name if mode=='dev' else self.config.test_file_name
        data_path = os.path.join(self.config.data_dir, file_name)
        
        df = self.read_tsv(data_path)
        labels: list[str] = df['label'].unique().tolist()
        label2id = {v:k for k, v in enumerate(labels)}
        df['label'] = df['label'].map(lambda x: label2id[x])
        
        if self.config.re_tokenize:
            df['premise'] = df['premise'].map(lambda x: ViTokenizer.tokenize(x))
            df['hypotheis'] = df['hypotheis'].map(lambda x: ViTokenizer.tokenize(x))
            
        return df


class VIMedNLIDataset:
    def __init__(
        self,
        config: VIMedNLIConfig,
        tokenizer: AutoTokenizer
    ) -> None:
        self.config = config
        self.processor = Processor(config=config)
        self.tokenizer = tokenizer
    
    def get_dataset(self, mode: str):
        cached_features_file = os.path.join(
            self.config.data_dir,
            'cached_{}_{}_{}'.format(
                mode,
                list(filter(None, self.config.pretrained_model_name_or_path.split("/"))).pop(),
                self.config.max_seq_len
            ))
        
        if os.path.exists(cached_features_file):
            logger.info(f'Load cached features from: {cached_features_file}')
            return Dataset.load_from_disk(cached_features_file)
        else:

            examples = self.processor.get_examples(mode=mode)
            dataset = Dataset.from_pandas(examples)
            
            dataset =  dataset.map(
                function=self.preprocess_function,
                remove_columns=['premise', 'hypothesis'],
                batched=True,
                num_proc=self.config.num_proc
            )
            dataset.save_to_disk(cached_features_file)
            
            return dataset
        
    def preprocess_function(self, examples):
        inputs = self.tokenizer(examples['premise'], examples['hypotheis'], max_length=self.config.max_length, truncation=True, padding=True)
        
        return inputs
    
if __name__=='__main__':
    pass