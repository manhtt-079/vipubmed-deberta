import os
import pandas as pd
from pyvi import ViTokenizer
from sklearn.model_selection import train_test_split
from typing import Set
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import datasets

from config.config import DatasetConf, get_logger

logger = get_logger()
class DatasetPipeLine(object):
    def __init__(self, conf: DatasetConf) -> None:
        self.conf = conf
        self.duplicates = None
        self.df = None
        self.setup()
    
    def setup(self):
        for p in [self.conf.cache_dir, self.conf.split_dir]:
            self.make_dir(p)
        return True
    
    @staticmethod
    def make_dir(dir_path: str):
        if not os.path.exists(dir_path):
            os.system(f'mkdir -p {dir_path}')
            os.system(f'chmod -R 777 {dir_path}')

    
    @property
    def file_paths(
        self,
    ):
        """Get all file paths under the specified directory

        Returns:
            List[str]: _description_
        """
        return [os.path.join(self.conf.data_dir, p) for p in os.listdir(self.conf.data_dir)
                if p.endswith(self.conf.extension) and p != self.conf.dedup_file]
    
    @staticmethod
    def jaccard_sim(
        docA: Set[str],
        docB: Set[str],
        threshold: float
    ) -> bool:
        return len(docA & docB) / len(docA | docB) > threshold
    
    def duplicated(self):
        self.df[self.conf.temp_column] = self.df[self.conf.column].apply(lambda x: tuple(set(x.lower().split())))

        duplicates = self.df.duplicated(subset=self.conf.temp_column)
        self.duplicates = duplicates
        logger.info(f'No. indentical duplicated records: {self.duplicates.sum()}')
        
        self.df[self.conf.temp_column] = self.df.comp.apply(lambda x: set(x))

        processor = partial(self.get_candidate)
        pool = Pool(self.conf.n_processes)
        candidates = []
        for c in tqdm(
            pool.map(processor, range(len(self.df.index))),
            desc='Data dedupping',
            total=len(self.df),
            ncols=100
        ):
            candidates.append(c)

        return candidates
    
    def get_candidate(self, idx: int):
        if self.duplicates[idx]:
            return pd.Series(dtype=object)
            
        subdf = self.df[idx+1:]
        subdup = self.duplicates[idx+1:]
        
        base = self.df.loc[self.df.index[idx], 'comp']
        candidate_indices = subdup.loc[~subdup].index
        candidates: pd.Series = (subdf.loc[subdf.index.isin(candidate_indices), 'comp'].apply(lambda x: self.jaccard_sim(base, x, threshold=self.conf.threshold)))
        
        return candidates

    def run(self):
        logger.info(f'Gather all *{self.conf.extension} under the directory: {self.conf.data_dir}')
        logger.info(self.file_paths)
        self.df = pd.concat([pd.read_csv(p) for p in self.file_paths])
        self.df.reset_index(drop=True, inplace=True)
        
        duplicates = self.duplicated()
        n = self.duplicates.sum()
        for e in duplicates:
            self.duplicates = self.duplicates | e

        logger.info(f'No. Jaccard duplicated records: {self.duplicates.sum()-n}')
        self.df = self.df.loc[~self.duplicates][[self.conf.column]]
        self.save_deduped_data()
    
    def create_dataset(self):
        logger.info('Splitting dataset')
        train_set, test_set = train_test_split(
            self.df,
            test_size=self.conf.test_size,
            random_state=self.conf.seed,
            shuffle=True
        )
        train_set, val_set = train_test_split(
            train_set,
            test_size=self.conf.test_size
        )
        
        train_set.to_csv(os.path.join(self.conf.split_dir, self.conf.train_data), index=False)
        val_set.to_csv(os.path.join(self.conf.split_dir, self.conf.val_data), index=False)
        test_set.to_csv(os.path.join(self.conf.split_dir, self.conf.test_data), index=False)
    
    def tokenize(self, example):
        example[self.conf.column] = ViTokenizer.tokenize(example['vi'])

        return example
        
    def pre_tokenizer(self):
        logger.info('Pre-tokenize dataset')
        data_files = {'train': self.conf.train_data, 'val': self.conf.val_data, 'test': self.conf.test_data}
        dataset = datasets.load_dataset(self.conf.split_dir, data_files=data_files, cache_dir=self.conf.cache_dir)
        
        logger.info(dataset)
        dataset = dataset.map(self.tokenize, batch_size=True, num_proc=self.conf.n_processes)
        dataset.save_to_disk(self.conf.cache_dir)
        
    
    def save_deduped_data(self):
        """Storage de-duplicated data
        """
        self.df[[self.conf.column]].to_csv(os.path.join(self.conf.data_dir, self.conf.dedup_file), index=False)


if __name__=='__main__':
    pass