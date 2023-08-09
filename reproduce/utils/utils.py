import os
import torch
import random
import numpy as np
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


MODEL_ARCHIVE_MAP = {
    'phobert-base': 'vinai/phobert-base',
    'phobert-large': 'vinai/phobert-large',
    'vihealth-bert': 'demdecuong/vihealthbert-base-word',
    'videberta': 'Fsoft-AIC/videberta-base',
    'vipubmed-deberta-xsmall': 'manhtt-079/vipubmed-deberta-xsmall',
    'vipubmed-deberta-base': 'manhtt-079/vipubmed-deberta-base'
}

def path_exists(func):
    def inner(*args):
        if not os.path.exists(args[1]):
            raise FileNotFoundError(f'{args[1]}: does not exist!')
        return func(*args)

    return inner

def compute_metrics(all_preds: list[int], all_labels: list[int]):
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'mac-precision': precision_score(all_labels, all_preds, average='macro'),
        'mac-recall': recall_score(all_labels, all_preds, average='macro'),
        'macro-f1': f1_score(all_labels, all_preds, average='macro')
    }

def seed_everything(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f'Set seed: {seed}.')


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7,  delta=1e-6, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.is_save = True

    def __call__(self, score: float):
        
        if self.best_score is None:
            self.best_score = score
            self.is_save = True
            
        elif score >= self.best_score + self.delta:
            if self.verbose:
                logger.info(f"Dev-score improved from: 〈{self.best_score} 🠖 {score}〉. Saving model...")
                
            self.best_score = score
            self.counter = 0
            self.is_save = True
            
        else:
            self.counter += 1
            logger.infor(f"Early stopping counter: {self.counter} out of {self.patience}.")
            if self.counter >= self.patience:
                logger.info(f"Reached the max patience: {self.patience}. Early stopping.")
                self.early_stop = True

if __name__ == '__main__':
    pass