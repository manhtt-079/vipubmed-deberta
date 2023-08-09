import datasets
from tqdm import tqdm
from transformers import AutoTokenizer
from config.config import TokenizerConf
from config.config import get_logger

logger = get_logger()
class Tokenizer(object):
    def __init__(self, conf: TokenizerConf) -> None:
        self.conf = conf
        logger.info(f'Loading dataset: {self.conf.dataset_dir}')
        self.dataset = datasets.load_from_disk(self.conf.dataset_dir)['train']
        logger.info(self.dataset)
        logger.info(f'Loading checkpoint: {self.conf.pretrained_model_name}')
        self.tokenizer = AutoTokenizer.from_pretrained(self.conf.pretrained_model_name)
    
    def batch_iterator(self):
        for i in tqdm(range(0, len(self.dataset), self.conf.batch_size)):
            yield self.dataset[i: i+self.conf.batch_size][self.conf.column]
            
    def train(self):
        tokenizer = self.tokenizer.train_new_from_iterator(
            text_iterator=self.batch_iterator(),
            vocab_size=self.conf.vocab_size,
            min_frequency=self.conf.min_frequency,
            show_progress=self.conf.show_progress
        )

        logger.info(f"Save pretrained tokenizer to: {self.conf.cache_dir}")
        tokenizer.save_pretrained(self.conf.cache_dir)
        
if __name__ == '__main__':
    pass