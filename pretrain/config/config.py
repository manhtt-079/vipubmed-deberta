import logging
import logging.config
from dataclasses import dataclass
from configparser import ConfigParser

config = ConfigParser()
config.read('./config/config.ini')

logging.config.fileConfig('./config/logger.ini')
logger = logging.getLogger()

def get_logger():
    return logger

@dataclass
class TokenizerConf:
    batch_size: int = int(config['tokenizer']['batch_size'])
    cache_dir: str = config['tokenizer']['cache_dir']
    column: str = config['tokenizer']['column']
    dataset_dir: str = config['tokenizer']['dataset_dir']
    min_frequency: int = int(config['tokenizer']['min_frequency'])
    show_progress: bool = True if config['tokenizer']['show_progress'].lower()=='true' else False
    vocab_size: int = int(config['tokenizer']['vocab_size'])
    pretrained_model_name: str = config['tokenizer']['pretrained_model_name']

@dataclass
class DatasetConf:
    column: str = config['dataset']['column']
    cache_dir: str = config['dataset']['cache_dir']
    data_dir: str = config['dataset']['data_dir']
    extension: str = config['dataset']['extension']
    n_processes: int = int(config['dataset']['n_processes'])
    threshold: float = float(config['dataset']['threshold'])
    dedup_file: str = config['dataset']['dedup_file']
    temp_column: str = config['dataset']['temp_column']
    seed: int = int(config['dataset']['seed'])
    test_size: int = int(config['dataset']['test_size'])
    split_dir: str = config['dataset']['split_dir']
    train_data: str = config['dataset']['train_data']
    val_data: str = config['dataset']['val_data']
    test_data: str = config['dataset']['test_data']
    
class TrainerConf(object):
    def __init__(self, sec: str):
        self.sec = 'trainer-' + sec
        
        self.data_seed: int = int(config['trainer']['data_seed'])
        self.dataloader_num_workers: int = int(config['trainer']['dataloader_num_workers'])
        self.do_train: bool = True if config['trainer']['do_train'].lower()=='true' else False
        self.do_eval: bool = True if config['trainer']['do_eval'].lower()=='true' else False
        self.evaluation_strategy: str =  config['trainer']['evaluation_strategy']
        self.eval_steps: int = int(config['trainer']['eval_steps'])
        self.fp16: bool = True if config['trainer']['fp16'].lower()=='true' else False
        self.gradient_accumulation_steps: int = int(config['trainer']['gradient_accumulation_steps'])
        self.learning_rate: float = float(config[self.sec]['learning_rate'])
        self.log_level: str = config['trainer']['log_level']
        self.logging_strategy: str = config['trainer']['logging_strategy']
        self.logging_steps: int = int(config['trainer']['logging_steps'])
        self.lr_scheduler_type: str = config['trainer']['lr_scheduler_type']
        self.max_steps: int = int(config[self.sec]['max_steps'])
        self.mlm_probability=float(config['trainer']['mlm_probability'])
        self.per_device_train_batch_size: int = int(config['trainer']['per_device_train_batch_size'])
        self.per_device_eval_batch_size: int = int(config['trainer']['per_device_eval_batch_size'])
        self.prediction_loss_only: bool = True if config['trainer']['prediction_loss_only'].lower()=='true' else False
        self.report_to: str = config['trainer']['report_to']
        self.save_strategy: str = config['trainer']['save_strategy']
        self.save_steps: int = int(config['trainer']['save_steps'])
        self.save_total_limit: int = int(config['trainer']['save_total_limit'])
        self.seed: int = int(config['trainer']['seed'])
        self.warmup_ratio: float = float(config['trainer']['warmup_ratio'])
        self.warmup_steps: int = int(config['trainer']['warmup_steps'])
        self.weight_decay: float = float(config['trainer']['weight_decay'])
        self.tokenizer_path: str = config['trainer']['tokenizer_path']
        self.max_steps = int(config[self.sec].get('max_steps', self.max_steps))
        self.learning_rate = float(config[self.sec].get('learning_rate', self.learning_rate))
        self.logging_dir = config[self.sec]['logging_dir']
        self.output_dir = config[self.sec]['output_dir']
        self.run_name = config[self.sec]['run_name']
        self.save_steps = int(config[self.sec].get('save_steps', self.save_steps))
        self.pretrained_model_name = config[self.sec]['pretrained_model_name']
        
if __name__=='__main__':
    pass