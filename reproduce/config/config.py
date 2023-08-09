from dataclasses import dataclass

@dataclass
class BaseConfig:
    data_dir: str
    checkpoint: str
    num_labels: int
    epochs: int
    seed: int = 42
    classifier_dropout: float = 0.2
    max_length: int = 512
    num_proc: int = 4
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight', 'layernorm_embedding.weight', 'final_layer_norm.weight', 'self_attn_layer_norm.weight']
    lr: float = 3e-5
    adam_epsilon: float = 1e-9
    weight_decay: float = 0.015
    warmup_ratio: float = 0.05
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    max_steps: int = 0
    gradient_accumulation_steps: int = 1
    logging_steps: int = 500
    eval_steps: int = 1000
    save_steps: int = 500
    re_tokenize: bool = True
    max_grad_norm: float = 1.0
    patience: int = 10
    pretrained_model_name_or_path: str = 'manhtt-079/vipubmed-deberta-base'
    
    def __post_init__(self):
        if 'deberta' not in self.pretrained_model_name_or_path:
            self.max_length = 256
            self.re_tokenize = False


@dataclass
class WsdConfig(BaseConfig):
    data_dir: str = '../dataset/wsd/'
    file_name: str = 'data.json'
    wsd_map: str = '../dataset/wsd/dictionary.json'
    checkpoint: str = './code/wsd/checkpoint/'
    epochs: int = 100
    intermediate_size: int = 128
    num_labels: int = 1
    lr: float = 3e-5
    patience: int = 100
    threshold: float = 0.5
    tuning_metric: str = 'macro-f1'
    gradient_accumulation_steps: int = 1
    pretrained_model_name_or_path: str = 'manhtt-079/vipubmed-deberta-base'

@dataclass
class VIMedNLIConfig(BaseConfig):
    data_dir: str = '../dataset/vimednli'
    train_file_name: str = 'train_vi_refined.tsv'
    dev_file_name: str = 'dev_vi_refined.tsv'
    test_file_name: str = 'test_vi_refined.tsv'
    checkpoint: str = './vimed/checkpoint/'
    epochs: int = 15
    num_labels: int = 2
    lr: float = 2e-5
    gradient_accumulation_steps: int = 1
    pretrained_model_name_or_path: str = 'manhtt-079/vipubmed-deberta-base'
    
    
@dataclass
class VIMqConfig(BaseConfig):
    data_dir: str = '../dataset/vimednli'
    train_file_name: str = 'train.json'
    dev_file_name: str = 'dev.json'
    test_file_name: str = 'test.json'
    label2id: dict[str] = {'method_diagnosis': 0, 'treatment': 1, 'severity': 2, 'cause': 3}
    id2label: dict[str] = {0: 'method_diagnosis', 1: 'treatment', 2: 'severity', 3: 'cause'}
    checkpoint: str = './vimed/checkpoint/'
    epochs: int = 15
    num_labels: int = 4
    lr: float = 2e-5
    gradient_accumulation_steps: int = 1
    pretrained_model_name_or_path: str = 'manhtt-079/vipubmed-deberta-base'

@dataclass
class CovIdConfig(BaseConfig):
    data_dir: str = '../dataset/vimednli'
    train_file_name: str = 'train_vi_refined.tsv'
    dev_file_name: str = 'dev_vi_refined.tsv'
    test_file_name: str = 'test_vi_refined.tsv'
    checkpoint: str = './vimed/checkpoint/'
    epochs: int = 15
    num_labels: int = 2
    lr: float = 2e-5
    gradient_accumulation_steps: int = 1
    pretrained_model_name_or_path: str = 'manhtt-079/vipubmed-deberta-base'


if __name__=='__main__':
    pass
