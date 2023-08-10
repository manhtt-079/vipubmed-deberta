from dataclasses import dataclass

@dataclass
class WsdConfig:
    data_dir: str = './dataset/'
    file_name: str = 'data.json'
    wsd_map: str = './dataset/dictionary.json'
    checkpoint: str = './checkpoint/'
    epochs: int = 100
    intermediate_size: int = 128
    num_labels: int = 1
    lr: float = 3e-5
    patience: int = 25
    threshold: float = 0.5
    tuning_metric: str = 'macro-f1'
    seed: int = 42
    classifier_dropout: float = 0.2
    max_length: int = 512
    num_proc: int = 4
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight', 'layernorm_embedding.weight', 'final_layer_norm.weight', 'self_attn_layer_norm.weight']
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
    pretrained_model_name_or_path: str = 'manhtt-079/vipubmed-deberta-base'
    
    def __post_init__(self):
        if 'deberta' not in self.pretrained_model_name_or_path:
            self.max_length = 256
            self.re_tokenize = False
            

if __name__=='__main__':
    pass