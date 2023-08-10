from dataclasses import dataclass

@dataclass
class VIMedNLIConfig:
    data_dir: str = '../dataset/vimednli'
    train_file: str = 'train_vi_refined.tsv'
    dev_file: str = 'dev_vi_refined.tsv'
    test_file: str = 'test_vi_refined.tsv'
    checkpoint: str = './vimed/checkpoint/'
    num_labels: int = 3
    label2id: dict[str, int] = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    id2label: dict[int, str] = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    epochs: int = 15
    seed: int = 42
    classifier_dropout: float = 0.2
    max_length: int = 512
    num_proc: int = 4
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight', 'layernorm_embedding.weight', 'final_layer_norm.weight', 'self_attn_layer_norm.weight']
    lr: float = 2e-5
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
    
if __name__=='__main__':
    pass