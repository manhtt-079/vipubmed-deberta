import random
import os
import json
import copy
import re
from typing import Any
from loguru import logger
from pyvi import ViTokenizer
from datasets import Dataset
from transformers import AutoTokenizer
from utils.utils import path_exists
from config.config import WsdConfig

class InputExample(object):
    def __init__(
        self,
        guid: int|str = None,
        prefix: str = None,
        suffix: str = None,
        acronym: str = None,
        expansion: str = None,
        label: int = None
    ) -> None:
        super().__init__()
        
        self.guid = guid
        self.prefix = prefix
        self.suffix = suffix
        self.acronym = acronym
        self.expansion = expansion
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class Processor(object):

    def __init__(self, config: WsdConfig) -> None:
        super().__init__()
        self.config = config
        self.wsd_map = self.read_json(self.config.wsd_map)

    @path_exists
    def read_json(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            raise RuntimeError(f'Error when reading: {path}')
    
    @staticmethod
    def clean(text: str):
        text = re.sub(r'[\n\t\u202f]', '', text)
        
        return text.strip()
    

    def neg_sampling(
        self,
        pos_data: list[InputExample],
        mode: str
    ) -> list:

        neg_data = []
        
        for e in pos_data:
            neg_expansions = self.wsd_map[e.acronym].copy()
            neg_expansions.remove(e.expansion)
            if mode=='train':
                random.seed(self.config.seed)
                neg_expansions = random.sample(neg_expansions, k=random.randint(1,3))
            
            for expansion in neg_expansions:
                neg_data.append(InputExample(
                    guid=e.guid,
                    prefix=e.prefix,
                    suffix=e.suffix,
                    acronym=e.acronym,
                    expansion=expansion,
                    label=0
                ))
        
        return neg_data
    
    def to_examples(
        self,
        json_data: list[dict[str, Any]],
        mode: str = 'train'
    ) -> list[InputExample]:
        
        pos_data: list[InputExample] = []
        for idx, e in enumerate(json_data):
            start_idx = e['start_char_idx']
            end_idx = e['start_char_idx'] + e['length_acronym']
            
            prefix = e['text'][:start_idx]
            acronym = e['text'][start_idx:end_idx]
            suffix = e['text'][end_idx:]
            
            if self.config.re_tokenize:
                prefix=ViTokenizer.tokenize(self.clean(prefix))
                suffix=ViTokenizer.tokenize(self.clean(suffix))
                
            pos_data.append(InputExample(
                guid=idx,
                prefix=prefix,
                suffix=prefix,
                acronym=acronym,
                expansion=e['expansion'],
                label=1
            ))
            
        neg_data = self.neg_sampling(pos_data=pos_data, mode=mode)
        logger.info(f'Train-set ---:--- Num pos/neg samples: {len(pos_data)}/{len(neg_data)}')
            
        return pos_data + neg_data


    def get_examples(self, mode: str):

        data_path = os.path.join(self.config.data_dir, mode, self.config.file_name)
        data = self.read_json(data_path)
        
        return self.to_examples(json_data=data)

class WsdDataset:
    def __init__(self, config: WsdConfig, tokenizer: AutoTokenizer) -> None:
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

            examples: list[dict[str, str]] = [e.to_dict() for e in self.processor.get_examples(mode=mode)]
            dataset = Dataset.from_list(examples)
            
            dataset =  dataset.map(
                function=self.preprocess_function,
                remove_columns=['guid', 'prefix', 'suffix', 'acronym', 'expansion'],
                batched=True,
                num_proc=self.config.num_proc
            )
            dataset.save_to_disk(cached_features_file)
            
            return dataset
        
    def preprocess_function(self, example):
        if self.config.max_seq_len==512:
            prefix_max_len, suffix_max_len, acronym_max_len, expansion_max_len = 240, 240, 16, 16
        else:
            prefix_max_len, suffix_max_len, acronym_max_len, expansion_max_len = 120, 120, 8, 8
        
        prefix_seqs = self.tokenizer.encode(example['prefix'], add_special_tokens=False, max_length=prefix_max_len, truncation=True, padding=True)
        suffix_seqs = self.tokenizer.encode(example['suffix'], add_special_tokens=False, max_length=suffix_max_len, truncation=True, padding=True)
        acronym_seqs = self.tokenizer.encode(example['acronym'], add_special_tokens=False, max_length=acronym_max_len, truncation=True, padding=True)
        expansion_seqs = self.tokenizer.encode(example['expansion'], add_special_tokens=False, max_length=expansion_max_len, truncation=True, padding=True)

        # input_ids: [cls_token_id] + [prefix_token_ids] + [acronym_token_ids] + [suffix_token_ids] + [sep_token_ids] + [expansion_token_ids] + [sep_token_ids]
        input_ids = [self.tokenizer.cls_token_id] + prefix_seqs
        start_idx = len(input_ids)
        end_idx = start_idx + len(acronym_seqs)
        input_ids += acronym_seqs + suffix_seqs + [self.tokenizer.sep_token_id]

        s2 = expansion_seqs + [self.tokenizer.sep_token_id]
        token_type_ids = [0]*len(input_ids) + [1]*len(s2)
        input_ids += s2
        attention_mask = [1]*len(input_ids)


        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'start_token_idx': start_idx,
            'end_token_idx': end_idx
        }
