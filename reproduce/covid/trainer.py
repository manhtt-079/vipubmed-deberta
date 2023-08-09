import logging
import os
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader

from torch.utils.tensorboard import SummaryWriter

from transformers import AdamW, get_linear_schedule_with_warmup

from early_stopping import EarlyStopping
from utils import compute_metrics
from utils import read_json

from reproduce.code.config.wsd_conf import WsdConfig
from covid19.model import NerModel


logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, config: WsdConfig) -> None:
        super().__init__()
        self.config = config
        self.model = NerModel(config=self.config)
        # GPU or CPU
        torch.cuda.set_device(self.args.gpu_id)
        print('GPU ID :',self.args.gpu_id)
        print('Cuda device:',torch.cuda.current_device())
        # self.device = args.device
        
        self.train_dataset = self.get_dataset()

        self.model.to(self.device)

    def get_dataset(self):
        pass
    
    def configure_optimizer(self) -> torch.optim.Optimizer:
        param_optimizer = [[name, param] for name, param in self.model.named_parameters() if param.requires_grad]
        optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer if not any(nd in name for nd in self.config.no_decay)],
            'weight_decay': self.config.weight_decay},
            {'params': [param for name, param in param_optimizer if any(nd in name for nd in self.config.no_decay)],
            'weight_decay': 0.0}
        ]
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            params=optimizer_grouped_parameters, 
            lr=self.config.lr, 
            betas=(0.9, 0.98), 
            eps=self.config.adam_epsilon
        )
        
        return optimizer
    
    def configure_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int):
        scheduler: torch.optim.lr_scheduler.LambdaLR = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config.warmup_ratio*num_training_steps,
            num_training_steps=num_training_steps
        )

        return scheduler
    
    def configure_optimizer_scheduler(self, num_training_steps: int):
        optimizer: torch.optim.Optimizer = self.configure_optimizer()
        scheduler: torch.optim.lr_scheduler.LambdaLR = self.configure_scheduler(optimizer=optimizer, num_training_steps=num_training_steps)
        
        return optimizer, scheduler
    
    
    
    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_loader = DataLoader(dataset=self.train_dataset, sampler=train_sampler, batch_size=self.config.per_device_train_batch_size)

        if self.config.max_steps > 0:
            max_steps = self.config.max_steps
            self.config.epochs = (
                self.config.max_steps // (len(train_loader) // self.config.gradient_accumulation_steps) + 1
            )
        else:
            max_steps = len(train_loader) // self.config.gradient_accumulation_steps * self.config.epochs
        
        
        optimizer, scheduler = self.configure_optimizer_scheduler(num_training_steps=max_steps)


        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset):,}")
        logger.info(f"  Num Epochs = {self.config.epochs:,}")
        logger.info(f"  Total train batch size = {self.config.per_device_train_batch_size*self.config.gradient_accumulation_steps:,}")
        logger.info(f"  Gradient Accumulation steps = %d", self.config.gradient_accumulation_steps)
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Logging steps = %d", self.config.logging_steps)
        logger.info(f"  Save steps = %d", self.config.save_steps)

        global_step = 0
        tr_loss = 0.0
        # self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)

        for it in train_iterator:
            print(f'epoch: {it}')
            epoch_iterator = tqdm(train_loader, desc="Iteration", position=0, leave=True)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                # batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                    "input_ids": batch[0].to(self.device),
                    "token_type_ids": batch[1].to(self.device),
                    "attention_mask": batch[2].to(self.device),
                    "start_token_idx": batch[3].to(self.device),
                    "end_token_idx": batch[4].to(self.device),
                    "labels": batch[5].to(self.device)
                }

                optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss, _ = outputs[:2]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        print("\nTuning metrics:", self.args.tuning_metric)
                        eval_loss, ids, pred_expansions, pred_scores = self.evaluate('test')
                        results = compute_metrics(self.args, ids, pred_expansions, pred_scores)
                        results['loss'] = eval_loss
                        logger.info("***** Eval results *****")
                        for key in sorted(results.keys()):
                            logger.info("  %s = %s", key, str(results[key]))

                        early_stopping(results[self.args.tuning_metric], self.model, self.args)
                        if early_stopping.early_stop:
                            print('Early Stopping')
                            break
                            
                        print(f'Training Loss {tr_loss / global_step}')
                    
                    # if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    #     self.save_model()
                
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break
            if 0 < self.args.max_steps < global_step or early_stopping.early_stop:
                train_iterator.close()
                break
        
        return global_step, tr_loss / global_step
    
    def write_evaluation_result(self, out_file, results):
        out_file = self.args.model_dir + "/" + out_file
        w = open(out_file, "w", encoding="utf-8")
        w.write("***** Eval results *****\n")
        for key in sorted(results.keys()):
            to_write = " {key} = {value}".format(key=key, value=str(results[key]))
            w.write(to_write)
            w.write("\n")
        w.close()

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")
        
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        ids = []
        pred_expansions = []
        pred_scores = []

        eval_loss = 0.0
        nb_eval_steps = 0
        correct = 0

        self.model.eval()
        

        for batch in eval_dataloader:
            # batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0].to(self.device),
                    "token_type_ids": batch[1].to(self.device),
                    "attention_mask": batch[2].to(self.device),
                    "start_token_idx": batch[3].to(self.device),
                    "end_token_idx": batch[4].to(self.device),
                    "labels": batch[5].to(self.device)
                }
                outputs = self.model(**inputs)
                loss, logits = outputs[:2]
                eval_loss += loss.item()

                labels = inputs['labels']
                preds = (logits > self.args.threshold).type(torch.int16)
                correct += sum(preds == labels).item()

                ids.extend(batch[6].tolist())
                pred_expansions.extend(list(batch[7]))
                
                pred_scores.extend(logits.tolist())

            nb_eval_steps += 1
        
        eval_loss = eval_loss / nb_eval_steps
        acc = correct/len(dataset)
        print(f'Classification Accuracy: {acc}')

        return eval_loss, ids, pred_expansions, pred_scores
    
    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(
                self.args.model_dir,
                config=self.config,
                args=self.args
                
            )
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except Exception:
            raise Exception("Some model files might be missing...")