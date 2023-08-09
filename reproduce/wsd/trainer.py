import torch
from loguru import logger
from tqdm.auto import tqdm, trange
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, DataCollatorWithPadding
from utils.utils import EarlyStopping, compute_wsd_metrics
from config.config import WsdConfig
from wsd.dataloader import WsdDataset
from wsd.model import WsdModel

class Trainer(object):
    def __init__(self, config: WsdConfig) -> None:
        super().__init__()
        self.config = config
        self.model = WsdModel(config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name_or_path)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        self.model.to(self.device)

        self.wsd_dataset = WsdDataset(config=self.config, tokenizer=self.tokenizer)
        self.train_loader = self.get_loader(mode='train')
        self.dev_loader = self.get_loader(mode='dev')
        self.test_loader = self.get_loader(mode='test')

    def get_loader(self, mode: str):
        if mode not in ['train', 'dev', 'test']:
            raise ValueError(f'mode variable must be in: ["train", "dev", "test"]')
        dataset = self.wsd_dataset.get_dataset(mode=mode)
        if mode == 'train':
            sampler = RandomSampler(dataset)
            batch_size = self.config.per_device_train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = self.config.per_device_eval_batch_size
            
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer,  return_tensors="pt")
        dataloader = DataLoader(
            dataset=self.train_dataset, 
            sampler=sampler, 
            collate_fn=data_collator,
            batch_size=batch_size
        )
        
        return dataloader
    
    
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
    
    @staticmethod
    def get_model_param_count(model, trainable_only: bool=True):
        return sum(p.numel() for p in model.parameters() if not trainable_only or p.requires_grad)
    
    def train(self):
        if self.config.max_steps > 0:
            max_steps = self.config.max_steps
            self.config.epochs = (
                self.config.max_steps // (len(self.train_loader) // self.config.gradient_accumulation_steps) + 1
            )
        else:
            max_steps = len(self.train_loader) // self.config.gradient_accumulation_steps * self.config.epochs
        
        optimizer, scheduler = self.configure_optimizer_scheduler(num_training_steps=max_steps)

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_loader*self.config.per_device_train_batch_size):,}")
        logger.info(f"  Num Epochs = {self.config.epochs:,}")
        logger.info(f"  Total train batch size = {self.config.per_device_train_batch_size*self.config.gradient_accumulation_steps:,}")
        logger.info(f"  Gradient Accumulation steps = %d", self.config.gradient_accumulation_steps)
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Logging steps = {self.config.logging_steps:,}")
        logger.info(f"  Save steps = {self.config.save_steps:,}")
        logger.info(f"  Number of trainable params: {self.get_model_param_count(self.model, trainable_only=True):,}")

        n_steps = 0
        train_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.config.num_train_epochs), desc="Epoch")
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)

        for it in train_iterator:
            logger.info(f'----:---- Epoch: {it} ----:----')
            epoch_iterator = tqdm(self.train_loader, desc="Iter", position=0, leave=True)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs[0]

                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps

                loss.backward()
                train_loss += loss.item()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    n_steps += 1

                if self.config.logging_steps > 0 and n_steps % self.config.logging_steps == 0:
                    eval_loss, results = self.evaluate('dev')
                    results['loss'] = eval_loss
                    
                    print('\n\n')
                    logger.info("***** Eval results *****")
                    logger.info(results)

                    early_stopping(results[self.config.tuning_metric])
                    if early_stopping.early_stop:
                        break
                    
                    if early_stopping.is_save:
                        self.model.save_pretrained(self.config.checkpoint)
                
                if 0 < self.config.max_steps < n_steps:
                    epoch_iterator.close()
                    break
            
            if 0 < self.config.max_steps < n_steps or early_stopping.early_stop:
                train_iterator.close()
                break
        
        print('\n\n')
        logger.info("***** Running testing *****")
        logger.info("  Num examples = %d", len(self.test_loader)*self.config.per_device_eval_batch_size)
        logger.info("  Batch size = %d", self.config.eval_batch_size)
        test_loss, test_results = self.evaluate(self.test_loader)
        test_results['loss'] = test_loss
        logger.info("***** Eval results *****")
        logger.info(test_results)
        
        return n_steps, train_loss / n_steps

    def evaluate(self, dataloader: DataLoader):
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(dataloader)*self.config.per_device_eval_batch_size)
        logger.info("  Batch size = %d", self.config.eval_batch_size)
        
        self.model.eval()
        all_preds, all_labels = [], []
        n_steps, total_loss = 0, 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, 'Eval'):
                batch.to(self.device)
                
                outputs = self.model(**batch)
                loss, logits = outputs[:2]
                total_loss += loss.item()

                label = batch['label']
                
                pred = torch.where(logits > self.config.threshold, 1, 0)
                all_preds.extend(pred.detach().cpu())
                all_labels.extend(label.detach().cpu())

            n_step += 1
        
        total_loss = total_loss / n_steps
        results = compute_wsd_metrics(all_preds=all_preds, all_labels=all_labels)

        return total_loss, results
    
if __name__ == '__main':
    pass