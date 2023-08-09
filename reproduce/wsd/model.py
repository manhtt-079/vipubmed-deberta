import torch
import torch.nn as nn
from transformers import AutoModel
from config.config import WsdConfig


class WsdModel(nn.Module):
    
    def __init__(self, config: WsdConfig):
        super(WsdModel, self).__init__()
        self.config = config
        self.intermediate_size = self.config.intermediate_size
        self.model = AutoModel.from_pretrained(self.config.pretrained_model_name_or_path)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.intermediate_size),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(self.intermediate_size, self.config.num_labels),
            nn.Sigmoid()
        )
        
        self.loss_func = nn.BCELoss()
        
    def forward(
        self,
        input_ids: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
        token_type_ids: torch.Tensor=None,
        start_token_idx: torch.Tensor=None,
        end_token_idx: torch.Tensor=None,
        labels: torch.Tensor=None,
        return_dict: bool = False
    ):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict
        )
        last_hidden_state = outputs[0]
        # Features of [CLS] tokens
        cls_features = last_hidden_state[:, 0]
        cls_features = self.dropout(cls_features)

        if start_token_idx is None or end_token_idx is None:
            raise Exception('Require `start_token_idx` and `end_token_idx`')
        
        # Get acronym features for each token and concatenate them
        acr_features = torch.cat(
            [torch.mean(last_hidden_state[idx, start_token_idx[idx]:end_token_idx[idx]], dim=0, keepdim=True)
             for idx in range(last_hidden_state.size()[0])
             ], dim=0
        )

        # Concate features: [cls, acr]
        features = torch.cat([cls_features, acr_features], dim=1)
        logits = self.classifier(features).view(-1)
        
        outputs = (logits, ), + outputs[2:]

        loss = None
        if labels is not None:
            loss = self.loss_func(logits, labels)
        
        return (loss,) + outputs