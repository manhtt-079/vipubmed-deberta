import torch
import torch.nn as nn
from transformers import AutoModel
from config.config import VIMqConfig


class NerModel(nn.Module):
    
    def __init__(self, config: VIMqConfig):
        super(NerModel, self).__init__()
        self.num_labels = config.num_labels

        self.model = AutoModel.from_pretrained(config.pretrained_model_name_or_path)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else self.model.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.model.config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.init_weights()
        
        self.loss_fct = nn.BCELoss()
    
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.classifier)
        self.classifier.bias.data.fill_(0.01)
    
    def forward(
        self,
        input_ids: torch.Tensor=None,
        attention_mask: torch.Tensor=None,
        token_type_ids: torch.Tensor=None,
        labels: torch.Tensor=None,
        return_dict: bool = False
    ):

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        output = (logits,) + outputs[2:]
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


        return ((loss,) + output) if loss is not None else output