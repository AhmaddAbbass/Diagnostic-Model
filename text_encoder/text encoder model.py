import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

class TextEncoder(nn.Module):
    def __init__(self, num_classes: int, base_model: str, from_scratch: bool = True):
        """
        Initialize the TextEncoder class.

        Parameters:
        - num_classes (int): The number of output classes (for classification).
        - base_model (str): The pretrained transformer model.
        - from_scratch (bool): Whether to train the model from scratch or load pretrained weights.
        """
        super(TextEncoder, self).__init__()

        # Load transformer model (ClinicalBERT in this case)
        if from_scratch:
            # Initialize from random weights
            cfg = AutoConfig.from_pretrained(base_model)
            self.bert = AutoModel.from_config(cfg)  # Random initialization of the BERT model
        else:
            # Load pretrained model
            self.bert = AutoModel.from_pretrained(base_model)

        # Get the hidden size from the BERT configuration
        hid = self.bert.config.hidden_size

        # Add classification head on top of the BERT model (a simple Linear layer)
        self.cls_head = nn.Linear(hid, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the model.

        Parameters:
        - input_ids (tensor): Input token IDs
        - attention_mask (tensor): Attention mask for padded tokens
        - labels (tensor, optional): Ground truth labels for training

        Returns:
        - logits (tensor): Class scores (logits)
        - loss (tensor, optional): Cross-entropy loss if labels are provided
        """
        # Pass input through BERT model
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        # Extract the output corresponding to the [CLS] token (first token of each sequence)
        cls_token_output = out.last_hidden_state[:, 0]  # [CLS] token is the first token

        # Pass the [CLS] token output through the classification head
        logits = self.cls_head(cls_token_output)

        # If labels are provided, calculate the loss (for training)
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
            return logits, loss

        # Return logits during inference
        return logits
