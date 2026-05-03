
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor]
    cls_loss: Optional[torch.Tensor]
    att_loss: Optional[torch.Tensor]
    logits: torch.Tensor
    attention_scores: torch.Tensor


class BertAttentionClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 3, att_lambda: float = 0.5):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, output_attentions=True)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(getattr(self.encoder.config, "hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(hidden, num_labels)
        self.ce_loss = nn.CrossEntropyLoss()
        self.att_lambda = att_lambda

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        rationale_targets: torch.Tensor | None = None,
    ) -> ModelOutput:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(pooled))

        last_layer_att = outputs.attentions[-1]              # [B, H, S, S]
        cls_attention = last_layer_att[:, :, 0, :].mean(1)   # [B, S]
        cls_attention = cls_attention * attention_mask.float()
        cls_attention = cls_attention / (cls_attention.sum(dim=1, keepdim=True) + 1e-12)

        loss = None
        cls_loss = None
        att_loss = None

        if labels is not None:
            cls_loss = self.ce_loss(logits, labels)
            loss = cls_loss

        if rationale_targets is not None:
            rt = rationale_targets.float() * attention_mask.float()
            has_any = (rt.sum(dim=1) > 0)
            if has_any.any():
                rt = rt / (rt.sum(dim=1, keepdim=True) + 1e-12)
                pred = torch.clamp(cls_attention, min=1e-12)
                token_ce = -(rt * torch.log(pred)).sum(dim=1)
                att_loss = token_ce[has_any].mean()
                if loss is None:
                    loss = self.att_lambda * att_loss
                else:
                    loss = loss + self.att_lambda * att_loss

        return ModelOutput(
            loss=loss,
            cls_loss=cls_loss,
            att_loss=att_loss,
            logits=logits,
            attention_scores=cls_attention,
        )
