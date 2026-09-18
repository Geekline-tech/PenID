import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    def __init__(self, margin: float = 0.3, lambda_cls: float = 0.5):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
        self.cls_loss = nn.CrossEntropyLoss()
        self.lambda_cls = lambda_cls

    def forward(
        self,
        emb_anchor: torch.Tensor,
        emb_pos: torch.Tensor,
        emb_neg: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_loss = self.triplet_loss(emb_anchor, emb_pos, emb_neg)
        c_loss = self.cls_loss(logits, labels)
        total = t_loss + self.lambda_cls * c_loss
        return total, t_loss, c_loss
