from typing import List, Union

import timm
import torch
import torch.nn as nn


class TimmBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, features_only=False, **kwargs) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, features_only=features_only, **kwargs
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Forward computation."""
        return self.model(x)
