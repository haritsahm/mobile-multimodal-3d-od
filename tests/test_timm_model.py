import numpy as np
import pytest
import torch

from src.models.backbones.timm_model import TimmBackbone

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


@pytest.mark.parametrize(
    "batch_size,input_shape", [(2, [256, 256]), (4, [384, 384]), (2, [640, 640])]
)
def test_timm_backbone(batch_size, input_shape):
    input_shape = np.array(input_shape)
    x = torch.rand(batch_size, 3, *input_shape.tolist(), device=DEVICE)

    model = TimmBackbone(model_name="repvit_m1", pretrained=True)
    model.to(DEVICE)

    output = model(x)

    # Check training output
    assert output.shape == (batch_size, model.num_classes)


@pytest.mark.parametrize(
    "batch_size,input_shape", [(2, [256, 256]), (4, [384, 384]), (2, [640, 640])]
)
def test_timm_backbone_features(batch_size, input_shape):
    input_shape = np.array(input_shape)
    x = torch.rand(batch_size, 3, *input_shape.tolist(), device=DEVICE)

    model = TimmBackbone(
        model_name="repvit_m1", pretrained=True, features_only=True, out_indicies=(1, 2, 3, 4)
    )
    model.to(DEVICE)

    output = model(x)

    # Check training output
    assert len(output) == 4
    for i, o in enumerate(output):
        assert list(o.shape[2:]) == list(input_shape // (2 ** (i + 2)))
