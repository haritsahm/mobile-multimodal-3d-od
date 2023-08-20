import numpy as np
import pytest
import torch
import timm

from src.models.backbones.fastvit import fastvit_t8
from src.models.backbones.timm_model import TimmBackbone

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")


@pytest.mark.parametrize(
    "batch_size,input_shape", [(2, [256, 256]), (4, [384, 384]), (2, [640, 640])]
)
def test_fastvit_t8(batch_size, input_shape):
    input_shape = np.array(input_shape)
    x = torch.rand(batch_size, 3, *input_shape.tolist(), device=DEVICE)

    model = fastvit_t8()
    model.to(DEVICE)

    output = model(x)

    # Check training output
    assert output.shape == (batch_size, model.num_classes)


@pytest.mark.parametrize(
    "batch_size,input_shape", [(2, [256, 256]), (4, [384, 384]), (2, [640, 640])]
)
def test_fastvit_t8_features(batch_size, input_shape):
    input_shape = np.array(input_shape)
    x = torch.rand(batch_size, 3, *input_shape.tolist(), device=DEVICE)

    model = fastvit_t8(fork_feat=True)
    model.to(DEVICE)

    output = model(x)

    # Check training output
    assert len(output) == 4
    for i, o in enumerate(output):
        assert list(o.shape[2:]) == list(input_shape // (2 ** (i + 2)))

@pytest.mark.parametrize(
    "batch_size,input_shape", [(2, [256, 256]), (4, [384, 384]), (2, [640, 640])]
)
def test_fastvit_as_timm(batch_size, input_shape):
    input_shape = np.array(input_shape)
    x = torch.rand(batch_size, 3, *input_shape.tolist(), device=DEVICE)

    model = TimmBackbone(
        model_name="fastvit_t8", pretrained=False, fork_feat=True,
    )

    model.to(DEVICE)

    output = model(x)

    # Check training output
    assert len(output) == 4
    for i, o in enumerate(output):
        assert list(o.shape[2:]) == list(input_shape // (2 ** (i + 2)))

    model = timm.create_model(
        model_name="fastvit_t8", pretrained=False, fork_feat=True,
    )

    model.to(DEVICE)

    output = model(x)

    # Check training output
    assert len(output) == 4
    for i, o in enumerate(output):
        assert list(o.shape[2:]) == list(input_shape // (2 ** (i + 2)))
