import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from resnet8_hls.model import (
    build_model_from_numpy,
    build_resnet_from_state_dict,
    describe_network,
    fuse_bn,
    fuse_bn_proj,
    get_fused_weights,
    hls_layer_names,
    spatial_map,
)

from .conftest import build_random_resnet


def test_fuse_bn_matches_conv_followed_by_batchnorm():
    torch.manual_seed(1)
    conv = nn.Conv2d(3, 5, 3, padding=1, bias=False)
    bn = nn.BatchNorm2d(5)
    bn.weight.data.uniform_(0.5, 1.5)
    bn.bias.data.normal_()
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.5, 2.0)
    conv.eval()
    bn.eval()
    x = torch.randn(2, 3, 8, 8)

    weight, bias = fuse_bn(
        conv.weight.detach().numpy(),
        bn.weight.detach().numpy(),
        bn.bias.detach().numpy(),
        bn.running_mean.numpy(),
        bn.running_var.numpy(),
    )
    fused = F.conv2d(x, torch.from_numpy(weight), torch.from_numpy(bias), padding=1)
    assert torch.allclose(bn(conv(x)), fused, atol=1e-5)
    assert weight.dtype == bias.dtype == np.float32


def test_fuse_bn_proj_returns_a_two_dimensional_weight():
    weight, bias = fuse_bn_proj(
        np.ones((6, 3, 1, 1), dtype=np.float32),
        np.ones(6),
        np.zeros(6),
        np.zeros(6),
        np.ones(6),
    )
    assert weight.shape == (6, 3)
    assert bias.shape == (6,)


def test_build_resnet_infers_the_architecture_from_a_state_dict():
    original = build_random_resnet(blocks=(2, 1, 3), stem=6, planes=(6, 12, 24), classes=7)
    rebuilt = build_resnet_from_state_dict(original.state_dict())
    assert {k: v.shape for k, v in rebuilt.state_dict().items()} == {
        k: v.shape for k, v in original.state_dict().items()
    }


def test_build_model_from_numpy_reproduces_the_original_outputs(tiny_model, tiny_state_dict):
    x = torch.randn(2, 3, 32, 32)
    rebuilt = build_model_from_numpy(tiny_state_dict)
    assert not rebuilt.training
    assert torch.allclose(rebuilt(x), tiny_model(x), atol=1e-6)


def _manual_forward(fused, spec, x):
    """Run the network using only the fused arrays, mirroring the generated HLS layers."""
    conv1_w, conv1_b = fused["conv1"]
    out = F.relu(F.conv2d(x, torch.from_numpy(conv1_w), torch.from_numpy(conv1_b), padding=1))
    for block in spec.blocks:
        arrays = [torch.from_numpy(a) for a in fused[block.name]]
        w1, b1, w2, b2 = arrays[:4]
        mid = F.relu(F.conv2d(out, w1, b1, stride=block.stride, padding=1))
        res = F.conv2d(mid, w2, b2, padding=1)
        if len(arrays) == 6:
            skip = F.conv2d(out, arrays[4][:, :, None, None], arrays[5], stride=block.stride)
        else:
            skip = out
        out = F.relu(res + skip)
    out = out.mean(dim=(2, 3))
    fc_w, fc_b = fused["fc"]
    return F.linear(out, torch.from_numpy(fc_w), torch.from_numpy(fc_b))


def test_fused_weights_layout_reproduces_the_float_model(tiny_model, tiny_state_dict):
    fused = get_fused_weights(tiny_state_dict)
    spec = describe_network(tiny_state_dict)
    x = torch.randn(3, 3, 32, 32)
    assert torch.allclose(_manual_forward(fused, spec, x), tiny_model(x), atol=1e-4)


def test_fused_weights_use_projection_only_where_the_shortcut_changes_shape(tiny_state_dict):
    fused = get_fused_weights(tiny_state_dict)
    lengths = {name: len(arrays) for name, arrays in fused.items() if "block" in name}
    assert lengths == {
        "layer1_block0": 4,
        "layer1_block1": 4,
        "layer2_block0": 6,
        "layer3_block0": 6,
        "layer3_block1": 4,
    }


def test_describe_network(tiny_state_dict):
    spec = describe_network(tiny_state_dict)
    assert [b.name for b in spec.blocks] == [
        "layer1_block0",
        "layer1_block1",
        "layer2_block0",
        "layer3_block0",
        "layer3_block1",
    ]
    assert [b.h_out for b in spec.blocks] == [32, 32, 16, 8, 8]
    assert [b.h_in for b in spec.blocks] == [32, 32, 32, 16, 8]
    assert [b.stride for b in spec.blocks] == [1, 1, 2, 2, 1]
    assert [b.c_in for b in spec.blocks] == [4, 4, 4, 8, 16]
    assert spec.stem_channels == 4
    assert spec.final_channels == spec.fc_in == 16
    assert spec.final_spatial == 8
    assert spec.fc_out == 10
    assert spec.blocks[2].prefix == "layer2.0"


def test_spatial_map_and_layer_names(tiny_state_dict):
    sizes = spatial_map(tiny_state_dict)
    assert sizes["conv1"] == (32, 32)
    assert sizes["layer3.1"] == (8, 8)
    names = hls_layer_names(tiny_state_dict)
    assert names[0] == "conv1"
    assert names[-2:] == ["gap", "fc"]
    assert len(names) == 1 + 5 + 2
