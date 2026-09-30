import runpy
import sys
from pathlib import Path

import numpy as np

from resnet8_hls.checkpoint import load_state_dict_numpy
from resnet8_hls.model import describe_network

ROOT = Path(__file__).resolve().parents[1]


def test_demo_checkpoint_script_writes_a_loadable_resnet8(tmp_path, monkeypatch):
    output = tmp_path / "nested" / "demo.pth"
    monkeypatch.setattr(sys, "argv", ["make_demo_checkpoint.py", str(output)])
    runpy.run_path(str(ROOT / "examples" / "make_demo_checkpoint.py"), run_name="__main__")

    state_dict = load_state_dict_numpy(output)
    assert all(isinstance(v, np.ndarray) for v in state_dict.values())
    spec = describe_network(state_dict)
    assert len(spec.blocks) == 3  # ResNet8: one residual block per stage
    assert (spec.stem_channels, spec.fc_in, spec.fc_out) == (16, 64, 10)
