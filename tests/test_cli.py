import json

import numpy as np
import pytest
from PIL import Image

from qbs.experiment import fit, parser


def options(tmp_path, **overrides):
    image = tmp_path / "input.png"
    Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(image)
    args = parser().parse_args(
        [
            "--image",
            str(image),
            "--out",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--width",
            "16",
            "--height",
            "16",
            "--strokes",
            "8",
            "--steps",
            "2",
            "--audit",
            "none",
        ]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_cpu_fit_exports_and_rejects_overwrite(tmp_path):
    args = options(tmp_path)
    result = fit(args)
    assert np.isfinite(result["best_psnr"])
    for name in ["config.json", "metrics.json", "trace.jsonl", "strokes.npz", "best.png"]:
        assert (tmp_path / "run" / name).exists()
    config = json.loads((tmp_path / "run" / "config.json").read_text())
    assert config["resolved_device"] == "cpu"
    with pytest.raises(FileExistsError):
        fit(args)


@pytest.mark.parametrize("invalid", [{"width": 6}, {"steps": 0}, {"tile": 0}, {"lr": 0}])
def test_fit_rejects_invalid_configuration(tmp_path, invalid):
    with pytest.raises(ValueError):
        fit(options(tmp_path, **invalid))


def test_large_learned_patch_is_rejected_before_checkpoint_load(tmp_path):
    with pytest.raises(ValueError, match="one patch"):
        fit(options(tmp_path, checkpoint="unavailable.pt", width=600))
