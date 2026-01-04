import os
import sys
from pathlib import Path

import pytest
import torch

from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.utils.platform_utils import is_npu, is_rocm

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"


@pytest.mark.parametrize("model_name", ["stepfun-ai/NextStep-1.1"])
def test_nextstep_1_1_model(model_name: str):
    """Test NextStep-1.1 text-to-image generation."""
    m = Omni(
        model=model_name,
        model_class_name="NextStep11Pipeline",  # Explicitly specify the pipeline class
    )
    # Use small resolution to avoid OOM
    height = 256
    width = 256
    outputs = m.generate(
        "a photo of a cat",
        height=height,
        width=width,
        num_inference_steps=2,  # Use minimal steps for testing
        guidance_scale=1.0,  # Minimal CFG for testing
        generator=torch.Generator("cuda").manual_seed(42),
        num_outputs_per_prompt=1,
        extra={
            "cfg_img": 1.0,
            "timesteps_shift": 1.0,
        },
    )
    # Extract images from request_output
    first_output = list(outputs)[0]
    assert first_output.final_output_type == "image"
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output[0]
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise ValueError("Invalid request_output structure or missing 'images' key")

    images = req_out.images

    assert len(images) == 1
    # Check image size
    assert images[0].width == width
    assert images[0].height == height
    images[0].save("nextstep_test_output.png")
    print(f"Test passed! Image saved to nextstep_test_output.png")
