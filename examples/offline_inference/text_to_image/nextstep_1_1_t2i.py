# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
NextStep-1.1 Text-to-Image Generation Example

This script demonstrates how to use NextStep-1.1 with vLLM-Omni for
text-to-image generation.

Example usage:
    python nextstep_1_1_t2i.py --prompt "A baby panda wearing an Iron Man mask"
    python nextstep_1_1_t2i.py --prompt "A photo of a cat" --height 512 --width 512
"""

import argparse
import time
from pathlib import Path

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, logger
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.utils.platform_utils import detect_device_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an image with NextStep-1.1."
    )
    parser.add_argument(
        "--model",
        default="stepfun-ai/NextStep-1.1",
        help="NextStep model name or local path.",
    )
    parser.add_argument(
        "--prompt",
        default="A baby panda wearing an Iron Man mask, ultra realistic, detailed",
        help="Text prompt for image generation.",
    )
    parser.add_argument(
        "--negative_prompt",
        default="lowres, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, worst quality, low quality, "
        "normal quality, jpeg artifacts, signature, watermark, username, blurry.",
        help="Negative prompt for classifier-free guidance.",
    )
    parser.add_argument(
        "--seed", type=int, default=3407, help="Random seed for deterministic results."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (cfg).",
    )
    parser.add_argument(
        "--cfg_img",
        type=float,
        default=1.0,
        help="Image-level classifier-free guidance scale.",
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Height of generated image."
    )
    parser.add_argument(
        "--width", type=int, default=512, help="Width of generated image."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="nextstep_output.png",
        help="Path to save the generated image (PNG).",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate for the given prompt.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="Number of sampling steps (default 28 for NextStep-1.1).",
    )
    parser.add_argument(
        "--timesteps_shift",
        type=float,
        default=1.0,
        help="Timesteps shift parameter for sampling.",
    )
    parser.add_argument(
        "--cfg_schedule",
        type=str,
        default="constant",
        choices=["constant", "linear"],
        help="CFG schedule type.",
    )
    parser.add_argument(
        "--use_norm",
        action="store_true",
        help="Apply layer normalization to sampled tokens.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = detect_device_type()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Configure parallel settings
    parallel_config = DiffusionParallelConfig(ulysses_degree=1, ring_degree=1)

    # Initialize Omni with NextStep-1.1
    omni = Omni(
        model=args.model,
        model_class_name="NextStep11Pipeline",  # Specify the pipeline class
        parallel_config=parallel_config,
    )

    # Print configuration
    print(f"\n{'=' * 60}")
    print("NextStep-1.1 Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Inference steps: {args.num_inference_steps}")
    print(f"  CFG scale: {args.guidance_scale}")
    print(f"  CFG image scale: {args.cfg_img}")
    print(f"  Image size: {args.width}x{args.height}")
    print(f"  Seed: {args.seed}")
    print(f"{'=' * 60}\n")

    # Time profiling for generation
    generation_start = time.perf_counter()

    # Generate images
    # NextStep-specific parameters are passed through 'extra' dict
    outputs = omni.generate(
        args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        generator=generator,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        num_outputs_per_prompt=args.num_images_per_prompt,
        seed=args.seed,
        extra={
            "cfg_img": args.cfg_img,
            "timesteps_shift": args.timesteps_shift,
            "cfg_schedule": args.cfg_schedule,
            "use_norm": args.use_norm,
        },
    )

    generation_end = time.perf_counter()
    generation_time = generation_end - generation_start

    # Print profiling results
    print(
        f"Total generation time: {generation_time:.4f} seconds "
        f"({generation_time * 1000:.2f} ms)"
    )

    # Extract images from OmniRequestOutput
    outputs = list(outputs)
    if not outputs:
        raise ValueError("No output generated from omni.generate()")

    logger.info(f"Outputs: {outputs}")

    # Extract images from request_output
    first_output = outputs[0]
    if not hasattr(first_output, "request_output") or not first_output.request_output:
        raise ValueError("No request_output found in OmniRequestOutput")

    req_out = first_output.request_output[0]
    if not isinstance(req_out, OmniRequestOutput) or not hasattr(req_out, "images"):
        raise ValueError("Invalid request_output structure or missing 'images' key")

    images = req_out.images
    if not images:
        raise ValueError("No images found in request_output")

    # Save generated images
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix or ".png"
    stem = output_path.stem or "nextstep_output"

    if len(images) <= 1:
        images[0].save(output_path)
        print(f"Saved generated image to {output_path}")
    else:
        for idx, img in enumerate(images):
            save_path = output_path.parent / f"{stem}_{idx}{suffix}"
            img.save(save_path)
            print(f"Saved generated image to {save_path}")


if __name__ == "__main__":
    main()
