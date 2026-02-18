# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
HuggingFace Reference Inference for NextStep-1.1

Uses the official NextStepPipeline from the model repo (models.gen_pipeline)
to generate images. This serves as a ground-truth reference for comparing
against vllm-omni inference outputs.

Example usage:
    python hf_nextstep_inference.py \
        --prompt "A baby panda wearing an Iron Man mask" \
        --seed 42 --output hf_baby_panda.png
"""

import argparse
import os
import sys
import time

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image using the official HF NextStepPipeline.")
    parser.add_argument(
        "--model",
        default="stepfun-ai/NextStep-1.1",
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--prompt",
        default="A baby panda wearing an Iron Man mask, holding a board with 'NextStep-1' written on it",
        help="Text prompt for image generation.",
    )
    parser.add_argument(
        "--positive_prompt",
        default="",
        help="Additional positive prompt appended to the main prompt.",
    )
    parser.add_argument(
        "--negative_prompt",
        default="lowres, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, worst quality, low quality, "
        "normal quality, jpeg artifacts, signature, watermark, username, "
        "blurry.",
        help="Negative prompt for classifier-free guidance.",
    )
    parser.add_argument("--height", type=int, default=512, help="Image height.")
    parser.add_argument("--width", type=int, default=512, help="Image width.")
    parser.add_argument("--cfg", type=float, default=7.5, help="Text CFG guidance scale.")
    parser.add_argument("--cfg_img", type=float, default=1.0, help="Image CFG guidance scale.")
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
    parser.add_argument(
        "--num_sampling_steps",
        type=int,
        default=28,
        help="Number of denoising steps.",
    )
    parser.add_argument(
        "--timesteps_shift",
        type=float,
        default=1.0,
        help="Timesteps shift parameter.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=str,
        default="hf_nextstep_output.png",
        help="Path to save the generated image.",
    )
    parser.add_argument(
        "--save_latents",
        action="store_true",
        help="Also save raw latent tensor as a .pt file for deeper comparison.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Ensure model is downloaded and get local path
    if os.path.isdir(args.model):
        model_path = args.model
    else:
        print(f"Downloading/locating model: {args.model}")
        model_path = snapshot_download(args.model)

    # Add model directory to sys.path so we can import models.gen_pipeline
    if model_path not in sys.path:
        sys.path.insert(0, model_path)

    from models.gen_pipeline import NextStepPipeline
    from vae.nextstep_ae import AutoencoderKL as NextStepVAE

    print(f"Loading tokenizer and model from: {model_path}")
    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    # Resolve VAE path: config has relative "vae/" which needs the model dir
    vae_path = os.path.join(model_path, "vae")
    vae = NextStepVAE.from_pretrained(vae_path)

    pipeline = NextStepPipeline(tokenizer=tokenizer, model=model, vae=vae).to(device="cuda", dtype=dtype)

    load_end = time.perf_counter()
    print(f"Model loaded in {load_end - load_start:.2f}s")

    # Print generation configuration
    print(f"\n{'=' * 60}")
    print("HF Reference Generation Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Prompt: {args.prompt!r}")
    print(f"  Image size: {args.width}x{args.height}")
    print(f"  CFG scale: {args.cfg}")
    print(f"  CFG image scale: {args.cfg_img}")
    print(f"  CFG schedule: {args.cfg_schedule}")
    print(f"  Sampling steps: {args.num_sampling_steps}")
    print(f"  Timesteps shift: {args.timesteps_shift}")
    print(f"  Use norm: {args.use_norm}")
    print(f"  Seed: {args.seed}")
    print(f"  Dtype: {args.dtype}")
    print(f"{'=' * 60}\n")

    # Generate image
    print("Generating image...")
    gen_start = time.perf_counter()

    images = pipeline.generate_image(
        args.prompt,
        hw=(args.height, args.width),
        num_images_per_caption=1,
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
        cfg=args.cfg,
        cfg_img=args.cfg_img,
        cfg_schedule=args.cfg_schedule,
        use_norm=args.use_norm,
        num_sampling_steps=args.num_sampling_steps,
        timesteps_shift=args.timesteps_shift,
        seed=args.seed,
    )

    gen_end = time.perf_counter()
    gen_time = gen_end - gen_start

    print(f"Generation time: {gen_time:.4f}s ({gen_time * 1000:.2f}ms)")

    # Save output image
    image = images[0]
    image.save(args.output)
    print(f"Saved image to {args.output}")

    if args.save_latents:
        print("Note: --save_latents requires pipeline modification; skipping.")
        print("  (The HF pipeline does not expose raw latents by default.)")


if __name__ == "__main__":
    main()
