# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Compare two generated images with quantitative metrics.

Computes MSE, PSNR, SSIM, and max pixel difference between two images,
then generates a side-by-side visual comparison with metrics overlay.

Example usage:
    python compare_outputs.py \
        --image1 hf_baby_panda.png --label1 "HuggingFace" \
        --image2 vllm_baby_panda.png --label2 "vllm-omni" \
        --output comparison.png
"""

import argparse
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Squared Error between two images."""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def compute_psnr(mse: float, max_val: float = 255.0) -> float:
    """Compute Peak Signal-to-Noise Ratio. Higher = more similar."""
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (simplified per-channel mean).

    Uses the standard SSIM formula with default constants:
        C1 = (K1 * L)^2, C2 = (K2 * L)^2 where K1=0.01, K2=0.03, L=255
    """
    L = 255.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = img1.mean(axis=(0, 1))
    mu2 = img2.mean(axis=(0, 1))
    sigma1_sq = img1.var(axis=(0, 1))
    sigma2_sq = img2.var(axis=(0, 1))
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean(axis=(0, 1))

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_per_channel = numerator / denominator
    return float(ssim_per_channel.mean())


def compute_max_pixel_diff(img1: np.ndarray, img2: np.ndarray) -> int:
    """Compute maximum absolute pixel difference."""
    return int(np.max(np.abs(img1.astype(np.int16) - img2.astype(np.int16))))


def create_comparison_image(
    img1: Image.Image,
    img2: Image.Image,
    label1: str,
    label2: str,
    metrics: dict,
    verdict: str,
) -> Image.Image:
    """Create a side-by-side comparison image with metrics overlay."""
    # Ensure same size
    w1, h1 = img1.size
    w2, h2 = img2.size
    h = max(h1, h2)
    gap = 20
    margin = 20
    header_height = 40
    metrics_height = 160

    total_w = margin + w1 + gap + w2 + margin
    total_h = margin + header_height + h + gap + metrics_height + margin

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Try to load a monospace font, fall back to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    except (OSError, IOError):
        font_large = ImageFont.load_default()
        font_small = font_large

    # Draw labels
    x1 = margin
    x2 = margin + w1 + gap
    y_header = margin

    draw.text((x1, y_header), label1, fill=(0, 0, 0), font=font_large)
    draw.text((x2, y_header), label2, fill=(0, 0, 0), font=font_large)

    # Paste images
    y_img = margin + header_height
    canvas.paste(img1, (x1, y_img))
    canvas.paste(img2, (x2, y_img))

    # Draw metrics below images
    y_metrics = y_img + h + gap

    # Determine color based on verdict
    verdict_color = (0, 128, 0) if "PASS" in verdict else (200, 0, 0)

    lines = [
        f"MSE:              {metrics['mse']:.4f}",
        f"PSNR:             {metrics['psnr']:.2f} dB",
        f"SSIM:             {metrics['ssim']:.6f}",
        f"Max Pixel Diff:   {metrics['max_pixel_diff']}",
        "",
        f"Verdict: {verdict}",
    ]

    for i, line in enumerate(lines):
        color = verdict_color if line.startswith("Verdict") else (0, 0, 0)
        draw.text((margin, y_metrics + i * 22), line, fill=color, font=font_small)

    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two images with quantitative metrics."
    )
    parser.add_argument(
        "--image1", required=True, help="Path to first image."
    )
    parser.add_argument(
        "--label1", default="Image 1", help="Label for first image."
    )
    parser.add_argument(
        "--image2", required=True, help="Path to second image."
    )
    parser.add_argument(
        "--label2", default="Image 2", help="Label for second image."
    )
    parser.add_argument(
        "--output",
        default="comparison.png",
        help="Path to save the comparison image.",
    )
    parser.add_argument(
        "--psnr_threshold",
        type=float,
        default=30.0,
        help="Minimum PSNR (dB) to pass. Default 30 (>40 = near-identical).",
    )
    parser.add_argument(
        "--ssim_threshold",
        type=float,
        default=0.90,
        help="Minimum SSIM to pass. Default 0.90 (1.0 = identical).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load images
    img1 = Image.open(args.image1).convert("RGB")
    img2 = Image.open(args.image2).convert("RGB")

    if img1.size != img2.size:
        print(
            f"WARNING: Image sizes differ ({img1.size} vs {img2.size}). "
            f"Resizing image2 to match image1."
        )
        img2 = img2.resize(img1.size, Image.LANCZOS)

    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # Compute metrics
    mse = compute_mse(arr1, arr2)
    psnr = compute_psnr(mse)
    ssim = compute_ssim(arr1, arr2)
    max_diff = compute_max_pixel_diff(arr1, arr2)

    metrics = {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "max_pixel_diff": max_diff,
    }

    # Determine verdict
    psnr_pass = psnr >= args.psnr_threshold or psnr == float("inf")
    ssim_pass = ssim >= args.ssim_threshold
    if psnr_pass and ssim_pass:
        verdict = "PASS - Images are sufficiently similar"
    else:
        reasons = []
        if not psnr_pass:
            reasons.append(f"PSNR {psnr:.2f} < {args.psnr_threshold}")
        if not ssim_pass:
            reasons.append(f"SSIM {ssim:.4f} < {args.ssim_threshold}")
        verdict = f"FAIL - {'; '.join(reasons)}"

    # Print results
    print(f"\n{'=' * 60}")
    print("Image Comparison Results")
    print(f"{'=' * 60}")
    print(f"  Image 1:          {args.image1} ({args.label1})")
    print(f"  Image 2:          {args.image2} ({args.label2})")
    print(f"  Size:             {img1.size[0]}x{img1.size[1]}")
    print(f"{'=' * 60}")
    print(f"  MSE:              {mse:.4f}")
    print(f"  PSNR:             {psnr:.2f} dB")
    print(f"  SSIM:             {ssim:.6f}")
    print(f"  Max Pixel Diff:   {max_diff}")
    print(f"{'=' * 60}")
    print(f"  Thresholds:       PSNR >= {args.psnr_threshold} dB, SSIM >= {args.ssim_threshold}")
    print(f"  Verdict:          {verdict}")
    print(f"{'=' * 60}\n")

    # Create and save comparison image
    comparison = create_comparison_image(
        img1, img2, args.label1, args.label2, metrics, verdict
    )
    comparison.save(args.output)
    print(f"Saved comparison image to {args.output}")

    # Exit with non-zero code on failure
    if "FAIL" in verdict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
