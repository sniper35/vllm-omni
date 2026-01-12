# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Local implementation of NextStep model for vLLM-Omni.

This module provides a local NextStep model implementation that doesn't require
trust_remote_code=True, avoiding import conflicts with vllm_omni.utils.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from vllm_omni.diffusion.models.nextstep_1_1.modeling_fm_head import FlowMatchingHead


@dataclass
class NextStepConfig:
    """Configuration for NextStep model with image generation capabilities."""

    # Base Llama config parameters
    hidden_size: int = 5120
    num_hidden_layers: int = 48
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    intermediate_size: int = 13824
    vocab_size: int = 152064
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    hidden_act: str = "silu"
    attention_bias: bool = True
    mlp_bias: bool = False
    tie_word_embeddings: bool = False
    bos_token_id: int = 151643
    eos_token_id: int = 151643

    # Image generation specific parameters
    num_channels: int = 16  # VAE latent channels
    patch_size: int = 2  # Latent patch size
    image_size: int = 64  # Base image grid size
    boi: int = 151667  # Beginning of image token
    eoi: int = 151668  # End of image token
    image_placeholder_id: int = 151669  # Image placeholder token

    # Flow matching head parameters
    genloss_width: int = 1536  # FlowMatchingHead hidden dim
    genloss_depth: int = 12  # FlowMatchingHead layers

    # Optional features
    use_gen_pos_embed: bool = False
    use_2d_rope: bool = False
    base_image_grid_size: int = 64

    # VAE path
    vae_name_or_path: str | None = None

    @classmethod
    def from_json_file(cls, json_path: str) -> NextStepConfig:
        """Load config from a JSON file."""
        with open(json_path, encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> NextStepConfig:
        """Create config from a dictionary."""
        # Extract only the fields that exist in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    def to_llama_config(self) -> LlamaConfig:
        """Convert to a LlamaConfig for the base transformer."""
        return LlamaConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            hidden_act=self.hidden_act,
            attention_bias=self.attention_bias,
            mlp_bias=self.mlp_bias,
            tie_word_embeddings=self.tie_word_embeddings,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, pe_interpolation: float = 1.0) -> np.ndarray:
    """Generate 2D sinusoidal positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32) / pe_interpolation
    grid_w = np.arange(grid_size, dtype=np.float32) / pe_interpolation
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])

    # Get embeddings for each dimension
    half_dim = embed_dim // 2
    omega = np.arange(half_dim // 2, dtype=np.float64)
    omega /= half_dim / 2.0
    omega = 1.0 / 10000**omega

    # Height embeddings
    pos_h = grid[1].reshape(-1)
    out_h = np.einsum("m,d->md", pos_h, omega)
    emb_h = np.concatenate([np.sin(out_h), np.cos(out_h)], axis=1)

    # Width embeddings
    pos_w = grid[0].reshape(-1)
    out_w = np.einsum("m,d->md", pos_w, omega)
    emb_w = np.concatenate([np.sin(out_w), np.cos(out_w)], axis=1)

    return np.concatenate([emb_h, emb_w], axis=1)


class NextStepModel(nn.Module):
    """
    NextStep model for autoregressive image generation.

    This is a local implementation that wraps a LlamaModel backbone
    with image generation components (projectors and flow matching head).
    """

    def __init__(self, config: NextStepConfig, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self._dtype = dtype

        # Token dimension for image patches
        self.token_dim = config.num_channels * config.patch_size * config.patch_size

        # Text embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # LM head for text generation (not used for image gen but needed for weight loading)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Image projectors
        self.image_in_projector = nn.Linear(self.token_dim, config.hidden_size)
        self.image_out_projector = nn.Linear(config.hidden_size, config.hidden_size)

        # Flow matching head for image token generation
        self.image_head = FlowMatchingHead(
            input_dim=self.token_dim,
            cond_dim=config.hidden_size,
            dim=config.genloss_width,
            layers=config.genloss_depth,
            mlp_ratio=1.0,
        )

        # Base Llama model (decoder layers, norm, rotary embeddings)
        llama_config = config.to_llama_config()
        self.model = LlamaModel(llama_config)

        # Optional 2D positional embeddings for generation
        if config.use_gen_pos_embed:
            max_grid = config.base_image_grid_size
            pos_embed = get_2d_sincos_pos_embed(config.hidden_size, max_grid)
            self.register_buffer(
                "gen_pos_embed",
                torch.from_numpy(pos_embed).float().unsqueeze(0),
                persistent=False,
            )

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def patchify(self, img: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patches.

        Args:
            img: (bsz, C, H, W) image tensor

        Returns:
            (bsz, num_patches, patch_size^2 * C) patch tensor
        """
        bsz, c, h, w = img.shape
        p = self.config.patch_size
        h_patches = h // p
        w_patches = w // p

        # Reshape to patches
        img = img.reshape(bsz, c, h_patches, p, w_patches, p)
        img = img.permute(0, 2, 4, 3, 5, 1)  # (bsz, h_patches, w_patches, p, p, c)
        img = img.reshape(bsz, h_patches * w_patches, p * p * c)
        return img

    def unpatchify(self, x: torch.Tensor, h: int | None = None, w: int | None = None) -> torch.Tensor:
        """
        Convert patches back to image.

        Args:
            x: (bsz, num_patches, patch_size^2 * C) patch tensor
            h: Optional height in patches (inferred from sqrt if not provided)
            w: Optional width in patches (inferred from sqrt if not provided)

        Returns:
            (bsz, C, H, W) image tensor
        """
        bsz, num_patches, _ = x.shape
        p = self.config.patch_size
        c = self.config.num_channels

        if h is None or w is None:
            # Assume square
            h = w = int(num_patches**0.5)

        # Reshape from patches
        x = x.reshape(bsz, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (bsz, c, h, p, w, p)
        x = x.reshape(bsz, c, h * p, w * p)
        return x

    def gen_pos_embed_with_ar(self, h: int, w: int) -> torch.Tensor:
        """
        Get 2D positional embeddings for autoregressive generation.

        Args:
            h: Image height
            w: Image width

        Returns:
            Positional embeddings of shape (1, h*w, hidden_size)
        """
        if not hasattr(self, "gen_pos_embed"):
            raise ValueError("use_gen_pos_embed is False, positional embeddings not available")

        # Calculate grid positions
        down_factor = self.config.patch_size * 8  # VAE factor * patch size
        h_grid = h // down_factor
        w_grid = w // down_factor

        # Extract relevant positions from precomputed embeddings
        base_grid = self.config.base_image_grid_size
        pos_embed = self.gen_pos_embed.reshape(1, base_grid, base_grid, -1)
        pos_embed = pos_embed[:, :h_grid, :w_grid, :]
        pos_embed = pos_embed.reshape(1, h_grid * w_grid, -1)

        return pos_embed.to(self.dtype)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Prepare input embeddings from token IDs and optional image latents.

        Args:
            input_ids: (bsz, seq_len) token IDs
            latents: Optional (bsz, C, H, W) image latents

        Returns:
            (bsz, seq_len, hidden_size) input embeddings
        """
        # Get text embeddings
        inputs_embeds = self.embed_tokens(input_ids)

        if latents is not None:
            # Convert latents to patches and project
            image_tokens = self.patchify(latents)
            image_embeds = self.image_in_projector(image_tokens.to(self.dtype))

            # Find BOI positions and replace with image embeddings
            boi_id = self.config.boi

            for batch_idx in range(input_ids.shape[0]):
                # Find BOI position
                boi_positions = (input_ids[batch_idx] == boi_id).nonzero(as_tuple=True)[0]
                if len(boi_positions) > 0:
                    # Find placeholder positions after BOI
                    start_pos = boi_positions[0].item() + 1
                    num_image_tokens = image_embeds.shape[1]

                    # Replace placeholder embeddings with image embeddings
                    inputs_embeds[batch_idx, start_pos : start_pos + num_image_tokens] = image_embeds[batch_idx]

        return inputs_embeds

    def forward_model(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool = True,
        position_ids: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPast:
        """
        Forward pass through the transformer model.

        Args:
            inputs_embeds: (bsz, seq_len, hidden_size) input embeddings
            attention_mask: Optional attention mask
            past_key_values: Optional cached key-value states
            use_cache: Whether to return key-value cache
            position_ids: Optional position IDs

        Returns:
            BaseModelOutputWithPast with last_hidden_state and past_key_values
        """
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            return_dict=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool = True,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """Standard forward method."""
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        return self.forward_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: NextStepConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cuda",
    ) -> NextStepModel:
        """
        Load a pretrained NextStep model.

        Args:
            model_path: Path to the model directory
            config: Optional config (loaded from model_path/config.json if not provided)
            dtype: Data type for model parameters
            device: Device to load model on

        Returns:
            Loaded NextStepModel
        """
        # Load config if not provided
        if config is None:
            config_path = os.path.join(model_path, "config.json")
            config = NextStepConfig.from_json_file(config_path)

        # Create model
        model = cls(config, dtype=dtype)

        # Load weights from safetensors
        model._load_safetensor_weights(model_path)

        # Move to device and dtype
        model = model.to(device=device, dtype=dtype)

        return model

    def _load_safetensor_weights(self, model_path: str) -> None:
        """Load weights from safetensors files."""
        from safetensors import safe_open

        # Find weight index file
        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index.get("weight_map", {})
            weight_files = set(weight_map.values())
        else:
            # Single file
            weight_files = {"model.safetensors"}

        # Build mapping from HF weights to local weights
        weight_mapping = self._build_weight_mapping()

        # Load weights from each file
        loaded_keys = set()
        for weight_file in weight_files:
            file_path = os.path.join(model_path, weight_file)
            if not os.path.exists(file_path):
                continue

            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    # Map HF key to local key
                    local_key = weight_mapping.get(key, key)

                    # Try to find the parameter
                    try:
                        param = self._get_parameter(local_key)
                        if param is not None:
                            tensor = f.get_tensor(key)
                            param.data.copy_(tensor)
                            loaded_keys.add(key)
                    except (KeyError, AttributeError):
                        # Weight not found in model, skip
                        pass

    def _build_weight_mapping(self) -> dict[str, str]:
        """Build mapping from HuggingFace weight names to local weight names."""
        mapping = {}

        # Embeddings
        mapping["embed_tokens.weight"] = "embed_tokens.weight"
        mapping["model.embed_tokens.weight"] = "embed_tokens.weight"

        # LM head
        mapping["lm_head.weight"] = "lm_head.weight"

        # Image projectors
        mapping["image_in_projector.weight"] = "image_in_projector.weight"
        mapping["image_in_projector.bias"] = "image_in_projector.bias"
        mapping["image_out_projector.weight"] = "image_out_projector.weight"
        mapping["image_out_projector.bias"] = "image_out_projector.bias"

        # Flow matching head - map to our local FlowMatchingHead
        # The HF model uses image_head.net.* which maps to our image_head.net.*
        # These should match automatically

        # Transformer layers - HF uses model.layers.*, we use model.model.layers.*
        for i in range(self.config.num_hidden_layers):
            hf_prefix = f"model.layers.{i}"
            local_prefix = f"model.layers.{i}"

            # Self attention
            for component in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            ]:
                for suffix in ["weight", "bias"]:
                    mapping[f"{hf_prefix}.{component}.{suffix}"] = f"{local_prefix}.{component}.{suffix}"

            # MLP
            for component in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
                for suffix in ["weight", "bias"]:
                    mapping[f"{hf_prefix}.{component}.{suffix}"] = f"{local_prefix}.{component}.{suffix}"

            # Layer norms
            for component in ["input_layernorm", "post_attention_layernorm"]:
                mapping[f"{hf_prefix}.{component}.weight"] = f"{local_prefix}.{component}.weight"

        # Final norm
        mapping["model.norm.weight"] = "model.norm.weight"

        return mapping

    def _get_parameter(self, key: str) -> nn.Parameter | None:
        """Get a parameter by dot-separated key."""
        parts = key.split(".")
        obj = self
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif hasattr(obj, "_modules") and part in obj._modules:
                obj = obj._modules[part]
            elif hasattr(obj, "_parameters") and part in obj._parameters:
                return obj._parameters[part]
            elif hasattr(obj, "_buffers") and part in obj._buffers:
                return obj._buffers[part]
            else:
                # Try numeric index for ModuleList
                try:
                    idx = int(part)
                    obj = obj[idx]
                except (ValueError, TypeError, IndexError):
                    return None

        if isinstance(obj, nn.Parameter):
            return obj
        elif isinstance(obj, torch.Tensor):
            return obj
        elif hasattr(obj, "weight"):
            return obj.weight
        return None
