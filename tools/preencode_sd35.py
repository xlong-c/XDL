#!/usr/bin/env python3
"""SD3.5 Text Encoder - 输出格式和 promt.pt 一致."""

from __future__ import annotations

import os
import torch
from diffusers import StableDiffusion3Pipeline


os.environ["HF_TOKEN"] = "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"
#!/usr/bin/env python
# coding=utf-8
"""
Script to pre-generate prompt embeddings for Stable Diffusion 3.5 training
to reduce computational overhead during training.
"""

import argparse
from transformers import CLIPTextModelWithProjection, T5EncoderModel, T5TokenizerFast
from transformers import CLIPTokenizer
from pathlib import Path


def load_pipeline(pretrained_model_path, revision=None, variant=None):
    """
    Load the SD3.5 pipeline components needed for text encoding
    """
    print(f"Loading model from: {pretrained_model_path}")

    # Load individual components instead of full pipeline to be more efficient
    # Note: SD3.5 has three encoders: clip_l, clip_g, t5xxl
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path,
        subfolder="tokenizer",
        revision=revision,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        pretrained_model_path,
        subfolder="tokenizer_2",
        revision=revision,
    )
    tokenizer_3 = T5TokenizerFast.from_pretrained(
        pretrained_model_path,
        subfolder="tokenizer_3",
        revision=revision,
    )

    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder",
        revision=revision,
        variant=variant,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder_2",
        revision=revision,
        variant=variant,
    )
    text_encoder_3 = T5EncoderModel.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder_3",
        revision=revision,
        variant=variant,
    )

    return {
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "tokenizer_3": tokenizer_3,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "text_encoder_3": text_encoder_3,
    }


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt_list,
    device,
    weight_dtype,
    text_input_ids=None,
    num_images_per_prompt=1,
):
    """Encode prompts using CLIP text encoder."""
    batch_size = len(prompt_list)
    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt_list,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
    elif text_input_ids is None:
        raise ValueError("Either tokenizer or text_input_ids must be provided")
    else:
        text_input_ids = text_input_ids.to(device)

    # Forward pass through CLIP encoder
    outputs = text_encoder(text_input_ids, output_hidden_states=True)
    pooled = outputs[0]  # Pooled output
    last_hidden = outputs.hidden_states[-2]  # Second-to-last hidden state
    prompt_embeds = last_hidden.to(dtype=weight_dtype, device=device)

    # Repeat embeddings for multiple images per prompt
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled.to(device)


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    prompt_list,
    max_sequence_length,
    num_images_per_prompt,
    device,
    weight_dtype,
    text_input_ids=None,
):
    """Encode prompts using T5 text encoder."""
    batch_size = len(prompt_list)
    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt_list,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
    elif text_input_ids is None:
        raise ValueError("Either tokenizer or text_input_ids must be provided")
    else:
        text_input_ids = text_input_ids.to(device)

    # Forward pass through T5 encoder
    prompt_embeds = text_encoder(text_input_ids)[0]
    prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=device)

    # Repeat embeddings for multiple images per prompt
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt_list,
    max_sequence_length,
    device,
    weight_dtype,
    num_images_per_prompt=1,
    text_input_ids_list=None,
):
    """
    Encode prompts using all three text encoders (CLIP1, CLIP2, T5) for SD3.5.

    Args:
        text_encoders: List of [clip_encoder_1, clip_encoder_2, t5_encoder]
        tokenizers: List of [clip_tokenizer_1, clip_tokenizer_2, t5_tokenizer]
        prompt_list: List of text prompts to encode
        max_sequence_length: Maximum sequence length for T5 encoder
        device: Target device for computations
        num_images_per_prompt: Number of images to generate per prompt
        text_input_ids_list: Pre-tokenized input IDs (optional)

    Returns:
        Tuple of:
            - prompt_embeds: Concatenated text embeddings from all encoders
            - pooled_embeds: Pooled embeddings from CLIP encoders
    """
    # Process CLIP encoders (first two)
    clip_tokenizers, clip_encoders = tokenizers[:2], text_encoders[:2]
    clip_embeds_list, pooled_list = [], []

    for i, (tok, enc) in enumerate(zip(clip_tokenizers, clip_encoders)):
        if tok is not None:
            # Use tokenizer to create token IDs
            from transformers import CLIPTokenizer

            if isinstance(tok, CLIPTokenizer):
                text_inputs = tok(
                    prompt_list,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt",
                )
                token_ids = text_inputs.input_ids.to(device)
            else:
                token_ids = None
        else:
            # Use pre-tokenized IDs
            token_ids = (
                text_input_ids_list[i].to(device)
                if text_input_ids_list and text_input_ids_list[i] is not None
                else None
            )
            if token_ids is None:
                raise ValueError(
                    f"No tokenizer or token IDs provided for CLIP encoder {i + 1}"
                )

        # Encode with CLIP
        embeds, pooled = _encode_prompt_with_clip(
            text_encoder=enc,
            tokenizer=None,  # Already tokenized
            prompt_list=prompt_list,
            device=device,
            weight_dtype=weight_dtype,
            text_input_ids=token_ids,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_embeds_list.append(embeds)
        pooled_list.append(pooled)

    # Concatenate CLIP embeddings
    clip_embeds = torch.cat(clip_embeds_list, dim=-1)
    pooled_embeds = torch.cat(pooled_list, dim=-1)

    # Process T5 encoder (third encoder)
    if tokenizers[2] is not None:
        # Use tokenizer to create token IDs
        t5_tokenizer = tokenizers[2]
        text_inputs = t5_tokenizer(
            prompt_list,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        t5_token_ids = text_inputs.input_ids.to(device)
    else:
        # Use pre-tokenized IDs
        t5_token_ids = (
            text_input_ids_list[2].to(device)
            if text_input_ids_list and text_input_ids_list[2] is not None
            else None
        )
        if t5_token_ids is None:
            raise ValueError("No tokenizer or token IDs provided for T5 encoder")

    # Encode with T5
    t5_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[2],
        tokenizer=None,  # Already tokenized
        prompt_list=prompt_list,
        max_sequence_length=max_sequence_length,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        weight_dtype=weight_dtype,
        text_input_ids=t5_token_ids,
    )

    # Pad CLIP embeddings to match T5 dimensionality and concatenate (following SD3.5's exact approach)
    t5_dim = t5_embeds.shape[-1]
    clip_embeds = torch.nn.functional.pad(
        clip_embeds,
        (0, t5_dim - clip_embeds.shape[-1]),
    )
    prompt_embeds = torch.cat([clip_embeds, t5_embeds], dim=-2)

    return prompt_embeds, pooled_embeds


def tokenize_prompt(tokenizer, prompt_list, max_length, device):
    """Tokenize a list of prompts and move to specified device."""
    text_inputs = tokenizer(
        prompt_list,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_inputs.input_ids.to(device)


def encode_single_prompt(text_encoders, tokenizers, prompt, device, weight_dtype):
    """
    Encode a single prompt using all three text encoders for SD3.5.

    Args:
        text_encoders: Dict containing all three text encoders
        tokenizers: Dict containing all three tokenizers
        prompt: Single text prompt to encode
        device: Device to run encoding on
        weight_dtype: Weight dtype for the embeddings

    Returns:
        Tuple of (prompt_embeds, pooled_embeds)
    """
    with torch.no_grad():
        # Process CLIP encoders (first two)
        clip_tokenizers = [tokenizers["tokenizer"], tokenizers["tokenizer_2"]]
        clip_encoders = [text_encoders["text_encoder"], text_encoders["text_encoder_2"]]
        clip_embeds_list, pooled_list = [], []

        for i, (tok, enc) in enumerate(zip(clip_tokenizers, clip_encoders)):
            # Tokenize and encode with CLIP
            token_ids = tokenize_prompt(tok, [prompt], 77, device)
            embeds, pooled = _encode_prompt_with_clip(
                text_encoder=enc,
                tokenizer=None,  # Already tokenized
                prompt_list=[prompt],
                device=device,
                weight_dtype=weight_dtype,
                text_input_ids=token_ids,
                num_images_per_prompt=1,
            )
            clip_embeds_list.append(embeds)
            pooled_list.append(pooled)

        # Concatenate CLIP embeddings
        clip_embeds = torch.cat(clip_embeds_list, dim=-1)
        pooled_embeds = torch.cat(pooled_list, dim=-1)

        # Process T5 encoder (third encoder)
        t5_tokenizer = tokenizers["tokenizer_3"]
        t5_token_ids = tokenize_prompt(
            t5_tokenizer, [prompt], 77, device
        )  # Use 77 for T5

        # Encode with T5
        t5_embeds = _encode_prompt_with_t5(
            text_encoder=text_encoders["text_encoder_3"],
            tokenizer=None,  # Already tokenized
            prompt_list=[prompt],
            max_sequence_length=77,  # T5 max length
            num_images_per_prompt=1,
            device=device,
            weight_dtype=weight_dtype,
            text_input_ids=t5_token_ids,
        )

        # Pad CLIP embeddings to match T5 dimensionality and concatenate (following SD3.5's exact approach)
        t5_dim = t5_embeds.shape[-1]
        clip_embeds = torch.nn.functional.pad(
            clip_embeds,
            (0, t5_dim - clip_embeds.shape[-1]),
        )
        prompt_embeds = torch.cat([clip_embeds, t5_embeds], dim=-2)

        # Move to CPU to save GPU memory
        prompt_embeds = prompt_embeds.cpu()
        pooled_embeds = pooled_embeds.cpu()

        return prompt_embeds, pooled_embeds


def generate_and_save_embeddings(
    prompts,
    pretrained_model_path,
    output_dir,
    device="cuda",
    revision=None,
    variant=None,
):
    """
    Generate and save embeddings for a list of prompts

    Args:
        prompts: List of prompt strings
        pretrained_model_path: Path to the pretrained model
        output_dir: Directory to save embeddings
        device: Device to use for encoding
        revision: Specific model revision to use
        variant: Model variant (e.g., 'fp16')
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize models with appropriate dtypes
    weight_dtype = torch.float16 if device == "cuda" else torch.float32

    # Load components
    components = load_pipeline(pretrained_model_path, revision, variant)

    # Move models to device and set to eval mode
    for name, model in components.items():
        if hasattr(model, "to"):
            model = model.to(
                device, dtype=weight_dtype if "encoder_3" not in name else torch.float16
            )
            model.eval()

    print(f"Processing {len(prompts)} prompts...")

    for i, prompt in enumerate(prompts):
        print(f"Encoding prompt {i + 1}/{len(prompts)}: {prompt[:50]}...")

        # Generate embeddings
        prompt_embeds, pooled_embeds = encode_single_prompt(
            text_encoders=components,
            tokenizers=components,
            prompt=prompt,
            device=device,
            weight_dtype=weight_dtype,
        )

        # Create filename-friendly version of prompt
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in " _-").rstrip()
        if len(safe_prompt) > 100:
            safe_prompt = safe_prompt[:100]

        # Save embeddings with combined information
        embed_filename = f"combined_embed_{i}_{safe_prompt}.pt"
        print(prompt_embeds.shape, pooled_embeds.shape)
        # Save both embeddings together in a single file
        embed_data = {"prompt_embeds": prompt_embeds, "pooled_embeds": pooled_embeds}
        torch.save(embed_data, os.path.join(output_dir, embed_filename))

        print(f"Saved: {embed_filename}")

    print(f"All embeddings saved to {output_dir}")


def load_embeddings(embed_path, device="cuda"):
    """
    Load pre-generated embeddings

    Args:
        embed_path: Path to combined embeddings file
        device: Device to load embeddings to

    Returns:
        Tuple of (prompt_embeds, pooled_prompt_embeds) on specified device
    """
    embed_data = torch.load(embed_path, map_location=device)
    prompt_embeds = embed_data["prompt_embeds"].to(device)
    pooled_embeds = embed_data["pooled_embeds"].to(device)

    return prompt_embeds, pooled_embeds


def main():
    parser = argparse.ArgumentParser(
        description="Generate pre-encoded prompt embeddings for SD3.5 training"
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Path to pretrained SD3.5 model or HuggingFace Hub identifier",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="precomputed_embeds",
        help="Directory to save generated embeddings",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        type=str,
        help="Prompts to encode (can provide multiple)",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        help="Path to text file containing one prompt per line",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for encoding ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--revision", type=str, default=None, help="Specific model revision to use"
    )
    parser.add_argument(
        "--variant", type=str, default=None, help="Model variant (e.g., 'fp16')"
    )

    args = parser.parse_args()

    # Get prompts from either command line or file
    prompts = []
    if args.prompts:
        prompts.extend(args.prompts)
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts.extend([line.strip() for line in f if line.strip()])

    if not prompts:
        # Default sample prompts
        prompts = [
            "Transfer the hairstyle from Image2 to Image1 while preserving facial features, expressions, and background."
        ]
        print("No prompts provided, using default hairstyle transfer prompts:")
        for i, p in enumerate(prompts):
            print(f"  {i + 1}. {p}")

    print(f"Using {len(prompts)} prompts for embedding generation")

    # Generate and save embeddings
    generate_and_save_embeddings(
        prompts=prompts,
        pretrained_model_path=args.pretrained_model_path,
        output_dir=args.output_dir,
        device=args.device,
        revision=args.revision,
        variant=args.variant,
    )


if __name__ == "__main__":
    main()
