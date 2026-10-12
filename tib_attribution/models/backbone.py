"""Use public model forwards so native masks, pooling and projections stay intact."""

from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch
from torch import nn


def feature_tensor(output):
    # Transformers 4 returns a tensor; Transformers 5 returns a model output.
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    raise TypeError(f"Unsupported feature output: {type(output).__name__}")


def layer_tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


class VisionLanguageBackbone:
    def __init__(self, model: nn.Module, processor=None):
        self.model = model
        self.processor = processor
        model_type = getattr(model.config, "model_type", "")
        if model_type not in {"clip", "siglip"}:
            raise ValueError(f"Supported backbones: CLIP and fixed-grid SigLIP/SigLIP2; got {model_type!r}")
        self.model_type = model_type
        self.has_cls_token = hasattr(model.vision_model.embeddings, "class_embedding")

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def encoder(self, modality):
        if modality not in {"vision", "text"}:
            raise ValueError("modality must be 'vision' or 'text'")
        return getattr(self.model, f"{modality}_model")

    def layer(self, modality, index):
        layers = self.encoder(modality).encoder.layers
        if not 0 <= index < len(layers):
            raise ValueError(f"target_layer={index} outside {modality} encoder's {len(layers)} layers")
        return layers[index]

    def features(self, modality, inputs):
        if modality == "vision":
            output = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        else:
            output = self.model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            )
        return feature_tensor(output)

    def valid_text_tokens(self, inputs):
        """A readout mask independent of the model's native attention/pooling mask."""
        if "attention_mask" in inputs:
            return inputs["attention_mask"].bool()
        tokenizer = getattr(self.processor, "tokenizer", None)
        source = tokenizer if tokenizer is not None else self.model.config.text_config
        pad_id = getattr(source, "pad_token_id", None)
        eos_id = getattr(source, "eos_token_id", None)
        ids = inputs["input_ids"]
        if pad_id is None:
            return torch.ones_like(ids, dtype=torch.bool)
        valid = ids != pad_id
        if pad_id == eos_id:
            # CLIP may use the same ID for EOS and padding: keep only the first EOS.
            is_eos = ids == eos_id
            valid |= is_eos & (is_eos.long().cumsum(-1) == 1)
        return valid

    @contextmanager
    def frozen(self):
        """Restore every module's training flag and every parameter's grad flag."""
        modules = [(module, module.training) for module in self.model.modules()]
        parameters = [(param, param.requires_grad) for param in self.model.parameters()]
        self.model.eval()
        for param, _ in parameters:
            param.requires_grad_(False)
        try:
            yield
        finally:
            for param, flag in parameters:
                param.requires_grad_(flag)
            for module, flag in modules:
                module.training = flag

    @contextmanager
    def inject(self, modality, index, transform: Callable, *, position="input"):
        """Temporarily replace a layer's input, with cleanup even on exceptions."""
        def hook(module, args, kwargs):
            if args:
                return (transform(args[0]), *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": transform(kwargs["hidden_states"])}

        if position == "input":
            handle = self.layer(modality, index).register_forward_pre_hook(hook, with_kwargs=True)
        elif position == "output":
            def output_hook(module, args, output):
                hidden = transform(layer_tensor(output))
                if isinstance(output, tuple):
                    return (hidden, *output[1:])
                if isinstance(output, list):
                    return [hidden, *output[1:]]
                return hidden
            handle = self.layer(modality, index).register_forward_hook(output_hook)
        else:
            raise ValueError("position must be 'input' or 'output'")
        try:
            yield
        finally:
            handle.remove()

    def capture(self, modality, inputs, index, *, position="input"):
        captured = []

        def record(hidden):
            captured.append(hidden.detach())
            return hidden

        with self.inject(modality, index, record, position=position), torch.no_grad():
            features = self.features(modality, inputs)
        return captured[0], features.detach()

    def with_hidden(self, modality, inputs, index, hidden, *, position="input"):
        with self.inject(modality, index, lambda _: hidden, position=position):
            return self.features(modality, inputs)

    def patch_grid(self, inputs):
        patch_size = self.model.config.vision_config.patch_size
        height, width = inputs["pixel_values"].shape[-2:]
        return height // patch_size, width // patch_size


def load_backbone(model_name="openai/clip-vit-base-patch32", *, device=None, local_files_only=False):
    """Load float32 eager attention; never download anything at package import time."""
    from transformers import AutoModel, AutoProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if local_files_only and not Path(model_name).is_dir():
        from huggingface_hub import snapshot_download
        # Loading a cached directory also avoids Transformers' background auto-conversion request.
        model_name = snapshot_download(repo_id=model_name, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_name, local_files_only=local_files_only, attn_implementation="eager"
    ).float().to(device)
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
    return VisionLanguageBackbone(model, processor)
