"""Differentiable HPSv3++ reward scorer (Qwen3-VL-8B + FiLM hybrid RankNet).

The caller supplies the iteration condition explicitly; the DVOC training
recipe uses 0.0. The scalar reward is the RankNet mean. Image preprocessing
uses differentiable tensor operations at 448 square resolution and reads
patch geometry from the loaded backbone processor.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from dvoc.rewards import paths

_FIXED_HW = 448


def _ensure_hpsv3pp_on_path() -> str:
    src = os.environ.get("HPSV3PP_SRC", "")
    if not src or not Path(src).is_dir():
        raise FileNotFoundError("Set HPSV3PP_SRC to the HPSv3++ source directory")
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def _write_runtime_config(config_path: str, backbone_path: str) -> str:
    """Rewrite model_name_or_path to the local Qwen3-VL checkpoint."""
    text = Path(config_path).read_text(encoding="utf-8")
    if config_path.endswith(".json"):
        import json

        data = json.loads(text)
        data["model_name_or_path"] = backbone_path
        suffix = ".json"
        payload = json.dumps(data, indent=2, ensure_ascii=False)
    else:
        import re

        if re.search(r"^model_name_or_path\s*:", text, flags=re.M):
            payload = re.sub(
                r"^model_name_or_path\s*:.*$",
                f'model_name_or_path: "{backbone_path}"',
                text,
                count=1,
                flags=re.M,
            )
        else:
            payload = text.rstrip() + f'\nmodel_name_or_path: "{backbone_path}"\n'
        suffix = ".yaml"
    tmp = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8")
    tmp.write(payload)
    tmp.close()
    return tmp.name


class HPSv3PPScorer(torch.nn.Module):
    def __init__(
        self,
        checkpoint_path: str | None = None,
        config_path: str | None = None,
        backbone_path: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        iter_norm: float | None = None,
    ):
        super().__init__()
        _ensure_hpsv3pp_on_path()

        checkpoint_path = checkpoint_path or paths.resolve_hpsv3pp_checkpoint()
        config_path = config_path or paths.resolve_hpsv3pp_config()
        backbone_path = backbone_path or paths.resolve_qwen3vl_model()
        if not checkpoint_path or not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                "HPSv3++ checkpoint not found. Set HPSV3PP_CHECKPOINT to the reward checkpoint file."
            )
        if not config_path or not Path(config_path).exists():
            raise FileNotFoundError(
                "HPSv3++ config not found. Set HPSV3PP_CONFIG to "
                "HPSv3-PlusPlus-src/hpsv3/config/train_stage2.yaml."
            )

        self.device = device
        self.dtype = dtype
        if iter_norm is None:
            iter_norm = float(os.environ.get("HPSV3PP_ITER", "0.3"))
        self.iter_norm = float(iter_norm)

        from hpsv3 import HPSv3RewardInferencer

        runtime_config = _write_runtime_config(config_path, backbone_path)
        try:
            self.inf = HPSv3RewardInferencer(
                config_path=runtime_config,
                checkpoint_path=checkpoint_path,
                device=device,
                differentiable=True,
            )
        finally:
            Path(runtime_config).unlink(missing_ok=True)
        self.inf.model.requires_grad_(False)
        try:
            self.inf.model.config.use_cache = False
            self.inf.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.inf.model.train()
        except Exception:
            self.inf.model.eval()

        self._diff_processor = self._build_diff_processor()
        self._dummy = Image.new("RGB", (_FIXED_HW, _FIXED_HW))
        self.forward_chunk_size = int(os.environ.get("HPSV3PP_FWD_CHUNK", "2"))
        if self.forward_chunk_size < 1:
            raise ValueError("HPSV3PP_FWD_CHUNK must be positive")

    def set_iter_norm(self, value: float) -> None:
        self.iter_norm = float(max(0.0, min(1.0, value)))

    def _build_diff_processor(self):
        ip = getattr(getattr(self.inf, "processor", None), "image_processor", None)
        if ip is not None and hasattr(ip, "_preprocess_differentiable"):
            return ip
        # `create_film_hybrid_model_and_processor` drops the `differentiable` flag that the
        # base factory uses to swap in the torch-op processor, so stage 2 hands back the
        # stock fast processor. Rebuild the fork's processor with the BACKBONE's geometry:
        # Qwen3-VL is 16-px patches with 0.5 mean/std, while the class defaults are
        # Qwen2-VL's 14 px + CLIP stats, which would produce a grid the model cannot fold.
        from hpsv3.model.differentiable_image_processor import Qwen2VLImageProcessor

        def _from_stock(name, default):
            value = getattr(ip, name, None)
            return default if value is None else value

        return Qwen2VLImageProcessor(
            min_pixels=_FIXED_HW * _FIXED_HW,
            max_pixels=_FIXED_HW * _FIXED_HW,
            patch_size=_from_stock("patch_size", 16),
            merge_size=_from_stock("merge_size", 2),
            temporal_patch_size=_from_stock("temporal_patch_size", 2),
            image_mean=_from_stock("image_mean", [0.5, 0.5, 0.5]),
            image_std=_from_stock("image_std", [0.5, 0.5, 0.5]),
        )

    def _iter_values(self, batch_size: int, device, iter_norm=None) -> torch.Tensor:
        value = self.iter_norm if iter_norm is None else float(iter_norm)
        return torch.full((batch_size,), value, device=device, dtype=torch.float32)

    def _diff_pixel_values(self, images01: torch.Tensor):
        if images01.shape[-1] != _FIXED_HW or images01.shape[-2] != _FIXED_HW:
            images01 = torch.nn.functional.interpolate(
                images01,
                size=(_FIXED_HW, _FIXED_HW),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp(0, 1)
        fps, grids = [], []
        for i in range(images01.shape[0]):
            item = images01[i : i + 1]
            try:
                fp, grid = self._diff_processor._preprocess(item, do_rescale=False)
            except TypeError:
                fp, grid = self._diff_processor._preprocess_differentiable(
                    item,
                    do_resize=True,
                    do_rescale=False,
                    do_normalize=True,
                    image_mean=getattr(
                        self._diff_processor, "image_mean", (0.48145466, 0.4578275, 0.40821073)
                    ),
                    image_std=getattr(
                        self._diff_processor, "image_std", (0.26862954, 0.26130258, 0.27577711)
                    ),
                )
            if isinstance(grid, tuple):
                gt, gh, gw = grid
            else:
                gt, gh, gw = grid
            fps.append(fp)
            grids.append([int(gt), int(gh), int(gw)])
        pixel_values = torch.cat(fps, dim=0).to(self.dtype)
        image_grid_thw = torch.tensor(grids, device=images01.device)
        return pixel_values, image_grid_thw

    def _scores(
        self, images01: torch.Tensor, prompts: Sequence[str], *, iter_norm=None
    ) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        if images01.dim() == 3:
            images01 = images01.unsqueeze(0)
        images01 = images01.to(self.device)
        batch_size = images01.shape[0]
        batch = self.inf.prepare_batch([self._dummy] * batch_size, prompts)
        pixel_values, grid = self._diff_pixel_values(images01)
        batch["pixel_values"] = pixel_values.to(self.device)
        if "image_grid_thw" in batch:
            batch["image_grid_thw"] = grid.to(self.device)
        rewards = self.inf.model(
            return_dict=True,
            iter_values=self._iter_values(batch_size, images01.device, iter_norm),
            **batch,
        )["logits"]
        return rewards[:, 0].float()

    @torch.no_grad()
    def __call__(
        self, images: torch.Tensor, prompts: Sequence[str], *, iter_norm=None
    ) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = list(prompts)
        if images.dim() == 3:
            images = images.unsqueeze(0)
        outs = []
        for i in range(0, images.shape[0], self.forward_chunk_size):
            outs.append(
                self._scores(
                    images[i : i + self.forward_chunk_size],
                    prompts[i : i + self.forward_chunk_size],
                    iter_norm=iter_norm,
                ).float()
            )
        return torch.cat(outs, dim=0)
