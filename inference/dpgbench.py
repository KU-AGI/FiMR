import math
import os
import argparse
import json
import datetime
import yaml
import sys
import shutil
from glob import glob
from typing import List, Tuple, Dict

sys.path.append("./src")

from tqdm import tqdm
import numpy as np
import PIL
import PIL.Image
import torch

from peft import LoraConfig, get_peft_model
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelSummary
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchvision import transforms

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, cwd=True)

from hydra import compose, initialize
from omegaconf import OmegaConf

from src.utils.model import get_model, load_deepspeed_ckpt
from src.dataset.eval_datamodule import EvalDataModule
from src.utils.formatting import FormattingEvaluatorV2

METADATA_FILENAME = "metadata.jsonl"
RATIONAL_FILENAME = "rationales.json"

EDIT_TEMPLATE = """You are a strict image editing assistant.
Your task is to revise a *failed* generated image according to the user's instruction and the original generation intent.

INPUT FORMAT:
1. The source image is located between {image_start_tag} and {image_end_tag}.
2. The original text-to-image generation prompt will be provided after the keyword 'INPUT_PROMPT:'
3. Step-by-step feedback will be provided after the keyword 'FEEDBACK:'.
   - The feedback will be a sequence of instructions, each starting with 'Step X:' (e.g., 'Step 1:', 'Step 2:', ...).
   - You MUST follow ALL steps in order and produce a final image that satisfies the entire sequence, not just an intermediate step.

CRITICAL RULES:
1. You MUST Look at the image between {image_start_tag} and {image_end_tag} as the ground truth.
2. Preserve the background, objects, and style from the input image unless explicitly asked to change them.
3. Do NOT generate a completely new image from scratch.
4. **You MUST strictly maintain the spatial layout and composition of the source image.**
5. You MUST also reference the INPUT_PROMPT as the original intended content of the image, but the visible source image remains the primary ground truth.
6. When applying FEEDBACK, carefully execute each step one by one while keeping previous changes consistent, and ensure the final result reflects all steps combined."""

class JanusTestWrapper(LightningModule):
    def __init__(self, config, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.config = config

        task_cfg = config.eval
        self.num_samples_per_prompt = task_cfg.get("num_samples_per_prompt", 4)
        self.use_self_correction = task_cfg.get("use_self_correction", True)
        self.temperature = task_cfg.get("temperature", 1.0)
        self.cfg_weight = task_cfg.get("cfg_weight", 5.0)

        self.txt_top_k = task_cfg.get("txt_top_k", None)
        self.txt_top_p = task_cfg.get("txt_top_p", None)
        self.img_top_k = task_cfg.get("img_top_k", None)
        self.img_top_p = task_cfg.get("img_top_p", None)

        self.image_token_num_per_image = 576
        self.img_size = 384
        self.patch_size = 16
        self.max_reflect_len = 2048

        self.image_start_tag = self.processor.image_start_tag
        self.image_end_tag = self.processor.image_end_tag
        self.image_tag = self.processor.image_tag
        self.pad = self.processor.pad_tag
        self.eos = self.processor.tokenizer.eos_token

        self.formatting_evaluator = FormattingEvaluatorV2()
        self.max_correction_steps = task_cfg.get("max_correction_steps", 1)

        self.gen_root = os.path.abspath(self.config.save_path)  # .../gen
        self.exp_root = os.path.dirname(self.gen_root)          # experiment root
        self.final_root = os.path.join(self.exp_root, "final_outputs")

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #
    def _get_stage_root(self, stage: str) -> str:
        if stage == "gen":
            return self.gen_root
        return os.path.join(self.exp_root, stage)

    def _get_stage_outpath(self, data_idx: int, stage: str, create: bool = False) -> str:
        root = self._get_stage_root(stage)
        if create:
            os.makedirs(root, exist_ok=True)
        outpath = os.path.join(root, f"{data_idx:05d}")
        if create:
            os.makedirs(outpath, exist_ok=True)
        return outpath

    def _stage_has_outputs(self, data_idx: int, stage: str) -> bool:

        stage_path = self._get_stage_outpath(data_idx, stage, create=False)
        if not os.path.isdir(stage_path):
            return False

        samples_dir = os.path.join(stage_path, "samples")
        rationale_file = os.path.join(stage_path, RATIONAL_FILENAME)

        if not os.path.isdir(samples_dir):
            return False
        existing_images = glob(os.path.join(samples_dir, "*.png"))
        if len(existing_images) < self.num_samples_per_prompt:
            return False

        if stage == "gen":
            return True

        if not os.path.isfile(rationale_file):
            return False
        try:
            with open(rationale_file, "r") as f:
                rats = json.load(f)
            if not isinstance(rats, list) or len(rats) < self.num_samples_per_prompt:
                return False
        except Exception:
            return False

        return True

    def _detect_resume_steps(self, batch: List[Tuple[Dict, int]]) -> Dict[int, int]:
        """
        Compute the next correction step to run for each data_idx.
        Returns 0 if correction_0 has not been run yet.
        """
        resume_steps = {}
        for _, data_idx in batch:
            step = 0
            while step < self.max_correction_steps and self._stage_has_outputs(
                data_idx, "gen" if step == 0 else f"correction_{step-1}"
            ):
                # Advance step if the next stage is already complete.
                next_stage = f"correction_{step}"
                if not self._stage_has_outputs(data_idx, next_stage):
                    break
                step += 1
            resume_steps[data_idx] = step
        return resume_steps

    def _mirror_stage_when_no_correction(self, data_idx: int, source_stage: str, current_step: int):
        target_stage = f"correction_{current_step}"
        if self._stage_has_outputs(data_idx, target_stage):
            return

        src_path = self._get_stage_outpath(data_idx, source_stage, create=False)
        if not os.path.isdir(src_path):
            return

        dst_path = self._get_stage_outpath(data_idx, target_stage, create=True)

        # Copy all samples.
        src_samples = os.path.join(src_path, "samples")
        dst_samples = os.path.join(dst_path, "samples")
        os.makedirs(dst_samples, exist_ok=True)
        for fname in os.listdir(src_samples):
            shutil.copyfile(os.path.join(src_samples, fname), os.path.join(dst_samples, fname))

        # Copy metadata and update iteration.
        src_metadata = os.path.join(src_path, METADATA_FILENAME)
        if os.path.isfile(src_metadata):
            with open(src_metadata, "r") as f:
                meta = json.load(f)
            meta["iteration"] = current_step
            with open(os.path.join(dst_path, METADATA_FILENAME), "w") as f:
                json.dump(meta, f)

        # Copy rationale and mark needs_review as False.
        src_rationale = os.path.join(src_path, RATIONAL_FILENAME)
        if os.path.isfile(src_rationale):
            with open(src_rationale, "r") as f:
                rats = json.load(f)
            for rat in rats:
                if isinstance(rat, dict):
                    rat["needs_review"] = False
                    rat["correction_step"] = current_step
            with open(os.path.join(dst_path, RATIONAL_FILENAME), "w") as f:
                json.dump(rats, f, indent=4)

    def _save_batch_grids_to_final_outputs(self, batch: List[Tuple[Dict, int]]):
        """Save 4 images as a 2x2 grid to final_outputs after each batch."""
        final_root = os.path.join(self.exp_root, "final_outputs")

        for sample_meta, data_idx in batch:
            # 1. Determine filename: use metadata file_name if present, else zero-padded data_idx.
            file_name = sample_meta.get("file_name", f"{data_idx:05d}")

            # 2. All stages to check (gen, correction_0, correction_1, ...).
            stages_to_save = ["gen"] + [f"correction_{i}" for i in range(self.max_correction_steps)]

            for stage in stages_to_save:
                # 3. Check that the stage passed strict validation (e.g. 4 images present).
                if not self._stage_has_outputs(data_idx, stage):
                    continue

                # 4. Set path: final_outputs/{stage}/grid/
                grid_root = os.path.join(final_root, stage, "grid")
                os.makedirs(grid_root, exist_ok=True)
                target_grid_path = os.path.join(grid_root, f"{file_name}.png")

                if os.path.exists(target_grid_path):
                    continue

                # 5. Load 4 images and create 2x2 grid.
                stage_dir = self._get_stage_outpath(data_idx, stage, create=False)
                samples_dir = os.path.join(stage_dir, "samples")
                image_paths = sorted(glob(os.path.join(samples_dir, "*.png")))

                if len(image_paths) == 4:
                    try:
                        imgs = [PIL.Image.open(p) for p in image_paths]
                        w, h = imgs[0].size

                        grid = PIL.Image.new("RGB", (w * 2, h * 2))
                        grid.paste(imgs[0], (0, 0))    # top-left
                        grid.paste(imgs[1], (w, 0))    # top-right
                        grid.paste(imgs[2], (0, h))    # bottom-left
                        grid.paste(imgs[3], (w, h))    # bottom-right

                        grid.save(target_grid_path)
                    except Exception as e:
                        print(f"Error creating grid for {file_name} at stage {stage}: {e}")

    # ------------------------------------------------------------------ #
    # Lightning hooks
    # ------------------------------------------------------------------ #
    def on_test_epoch_start(self):
        self.model.eval()

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        prepared_data = self._prepare_batch_data(batch)
        if prepared_data["prompts"]:
            self._run_generation_loop(prepared_data)

        resume_steps = self._detect_resume_steps(batch)
        start_step = min(resume_steps.values(), default=0)

        if self.use_self_correction:
            for current_step in range(start_step, self.max_correction_steps):
                correction_data = self._prepare_correction_batch_data(
                    batch, current_step, resume_steps
                )
                if len(correction_data["prompts"]) > 0:
                    regen_list = self._run_self_correction_loop(correction_data, current_step)
                    
                    if len(regen_list) > 0:
                        self._run_feedback_img_generation_loop(regen_list)

        self._save_batch_grids_to_final_outputs(batch)

    def on_test_epoch_end(self, outputs=None):
        rank = dist.get_rank()
        print("--------------------------------------------------")
        print(f"RANK: {rank} DONE!")
        print("--------------------------------------------------")

    # ------------------------------------------------------------------ #
    # Stage preparation
    # ------------------------------------------------------------------ #
    def _prepare_batch_data(self, batch: List[Tuple[Dict, int]]) -> Dict[str, List]:
        prompts, data_indices, sample_paths = [], [], []

        for sample, data_idx in batch:
            outpath = self._get_stage_outpath(data_idx, "gen", create=True)
            samples_dir = os.path.join(outpath, "samples")

            existing = sorted(glob(os.path.join(samples_dir, "*.png")))
            if len(existing) >= self.num_samples_per_prompt:
                continue

            os.makedirs(samples_dir, exist_ok=True)
            with open(os.path.join(outpath, METADATA_FILENAME), "w") as f:
                json.dump(sample, f)

            prompts.append(sample["prompt"])
            data_indices.append(data_idx)
            sample_paths.append(samples_dir)

        return {
            "prompts": prompts,
            "data_indices": data_indices,
            "sample_paths": sample_paths,
        }

    def _prepare_correction_batch_data(
        self,
        batch: List[Tuple[Dict, int]],
        current_step: int,
        resume_steps: Dict[int, int],
    ) -> Dict[str, List]:
        prompts, data_indices, stage_paths = [], [], []
        image_tensors, group_sizes, selected_indices, prev_rationales = [], [], [], []

        source_stage = "gen" if current_step == 0 else f"correction_{current_step-1}"
        evaluate_all = current_step == 0 and all(
            resume_steps.get(idx, 0) == 0 for _, idx in batch
        )

        for sample, data_idx in batch:
            sample_resume = resume_steps.get(data_idx, 0)
            if current_step < sample_resume:
                continue

            stage_outpath = self._get_stage_outpath(data_idx, source_stage, create=False)
            if not os.path.isdir(stage_outpath):
                continue

            samples_dir = os.path.join(stage_outpath, "samples")
            if not os.path.isdir(samples_dir):
                continue

            rationale_path = os.path.join(stage_outpath, RATIONAL_FILENAME)
            if os.path.isfile(rationale_path):
                with open(rationale_path, "r") as f:
                    prev_rat = json.load(f)
            else:
                prev_rat = [None] * self.num_samples_per_prompt

            if len(prev_rat) < self.num_samples_per_prompt:
                prev_rat.extend([None] * (self.num_samples_per_prompt - len(prev_rat)))

            indices_to_eval = []
            for img_idx in range(self.num_samples_per_prompt):
                rationale = prev_rat[img_idx]
                needs_review = True if evaluate_all else (
                    rationale is None or rationale.get("needs_review", True)
                )
                if needs_review:
                    img_path = os.path.join(samples_dir, f"{img_idx:05d}.png")
                    if os.path.exists(img_path):
                        indices_to_eval.append(img_idx)

            if not indices_to_eval:
                self._mirror_stage_when_no_correction(
                    data_idx=data_idx,
                    source_stage=source_stage,
                    current_step=current_step,
                )
                continue
            pil_images = []
            for img_idx in indices_to_eval:
                img_path = os.path.join(samples_dir, f"{img_idx:05d}.png")
                with PIL.Image.open(img_path) as img:
                    pil_images.append(img.convert("RGB"))

            processed = self.processor.image_processor(pil_images)
            pixel_values = processed["pixel_values"] if isinstance(processed, dict) else processed.pixel_values
            if isinstance(pixel_values, torch.Tensor):
                batch_pixels = pixel_values
            else:
                batch_pixels = torch.stack(pixel_values)

            for tensor in batch_pixels:
                image_tensors.append(tensor)

            prompts.append(sample["prompt"])
            data_indices.append(data_idx)
            stage_paths.append(stage_outpath)
            group_sizes.append(len(indices_to_eval))
            selected_indices.append(indices_to_eval)
            prev_rationales.append(prev_rat)

        return {
            "prompts": prompts,
            "data_indices": data_indices,
            "stage_paths": stage_paths,
            "image_tensors": image_tensors,
            "group_sizes": group_sizes,
            "selected_indices": selected_indices,
            "prev_rationales": prev_rationales,
        }

    # ------------------------------------------------------------------ #
    # Embed builders
    # ------------------------------------------------------------------ #
    def prepare_input_embeds(self, prompt_list: List[str], system_prompt: str = ""):
        batch_size = len(prompt_list)
        input_ids_list = []
        for prompt in prompt_list:
            conversation = [
                {"role": "<|User|>", "content": prompt},
                {"role": "<|Assistant|>", "content": ""},
            ]
            sft = self.processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.processor.sft_format,
                system_prompt=system_prompt,
            )
            full_prompt = sft + self.processor.image_start_tag
            input_ids = self.processor.tokenizer.encode(full_prompt)
            input_ids_list.append(torch.LongTensor(input_ids))

        max_len = max(len(ids) for ids in input_ids_list)
        tokens = torch.full(
            (batch_size * 2, max_len),
            self.processor.pad_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (batch_size * 2, max_len), dtype=torch.long, device=self.device
        )

        for i, ids in enumerate(input_ids_list):
            pad_len = max_len - len(ids)
            tokens[i * 2, pad_len:] = ids
            attention_mask[i * 2, pad_len:] = 1

            tokens[i * 2 + 1, pad_len:] = ids
            tokens[i * 2 + 1, pad_len + 1 : -1] = self.processor.pad_id
            attention_mask[i * 2 + 1, pad_len:] = 1

        input_embeds = self.model.language_model.get_input_embeddings()(tokens)
        return input_embeds, attention_mask

    def prepare_correction_input_embeds(
        self,
        prompt_list: List[str],
        image_tensors: List[torch.Tensor],
        group_sizes: List[int],
    ):
        batch_size = len(prompt_list)
        input_ids_list = []

        for prompt in prompt_list:
            conversation = [
                {"role": "<|User|>", "content": prompt},
                {"role": "<|Assistant|>", "content": ""},
            ]
            sft = self.processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.processor.sft_format,
                system_prompt="",
            )
            full_prompt = (
                sft
                + self.processor.image_start_tag
                + f"{self.image_end_tag}\nFirst, summarize the input prompt by keeping only explicitly stated visual facts\n"
            )
            input_ids = self.processor.tokenizer.encode(full_prompt)
            input_ids_list.append(torch.LongTensor(input_ids))

        pad_id = self.processor.tokenizer.encode(self.pad, add_special_tokens=False)[0]
        boi_token_id = self.processor.tokenizer.encode(
            self.image_start_tag, add_special_tokens=False
        )[0]
        max_len = max(len(ids) + self.image_token_num_per_image for ids in input_ids_list)

        images = torch.stack(image_tensors).to(self.device)
        image_embeds = self.model.aligner(self.model.vision_model(images))

        pad_embeds = self.model.language_model.get_input_embeddings()(torch.LongTensor([pad_id]).to(self.device))

        hidden_dim = image_embeds.shape[-1]

        total_rows = sum(group_sizes)
        token_embeds = pad_embeds.repeat(total_rows, max_len, 1).to(dtype=image_embeds.dtype, device=self.device)
        attention_mask = torch.zeros(
            (total_rows, max_len), dtype=torch.long, device=self.device
        )

        pointer = 0
        for i in range(batch_size):
            input_ids = input_ids_list[i]
            split_point_tensor = (input_ids == boi_token_id).nonzero(as_tuple=True)[0]
            split_point = split_point_tensor[-1]
            text_embeds = self.model.language_model.get_input_embeddings()(
                input_ids.to(self.device)
            )

            left_embeds = text_embeds[: split_point + 1].clone()
            right_embeds = text_embeds[split_point + 1 :].clone()

            for j in range(group_sizes[i]):
                combined = torch.cat([left_embeds, image_embeds[pointer + j], right_embeds])
                seq_len = len(combined)
                pad_len = max_len - seq_len
                token_embeds[pointer + j, pad_len:] = combined
                attention_mask[pointer + j, pad_len:] = 1

            pointer += group_sizes[i]

        return token_embeds, attention_mask

    def prepare_feedback_img_input_embeds(self, prompts: List[str], feedbacks: List[str], image_tensors: List[torch.Tensor]):
        batch_size = len(feedbacks)
        system_prompt = EDIT_TEMPLATE.format(image_start_tag=self.image_start_tag, image_end_tag=self.image_end_tag)

        input_ids_list = []

        assert len(prompts) == len(feedbacks)

        for prompt, feedback in zip(prompts, feedbacks):

            conversation = [
                {
                    "role": "<|User|>",
                    "content": (
                        f"{self.image_start_tag}{self.image_tag}{self.image_end_tag}\n"
                        "Please edit the image as instructed.\n"
                        "The FEEDBACK will be given as multiple steps (Step 1, Step 2, ...). "
                        "You MUST apply all steps in order and produce a final image reflecting all changes.\n"
                        f"INPUT_PROMPT: {prompt}\n"
                        f"FEEDBACK: \n{feedback}"
                    ),
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.processor.sft_format,
                system_prompt=system_prompt,
            )
            
            full_prompt = sft_format + self.processor.image_start_tag
            input_ids = self.processor.tokenizer.encode(full_prompt)
            input_ids_list.append(torch.LongTensor(input_ids))

        boi_token_id = self.processor.tokenizer.encode(self.image_start_tag, add_special_tokens=False)[0]
        eoi_token_id = self.processor.tokenizer.encode(self.image_end_tag, add_special_tokens=False)[0]
        image_token_id = self.processor.tokenizer.encode(self.image_tag, add_special_tokens=False)[0]
        assistant_id = self.processor.tokenizer.encode("<|Assistant|>", add_special_tokens=False)[0]
        pad_id = self.processor.tokenizer.encode(self.pad, add_special_tokens=False)[0]

        max_len = max(len(ids) + self.image_token_num_per_image for ids in input_ids_list)
        
        images = torch.stack(image_tensors).to(self.device)

        with torch.no_grad():
            _, _, all_image_ids = self.model.gen_vision_model.encode(images)
            image_ids = all_image_ids[2]
            image_ids = image_ids.view(batch_size, -1)

        image_embeds = self.model.gen_aligner(self.model.gen_embed(image_ids))

        pad_embeds = self.model.language_model.get_input_embeddings()(torch.LongTensor([pad_id]).to(self.device))

        hidden_dim = image_embeds.shape[-1]
        token_embeds = pad_embeds.repeat(batch_size * 2, max_len, 1).to(dtype=image_embeds.dtype, device=self.device)
        attention_mask = torch.zeros((batch_size * 2, max_len), dtype=torch.long, device=self.device)

        for i in range(batch_size*2):
                  
            input_ids = input_ids_list[i//2]

            if i % 2 == 0:
                split_point_tensor = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
                split_point = split_point_tensor[0]
                text_embeds = self.model.language_model.get_input_embeddings()(input_ids.to(self.device))

                left_embeds = text_embeds[:split_point].clone()
                right_embeds = text_embeds[split_point + 1:].clone()
                combined_embeds = torch.cat([left_embeds, image_embeds[i//2], right_embeds])

            else:
                split_point_boi_tensor = (input_ids == image_token_id).nonzero(as_tuple=True)[0] - 1
                split_point_eoi_tensor = (input_ids == image_token_id).nonzero(as_tuple=True)[0] + 1
                split_point_assistant_tensor = (input_ids == assistant_id).nonzero(as_tuple=True)[0]

                split_point_boi = split_point_boi_tensor[0]
                split_point_eoi = split_point_eoi_tensor[0]
                split_point_assistant = split_point_assistant_tensor[0]

                text_embeds = self.model.language_model.get_input_embeddings()(input_ids.to(self.device))

                left_embeds = text_embeds[:split_point_boi + 1].clone()
                middle_embeds = text_embeds[split_point_eoi].unsqueeze(0).clone()
                right_embeds = text_embeds[split_point_eoi + 1:split_point_assistant + 2].clone()
                end_embeds = text_embeds[split_point_assistant + 2:].clone()

                right_pad_embeds = pad_embeds.repeat(right_embeds.size(0), 1)

                combined_embeds = torch.cat([left_embeds, image_embeds[i//2], middle_embeds, right_pad_embeds, end_embeds], dim=0)

            seq_len = len(combined_embeds)
            pad_len = max_len - seq_len        
            token_embeds[i, pad_len:] = combined_embeds
            attention_mask[i, pad_len:] = 1

        return token_embeds, attention_mask 

    # ------------------------------------------------------------------ #
    # Generation & correction
    # ------------------------------------------------------------------ #
    def _run_generation_loop(self, prepared_data: Dict[str, List]):
        prompts = prepared_data["prompts"]
        sample_paths = prepared_data["sample_paths"]

        inputs_embeds, attention_masks = self.prepare_input_embeds(prompts)

        for iteration in range(self.num_samples_per_prompt):
            seed_everything(iteration + self.config.seed, workers=True)
            images = self._run_simple_generation(inputs_embeds, attention_masks)
            self._save_images(images, sample_paths, iteration)

    def _run_simple_generation(self, inputs_embeds, attention_masks):
        generated_tokens = self.generate_img(inputs_embeds, attention_masks)
        images = self.decode_tokens(generated_tokens)
        return images

    def _run_self_correction_loop(self, prepared_data, current_step: int):
        prompts = prepared_data["prompts"]
        data_indices = prepared_data["data_indices"]
        stage_paths = prepared_data["stage_paths"]
        image_tensors = prepared_data["image_tensors"]
        group_sizes = prepared_data["group_sizes"]
        selected_indices = prepared_data["selected_indices"]
        prev_rationales = prepared_data["prev_rationales"]

        if not prompts:
            return []

        token_embeds, attention_masks = self.prepare_correction_input_embeds(
            prompts, image_tensors, group_sizes
        )
        rationales = self._run_feedback_generation(token_embeds, attention_masks)

        regen_list = []
        stage_name = f"correction_{current_step}"
        pointer = 0

        for i in range(len(prompts)):
            indices = selected_indices[i]
            if not indices:
                pointer += group_sizes[i]
                continue

            source_path = stage_paths[i]
            target_path = self._get_stage_outpath(data_indices[i], stage_name, create=True)
            samples_dir = os.path.join(target_path, "samples")
            os.makedirs(samples_dir, exist_ok=True)

            src_samples_dir = os.path.join(source_path, "samples")
            src_metadata = os.path.join(source_path, METADATA_FILENAME)
            dst_metadata = os.path.join(target_path, METADATA_FILENAME)
            if os.path.isfile(src_metadata):
                with open(src_metadata, "r") as f:
                    meta = json.load(f)
                meta["iteration"] = current_step
                with open(dst_metadata, "w") as f:
                    json.dump(meta, f)

            prev_rat_list = prev_rationales[i] or [None] * self.num_samples_per_prompt
            if len(prev_rat_list) < self.num_samples_per_prompt:
                prev_rat_list.extend(
                    [None] * (self.num_samples_per_prompt - len(prev_rat_list))
                )
            new_rat_list = list(prev_rat_list)

            for img_idx in range(self.num_samples_per_prompt):
                if img_idx not in indices:
                    src_img = os.path.join(src_samples_dir, f"{img_idx:05d}.png")
                    dst_img = os.path.join(samples_dir, f"{img_idx:05d}.png")
                    if os.path.exists(src_img):
                        shutil.copyfile(src_img, dst_img)

            for local_idx, img_idx in enumerate(indices):
                rationale = rationales[pointer + local_idx]
                parse_error = rationale.get("parse_error", False)
                src_img = os.path.join(src_samples_dir, f"{img_idx:05d}.png")
                dst_img = os.path.join(samples_dir, f"{img_idx:05d}.png")

                if parse_error:
                    rationale.update(
                        {
                            "needs_review": True,
                            "image_index": img_idx,
                            "correction_step": current_step,
                        }
                    )
                    new_rat_list[img_idx] = rationale
                    if os.path.exists(src_img):
                        shutil.copyfile(src_img, dst_img)
                    continue

                answers = rationale.get("pred_only_answer_list", [])
                feedback = rationale.get("pred_feedback", "")
                is_all_yes = bool(answers) and all("yes" in ans.lower() for ans in answers)
                no_feedback_needed = "no need to generate feedback" in feedback.lower()
                needs_review_flag = rationale.get("needs_review", True)
                needs_review = needs_review_flag and not (is_all_yes or no_feedback_needed)

                rationale.update(
                    {
                        "needs_review": needs_review,
                        "image_index": img_idx,
                        "correction_step": current_step,
                    }
                )
                new_rat_list[img_idx] = rationale

                if needs_review:
                    regen_list.append(
                        {
                            "prompt": prompts[i],
                            "image_tensor": image_tensors[pointer + local_idx],
                            "save_name": dst_img,
                            "feedback": feedback,
                        }
                    )
                else:
                    if os.path.exists(src_img):
                        shutil.copyfile(src_img, dst_img)

            with open(os.path.join(target_path, RATIONAL_FILENAME), "w") as f:
                json.dump(new_rat_list, f, indent=4)

            pointer += group_sizes[i]

        return regen_list

    def _run_feedback_generation(self, token_embeds: torch.Tensor, attention_masks: torch.Tensor) -> List[dict]:
        generated_texts = self.generate_text(token_embeds, attention_masks)
        
        res = []
        for text in generated_texts:
            stripped = text.strip()

            parsed_parts = self.formatting_evaluator._split_text_into_parts(stripped)
            
            if all(p is not None for p in parsed_parts) and len(parsed_parts) == 4:
                p1, p2, p3, p4 = parsed_parts
                
                pred_tup_parsed = self.formatting_evaluator._parse_part1(p2)
                
                pred_answer_paragraphs = self.formatting_evaluator._extract_answer_paragraphs(p3)
                pred_only_answer_list = [
                    self.formatting_evaluator._get_answer_from_paragraph(ans).strip()
                    for ans in pred_answer_paragraphs
                    if ans.strip()
                ]
                
                res.append({
                    "original_answer": stripped,
                    "pred_summarize": p1,
                    "pred_tuple": p2,
                    "pred_answer": pred_answer_paragraphs,
                    "pred_only_answer_list": pred_only_answer_list,
                    "pred_feedback": p4,
                    "needs_review": True,
                    "parse_error": False,
                })
            else:
                # Handle parse failure.
                res.append({
                    "original_answer": stripped,
                    "pred_summarize": None,
                    "pred_tuple": None,
                    "pred_answer": [],
                    "pred_only_answer_list": [],
                    "pred_feedback": "",
                    "needs_review": True,
                    "parse_error": True,
                    "error_msg": "FormattingEvaluator parsing failed: expected 4 parts but failed to split."
                })
        
        return res

    def _run_feedback_img_generation_loop(self, regen_list: List[Dict]):
        if not regen_list:
            return regen_list

        prompts = [item["prompt"] for item in regen_list]

        image_tensors = [item["image_tensor"] for item in regen_list]
        save_names = [item["save_name"] for item in regen_list]
        feedbacks = [item["feedback"] for item in regen_list]

        token_embeds, attention_masks = self.prepare_feedback_img_input_embeds(
           prompts, feedbacks, image_tensors
        )
        generated_tokens = self.generate_img(token_embeds, attention_masks)
        images = self.decode_tokens(generated_tokens)

        for img, save_name in zip(images, save_names):
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            PIL.Image.fromarray(img).save(save_name)

        return regen_list

    # ------------------------------------------------------------------ #
    # Core model I/O helpers
    # ------------------------------------------------------------------ #
    def generate_text(self, inputs_embeds, attention_masks):
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_masks,
            max_new_tokens=self.max_reflect_len,
            use_cache=True,
            do_sample=True,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            top_k=self.txt_top_k,
            top_p=self.txt_top_p,
            temperature=self.temperature,
        )
        answer = self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return answer

    @torch.inference_mode()
    def generate_img(self, inputs_embeds, attention_masks):
        batch_size = inputs_embeds.shape[0] // 2
        generated_tokens = torch.zeros(
            (batch_size, self.image_token_num_per_image),
            dtype=torch.int,
            device=self.device,
        )

        attention_mask = attention_masks
        position_ids = attention_mask.long().cumsum(1) - 1
        position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        past_len = attention_mask.sum(dim=1, keepdim=True).long()

        past_key_values = None
        for i in range(self.image_token_num_per_image):
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            hidden_states = outputs.last_hidden_state
            past_key_values = outputs.past_key_values
            logits = self.model.gen_head(hidden_states[:, -1, :])

            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + self.cfg_weight * (logit_cond - logit_uncond)

            next_token = self._sample_from_logits(logits, is_text=False)
            generated_tokens[:, i] = next_token.squeeze(-1)

            next_token_pair = next_token.repeat(1, 2).view(-1)
            inputs_embeds = self.model.prepare_gen_img_embeds(next_token_pair).unsqueeze(1)

            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long, device=self.device)], dim=1)
            position_ids = past_len.clone()
            past_len += 1

        return generated_tokens

    def decode_tokens(self, generated_tokens: torch.Tensor) -> np.ndarray:
        batch_size = generated_tokens.shape[0]
        shape = [batch_size, 8, self.img_size // self.patch_size, self.img_size // self.patch_size]

        dec = self.model.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=shape)
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

        return dec

    def _sample_from_logits(self, logits: torch.Tensor, is_text: bool) -> torch.Tensor:
        if is_text:
            top_k = self.txt_top_k
            top_p = self.txt_top_p
        else:
            top_k = self.img_top_k
            top_p = self.img_top_p

        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        
        probs = torch.softmax(logits / self.temperature, dim=-1)
        
        if top_p:
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p 
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
            
        return next_token   

    def _save_images(self, images, sample_paths, iteration):
        for img, path in zip(images, sample_paths):
            dst = os.path.join(path, f"{iteration:05d}.png")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if isinstance(img, np.ndarray):
                PIL.Image.fromarray(img).save(dst)
            elif isinstance(img, PIL.Image.Image):
                img.save(dst)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")


# ---------------------------------------------------------------------- #
# Trainer / Config helpers
# ---------------------------------------------------------------------- #
def get_trainer(config, device):
    trainer = Trainer(
        accelerator=device,
        devices=config.world_size,
        strategy="ddp",
        max_epochs=1,
        precision="bf16",
        callbacks=[ModelSummary(max_depth=2)],
    )
    return trainer


def load_config(cfg_path, overrides):
    with initialize(version_base=None, config_path=cfg_path, job_name="eval_sft_dpgbench"):
        cfg = compose(config_name="dpgbench", overrides=overrides)
    return cfg


def main(args):
    config = load_config(cfg_path=args.cfg_path, overrides=args.overrides)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if getattr(config, "dpgbench_path", None):
        config.save_path = os.path.join(config.dpgbench_path, "gen")
    elif config.save_path is not None and config.exp_name is not None:
        config.save_path = os.path.join(config.save_path, config.exp_name, config.task_name, "gen")
        os.makedirs(config.save_path, exist_ok=True)
    else:
        raise ValueError("save_path or exp_name not provided.")

    if config.type == "sft" and os.path.isdir(config.ckpt_path):
        print(f"Loading finetuned weights from: {config.ckpt_path}")
        config.model.model_path = config.ckpt_path
        model, processor = get_model(config)   

    lightning_module = JanusTestWrapper(config=config, model=model, processor=processor)
    lightning_module.setup("test")

    trainer = get_trainer(config, device)
    eval_datamodule = EvalDataModule(config)

    start_time = datetime.datetime.now()
    trainer.test(lightning_module, datamodule=eval_datamodule)

    end_time = datetime.datetime.now()
    elapsed_min = (end_time - start_time).total_seconds() / 60
    print("------------------------------------------")
    print(f"Elapsed Time: {elapsed_min:.2f} minutes")
    print("------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="../../configs")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--s_idx", type=int, default=None)
    parser.add_argument("--e_idx", type=int, default=None)
    args, unknown = parser.parse_known_args()
    main(args)