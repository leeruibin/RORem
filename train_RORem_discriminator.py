#!/usr/bin/env python
# coding=utf-8
import os
import argparse
import logging
import math

import shutil
import warnings
from pathlib import Path

from huggingface_hub import create_repo
import accelerate
import datasets
from datasets import Dataset, Image
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from myutils.img_util import meta_to_discriminator_dataset_format
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from model.unet_sdxl_discriminator import UNet2DConditionDiscriminator
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from pipelines.RORem_inpaint_pipeline import StableDiffusionXLInpaintPipeline

from diffusers.utils import deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
import time

from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict


if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")

TORCH_DTYPE_MAPPING = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")



def parse_args():
    parser = argparse.ArgumentParser(description="Script to train Stable Diffusion XL for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_unet",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_lora",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--comments",
        type=str,
        default=None,
        help="Left a comment for the current run.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--vae_precision",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help=(
            "The vanilla SDXL 1.0 VAE can cause NaNs due to large activation values. Some custom models might already have a solution"
            " to this problem, and this flag allows you to use mixed precision to stabilize training."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--meta_path",
        type=str,
        nargs="*",
        default=[
                 "xxx/meta.json","xxx/meta.json"
                 ],
        help=(
            "The path for meta info about the dataset."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiment/SDXL_discriminator",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default="20240421", help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution."
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt)",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )

    parser.add_argument('--trainable_modules', nargs='*', type=str, default=["cls_pred_branch"])

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", default=True, help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def convert_to_np(image, resolution):
    if isinstance(image, str):
        image = PIL.pilimage.open(image)
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def convert_to_np_single(image, resolution):
    if isinstance(image, str):
        image = PIL.pilimage.open(image)
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)[:1,:,:]

def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )

    # we use UNet discriminator here
    unet = UNet2DConditionDiscriminator.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant,low_cpu_mem_usage=False
    )
    if args.pretrained_unet is not None:
        unet = UNet2DConditionDiscriminator.from_pretrained(args.pretrained_unet)
    
    logger.info("Initializing the UNet from the pretrained UNet.")
    in_channels = 9
    unet.register_to_config(in_channels=in_channels)
    unet.requires_grad_(False)

    # TODO we add lora here and put class_pre_branch to trainable
    if args.trainable_modules is None or len(args.trainable_modules)==0:
        unet.requires_grad_(True)
    else:
        # attn1 is self attention, attn2 is cross attention
        for name, module in unet.named_modules():
            if name.endswith(tuple(args.trainable_modules)):
                print(f'{name} in <unet> will be optimized.' )
                for params in module.parameters():
                    params.requires_grad = True

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    # model.save_pretrained(os.path.join(output_dir, "unet"))
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()
                StableDiffusionXLInpaintPipeline.save_lora_weights(
                    output_dir,
                    unet_lora_layers=unet_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):

            for i in range(len(models)):
                model = models.pop()
                unet_ = model
                del load_model
            lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
            unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
            unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
            incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
            # Make sure the trainable params are in float32. This is again needed since the base models
            # are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            if args.mixed_precision == "fp16":
                models = [unet_]

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )


    optimizer_cls = torch.optim.AdamW

    if not isinstance(args.meta_path,list):
        meta_folder = os.path.dirname(args.meta_path)
    else:
        meta_folder = None
    dataset_dict = meta_to_discriminator_dataset_format(args.meta_path,meta_folder)

    dataset = Dataset.from_dict(dataset_dict).cast_column("input_image", Image()).cast_column("edited_image", Image()).cast_column("mask", Image())

    # Preprocessing the datasets.

    # 6. Get the column names for input/target.
    dataset_columns = ("input_image", "edited_image", "edit_prompt")
    original_image_column, edited_image_column, edit_prompt_column = dataset_columns

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        warnings.warn(f"weight_dtype {weight_dtype} may cause nan during vae encoding", UserWarning)

    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        warnings.warn(f"weight_dtype {weight_dtype} may cause nan during vae encoding", UserWarning)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions, tokenizer):
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    def preprocess_images(examples):
        original_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[original_image_column]]
        )
        edited_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[edited_image_column]]
        )

        mask_images = np.concatenate([convert_to_np_single(image,args.resolution) for image in examples["mask"] ])
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = np.concatenate([original_images, edited_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1

        mask_tensor = torch.tensor(mask_images)
        mask_tensor = mask_tensor / 255
        concat_mask = torch.cat([images,mask_tensor])
        concat_mask = train_transforms(concat_mask)
        image_length = images.shape[0]
        images,mask_tensor = concat_mask[:image_length,:,:],concat_mask[image_length:,:,:]
        return images,mask_tensor

    # Load scheduler, tokenizer and models.
    tokenizer_1 = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )
    text_encoder_cls_1 = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_2 = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder_1 = text_encoder_cls_1.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    text_encoder_2 = text_encoder_cls_2.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
    )

    # We ALWAYS pre-compute the additional condition embeddings needed for SDXL
    # UNet as the model is already big and it uses two text encoders.
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    tokenizers = [tokenizer_1, tokenizer_2]
    text_encoders = [text_encoder_1, text_encoder_2]

    # Freeze vae and text_encoders
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Set UNet to trainable.
    # since we only train lora, we can put unet to the weight_dtype
    unet = unet.to(accelerator.device, dtype=weight_dtype)
    # unet.train()

    unet_lora_config = LoraConfig(
        r=args.rank,
        use_dora=False,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)
    pretrained_unet_dict = unet.state_dict()

    # load pretrain
    if args.pretrained_lora is not None:
        lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(args.pretrained_lora)
        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
        
        cls_parameter = torch.load(f"{args.pretrained_lora}/cls_pred_branch.pt")
        print(len(cls_parameter))
        for key,value in cls_parameter.items():
            if key in pretrained_unet_dict.keys():
                print(key)
        unet.load_state_dict(cls_parameter,strict=False)


    if args.mixed_precision != "fp32":
        models = [unet]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))

    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt(text_encoders, tokenizers, prompt):
        prompt_embeds_list = []

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
        return prompt_embeds, pooled_prompt_embeds

    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_prompts(text_encoders, tokenizers, prompts):
        prompt_embeds_all = []
        pooled_prompt_embeds_all = []

        for prompt in prompts:
            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt)
            prompt_embeds_all.append(prompt_embeds)
            pooled_prompt_embeds_all.append(pooled_prompt_embeds)

        return torch.stack(prompt_embeds_all), torch.stack(pooled_prompt_embeds_all)

    # Adapted from examples.dreambooth.train_dreambooth_lora_sdxl
    # Here, we compute not just the text embeddings but also the additional embeddings
    # needed for the SD XL UNet to operate.
    def compute_embeddings_for_prompts(prompts, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds_all, pooled_prompt_embeds_all = encode_prompts(text_encoders, tokenizers, prompts)
            add_text_embeds_all = pooled_prompt_embeds_all

            prompt_embeds_all = prompt_embeds_all.to(accelerator.device)
            add_text_embeds_all = add_text_embeds_all.to(accelerator.device)
        return prompt_embeds_all, add_text_embeds_all

    # Get null conditioning
    def compute_null_conditioning():
        null_conditioning_list = []
        for a_tokenizer, a_text_encoder in zip(tokenizers, text_encoders):
            null_conditioning_list.append(
                a_text_encoder(
                    tokenize_captions([""], tokenizer=a_tokenizer).to(accelerator.device),
                    output_hidden_states=True,
                ).hidden_states[-2]
            )
        return torch.concat(null_conditioning_list, dim=-1)

    null_conditioning = compute_null_conditioning()

    def compute_time_ids():
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        original_size = target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=weight_dtype)
        return add_time_ids.to(accelerator.device).repeat(args.train_batch_size, 1)

    add_time_ids = compute_time_ids()

    one_time_prompt_embeds_all, one_time_add_text_embeds_all = compute_embeddings_for_prompts([""], text_encoders, tokenizers)

    def preprocess_train(examples):
        # Preprocess images.
        # preprocessed_images = preprocess_images(examples)
        preprocessed_images,preprocess_mask = preprocess_images(examples)
        # Since the original and edited images were concatenated before
        # applying the transformations, we need to separate them and reshape
        # them accordingly.
        original_images, edited_images = preprocessed_images.chunk(2)
        original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)
        mask_images = preprocess_mask.reshape(-1,1,args.resolution, args.resolution)
        # Collate the preprocessed images into the `examples`.
        examples["original_pixel_values"] = original_images
        examples["edited_pixel_values"] = edited_images
        examples["mask_pixel_values"] = mask_images
        examples["score"] = torch.tensor(examples['score']).float()

        bsz = len(examples[edit_prompt_column])
        prompt_embeds_all, add_text_embeds_all = one_time_prompt_embeds_all.repeat(bsz,1,1,1), one_time_add_text_embeds_all.repeat(bsz,1,1)
        examples["prompt_embeds"] = prompt_embeds_all
        examples["add_text_embeds"] = add_text_embeds_all
        return examples

    with accelerator.main_process_first():
        train_dataset = dataset.shuffle(seed=args.seed).with_transform(preprocess_train)

    def collate_fn(examples):
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
        edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
        prompt_embeds = torch.concat([example["prompt_embeds"] for example in examples], dim=0)
        add_text_embeds = torch.concat([example["add_text_embeds"] for example in examples], dim=0)
        mask_pixel_values = torch.stack([example["mask_pixel_values"] for example in examples])
        mask_pixel_values = mask_pixel_values.to(memory_format=torch.contiguous_format).float()

        score_values = torch.stack([example["score"] for example in examples])
        score_values = score_values.to(memory_format=torch.contiguous_format).float()

        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "prompt_embeds": prompt_embeds,
            "add_text_embeds": add_text_embeds,
            "mask_pixel_values": mask_pixel_values,
            "score_values": score_values,
        }


    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler, test_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler, test_dataloader
    )


    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=TORCH_DTYPE_MAPPING[args.vae_precision])

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.

    if accelerator.is_main_process:
        accelerator.init_trackers("SDXL-Discriminator", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps


    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num test examples = {len(test_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    latent_size = args.resolution // 8

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            st_time = time.time()
            with accelerator.accumulate(unet):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                if args.pretrained_vae_model_name_or_path is not None:
                    edited_pixel_values = batch["edited_pixel_values"].to(dtype=weight_dtype)
                else:
                    edited_pixel_values = batch["edited_pixel_values"]

                # edited_pixel_values = edited_pixel_values * (batch["mask_pixel_values"] < 0.5)
                latents = vae.encode(edited_pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    latents = latents.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                timesteps = torch.zeros((bsz,),device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)


                # SDXL additional inputs
                encoder_hidden_states = batch["prompt_embeds"]
                add_text_embeds = batch["add_text_embeds"]

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                if args.pretrained_vae_model_name_or_path is not None:
                    original_pixel_values = batch["original_pixel_values"].to(dtype=weight_dtype)
                else:
                    original_pixel_values = batch["original_pixel_values"]

                original_image_embeds = vae.encode(original_pixel_values).latent_dist.sample() * vae.config.scaling_factor
                if args.pretrained_vae_model_name_or_path is None:
                    original_image_embeds = original_image_embeds.to(weight_dtype)

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                mask_pixel_values = batch["mask_pixel_values"]

                mask_pixel_values = torch.nn.functional.interpolate(
                    mask_pixel_values, size=(latent_size, latent_size)
                )

                concatenated_noisy_latents = torch.cat([noisy_latents, mask_pixel_values, original_image_embeds], dim=1)

                # Predict the noise residual and compute loss
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                concatenated_noisy_latents = concatenated_noisy_latents.to(dtype=weight_dtype)
                pred_label = unet(
                    concatenated_noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )

                target = batch["score_values"].to(dtype=weight_dtype)

                loss = F.mse_loss(pred_label.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                cost_time = time.time()-st_time

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss,"step_time":cost_time}, step=global_step)
                train_loss = 0.0

                if args.checkpointing_steps > 0 and (global_step % args.checkpointing_steps == 0 or global_step==1):
                    if accelerator.is_main_process:
                    # accelerator.wait_for_everyone()
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # accelerator.save_state(save_path)
                        # for discriminator, we save two part
                        # The first part is the tained LoRA
                        unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrap_model(unet)))
                        StableDiffusionXLInpaintPipeline.save_lora_weights(
                            save_path,
                            unet_lora_layers=unet_lora_layers_to_save
                        )
                        # The second part is the pre_cls branch
                        cls_pred_branch_dict = {}
                        for key,values in unwrap_model(unet).state_dict().items():
                            if "cls_pred_branch" in key:
                                cls_pred_branch_dict[key] = values
                        torch.save(cls_pred_branch_dict,os.path.join(save_path,"cls_pred_branch.pt"))

                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_version_path = os.path.join(args.output_dir,"final_unet")
        unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrap_model(unet)))
        StableDiffusionXLInpaintPipeline.save_lora_weights(
            final_version_path,
            unet_lora_layers=unet_lora_layers_to_save
        )
        # we just save lora parameters and pre_cls branch
        cls_pred_branch_dict = {}
        for key,values in unwrap_model(unet).state_dict().items():
            if "cls_pred_branch" in key:
                cls_pred_branch_dict[key] = values
        torch.save(cls_pred_branch_dict,os.path.join(final_version_path,"cls_pred_branch.pt"))

    accelerator.end_training()


if __name__ == "__main__":
    main()