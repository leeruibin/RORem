import os
import torch
import time
import json
from PIL import Image
from pipelines.RORem_discriminator_pipeline import StableDiffusionXLDiscriminatorPipeline
from peft import LoraConfig
from model.unet_sdxl_discriminator import UNet2DConditionDiscriminator
import argparse
from diffusers.utils import load_image

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        help="Path to pretrained LDM teacher model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--RORem_discriminator",
        type=str,
        default=None,
        required=True,
        help="Path to pretrain RORem Unet",
    )
    parser.add_argument(
        "--RORem_LoRA",
        type=str,
        default=None,
        required=True,
        help="Path to pretrain RORem Unet",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="Path to the mask image.",
    )
    parser.add_argument(
        "--edited_path",
        type=str,
        default=None,
        help="Path to save the removal result.",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--resolution",
        default=512,
        type=int
    )
    parser.add_argument(
        "--dilate_size",
        default=20,
        type=int,
        help="dilate the mask"
    )
    parser.add_argument(
        "--threshold",
        default=0.5,
        type=float,
    )
    args = parser.parse_args()

    return args

def main(args):
    # load Pipeline
    pipeline = StableDiffusionXLDiscriminatorPipeline.from_pretrained(
        args.pretrained_model,variant="fp16",torch_dtype=torch.float16
    ).to("cuda")

    # initail Unet part
    unet = UNet2DConditionDiscriminator.from_pretrained(
        args.pretrained_model, subfolder="unet",low_cpu_mem_usage=False
    ).to("cuda",dtype=torch.float16)

    # load pretrain LoRA and cls_pred_branch
    unet_lora_config = LoraConfig(
        r=4,
        use_dora=False,
        lora_alpha=4,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)

    ckpt_path = args.RORem_discriminator

    unet.load_attn_procs(ckpt_path,weight_name="pytorch_lora_weights.safetensors")
    cls_dict = torch.load(f"{ckpt_path}/cls_pred_branch.pt")
    unet.load_state_dict(cls_dict,strict=False)

    pipeline.unet = unet

    source_image = load_image(args.input_path).resize([512,512])
    mask_image = load_image(args.mask_path).resize([512,512])
    GT_image = load_image(args.edited_path).resize([512,512])
    predict_score = pipeline(prompt="",height=512,width=512,image=source_image,mask_image=mask_image,edited_image=GT_image)
    if predict_score > args.threshold:
        print("This is a valid removal case.")
    else:
        print("This is a failed removal case.")
    

if __name__ == "__main__":
    args = parse_args()
    main(args)


