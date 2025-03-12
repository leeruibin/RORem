from diffusers import AutoPipelineForInpainting
import torch
import os
from diffusers import UNet2DConditionModel
import argparse
from myutils.img_util import dilate_mask
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
        "--RORem_unet",
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
        "--save_path",
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
    args = parser.parse_args()

    return args

def main(args):

    if args.pretrained_model is None:
        pretrain_path = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    else:
        pretrain_path = args.pretrained_model
    # load pretrained SDXL-inpainting model
    pipe_edit = AutoPipelineForInpainting.from_pretrained(
        pretrain_path,
        torch_dtype=torch.float16, 
        variant="fp16"
    )

    # load RORem Unet
    unet = UNet2DConditionModel.from_pretrained(args.RORem_unet).to("cuda",dtype=torch.float16)
    print(f"Finish loading unet from {args.pretrained_unet}!!")
    pipe_edit.unet = unet

    pipe_edit.to("cuda")



    height = width = args.resolution
    image_name = args.image_path.split("/")[-1]
    if args.save_path is None:
        save_folder = "removal_result"
        os.makedirs(save_folder,exist_ok=True)
        args.save_path = f"{save_folder}/{image_name}"
    else:
        save_folder = os.path.dirname(args.save_path)
        os.makedirs(save_folder,exist_ok=True)
    input_image = load_image(args.input_path).resize((args.resolution,args.resolution))
    input_mask = load_image(args.mask_path).resize((args.resolution,args.resolution))
    if args.dilate_size != 0:
        mask_image = dilate_mask(mask_image,args.dilate_size)
    prompts = ""
    Removal_result = pipe_edit(
            prompt=prompts,
            height=height,
            width=width,
            image=input_image,
            mask_image=input_mask,
            guidance_scale=1.,
            num_inference_steps=50,  # steps between 15 and 30 also work well
            strength=0.99,  # make sure to use `strength` below 1.0
        ).images[0]

    Removal_result.save(save_folder)
    
    # we also find by adding these prompt, the model can work even better
    # prompts = "4K, high quality, masterpiece, Highly detailed, Sharp focus, Professional, photorealistic, realistic"
    # negative_prompts = "low quality, worst, bad proportions, blurry, extra finger, Deformed, disfigured, unclear background"
    # Removal_result = pipe_edit(
    #         prompt=prompts,
    #         negative_prompt=negative_prompts,
    #         height=height,
    #         width=width,
    #         image=input_image,
    #         mask_image=input_mask,
    #         guidance_scale=1.,
    #         num_inference_steps=50,  # steps between 15 and 30 also work well
    #         strength=0.99,  # make sure to use `strength` below 1.0
    #     ).images[0]

    # Removal_result.save(save_folder)

if __name__ == "__main__":
    args = parse_args()
    main(args)