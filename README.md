# RORem: Training a Robust Object Remover with Human-in-the-Loop

<a href='https://arxiv.org/abs/2501.00740'><img src='https://img.shields.io/badge/arXiv-2501.00740-b31b1b.svg'></a>

[Ruibin Li](https://github.com/leeruibin)<sup>1,2</sup>
| [Tao Yang](https://github.com/yangxy)<sup>3</sup> | 
[Song Guo](https://scholar.google.com/citations?user=Ib-sizwAAAAJ&hl=en)<sup>4</sup> | 
[Lei Zhang](https://scholar.google.com/citations?user=wzdCc-QAAAAJ&hl=en)<sup>1,2</sup> | 

<sup>1</sup>The Hong Kong Polytechnic University, <sup>2</sup>OPPO Research Institute, <sup>3</sup>ByteDance, <sup>4</sup>The Hong Kong University of Science and Technology.

## 📌 TODO
<!-- ✅ -->
- [x] ✅ RORem Dataset
- [x] ✅ Training Code
- [ ] ⬜️ Update dataset to huggingface
- [ ] ⬜️ RORem Model, LoRA, Discriminator
- [ ] ⬜️ Make huggingface demo

## 😃 prepare enviroment

```
git clone https://github.com/leeruibin/RORem.git
cd RORem
conda env create -n environment.yaml
conda activate RORem
```

Install xformers to speedup the training, note that the xformers version should match torch version.

```
pip install xformers==0.0.28.post3
```

We use wandb to record the intermediate state during the training process, so make sure you finish the following process

```
pip install wandb
wandb login
```
enter the WANDB_API_KEY in the shell or direct export WANDB_API_KEY=<your-api-key> to the environment variable.

## ⭐ Download RORem dataset

| Dataset    |  Download                                                  |
| -----------| --------------------------------------------               |
| RORem&RORD | [Google cloud](https://drive.google.com/file/d/1sE6IOhHNCKiwFLW4a2ZWcwU4_bhvGcSA/view?usp=sharing) (73.15GB) |
| Mulan      | [Google cloud](https://drive.google.com/file/d/1-dX5GfxyGEGBSfFeBgl5vMH9ODdCpbuq/view?usp=sharing) (3.26GB) |
| Final HR   | [Google cloud](https://drive.google.com/file/d/1S3p_yLjPuhZbh7S15actNaAOEPvUlW5C/view?usp=sharing) (7.9GB) |
| All        | [Google cloud](https://drive.google.com/drive/folders/1KDwQ0MF2yJ78X6Ketw4oh8jSokQYS63Q?usp=sharing) (84.31GB) |

Please note that we employed the SafeStableDiffusionSafetyChecker to filter out inappropriate content, which may result in minor discrepancies between the final image-text pairs and those presented in the original paper.

For each dataset, we build folder structure as:

```
.
├── source
├── mask
├── GT
└── meta.json #
```
The meta.json file record the triple as:
```
{"source":"source/xxx.png","mask":"mask/xxx.png","GT":"GT/xxx.png"}
```

By path the absolute path of meta.json, the training script can parse the path of each triple.

## 🔥 Training

### To train RORem, with the following training script

```
accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    train_RORem.py \
    --train_batch_size 16 \
    --output_dir <your_path_to_save_checkpoint> \
    --meta_path xxx/Final_open_RORem/meta.json \
    --max_train_steps 50000 \
    --random_flip \
    --resolution 512 \
    --pretrained_model_name_or_path diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
    --mixed_precision fp16 \
    --checkpoints_total_limit 5 \
    --checkpointing_steps 5000 \
    --learning_rate 5e-5 \
    --validation_steps 2000 \
    --seed 4 \
    --report_to wandb \
```

Using Deepspeed zero2 requires less GPU memory.

```
accelerate launch --config_file config/deepspeed_config.yaml \
    --multi_gpu \
    --num_processes 8 \
    train_RORem.py \
    --train_batch_size 16 \
    --output_dir <your_path_to_save_checkpoint> \
    --meta_path xxx/Final_open_RORem/meta.json \
    --max_train_steps 50000 \
    --random_flip \
    --resolution 512 \
    --pretrained_model_name_or_path diffusers/stable-diffusion-xl-1.0-inpainting-0.1 \
    --mixed_precision fp16 \
    --checkpoints_total_limit 5 \
    --checkpointing_steps 5000 \
    --learning_rate 5e-5 \
    --validation_steps 2000 \
    --seed 4 \
    --report_to wandb \
```

OR you can directly submit the training shell as:

```
bash run_train_RORem.sh
```

## ⏰ Update
The code and model will be ready soon.

⭐: If RORem is helpful to you, please help star this repo. Thanks! 🤗

## 🌟 Overview Framework

![pipeline](figures/pipeline.png)


Overview of our training data generation and model training process. In stage 1, we gather 60K training triplets from open-source datasets to train an initial removal model. In stage 2, we apply the trained model to a test set and engage human annotators to select high-quality samples to augment the training set. In stage 3, we train a discriminator using the human feedback data, and employ it to automatically annotate high quality training samples. We iterate stages 2\&3 for several rounds, ultimately obtaining over 200K object removal training triplets as well as the trained model.

![number](figures/data_collection.png)

## 🌟 Visual Results

### Quantative comparsion

We invite human annotators to evaluate the success rate of different methods. Furthermore, by refining our discriminator, we can see that the success rates estimated by $D_{\phi}$ closely align with human annotations in the test set (with deviations less than 3% in most cases). This indicates that our trained $D_{\phi}$ effectively mirrors human preferences.

![result](figures/quantative_result.png)

### Qualitative Comparisons
![result](figures/result.png)

<!-- ### Citations
If our code helps your research or work, please consider citing our paper.
The following are BibTeX references: -->

### License
This project is released under the [Apache 2.0 license](LICENSE).

## BibTeX

```bibtex
@article{li2024RORem,
  title={RORem: Training a Robust Object Remover with Human-in-the-Loop},
  author={Ruibin Li and Tao, Yang and Song, Guo and Lei, Zhang},
  year={2025},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
}
```

## Acknowledgements

This implementation is developed based on the [diffusers](https://github.com/huggingface/diffusers/) library and utilizes the [Stable Diffusion XL-inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) model. We would like to express our gratitude to the open-source community for their valuable contributions.

<details>

<summary>statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=leeruibin/RORem)
![stars](https://img.shields.io/github/stars/leeruibin/RORem)
![forks](https://img.shields.io/github/forks/leeruibin/RORem)

</details>