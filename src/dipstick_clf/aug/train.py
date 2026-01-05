"""Dipstick image data augmentation"""

# Imports
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import UNet2DConditionModel
from diffusers import DDPMScheduler
from diffusers import AutoencoderKL
from diffusers import DreamBoothLoraTrainer
from transformers import AutoTokenizer
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

# Config
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
INSTANCE_TOKEN = "hygiecatch"  # unique token for your dipstick
INSTANCE_PROMPT = f"a photo of a {INSTANCE_TOKEN} urine dipstick"
IMAGES_DIR = PROCESSED_DATA_DIR / "dipstick_test/no_aug"
OUTPUT_DIR = MODELS_DIR / "dipstick_aug"

LEARNING_RATE = 1e-4
TRAIN_STEPS = 800
BATCH_SIZE = 1

# Load models
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
text_encoder = torch.load
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")

# Setup
trainer = DreamBoothLoraTrainer(
    instance_data_dir=IMAGES_DIR,
    instance_prompt=INSTANCE_PROMPT,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    output_dir=OUTPUT_DIR,
    train_batch_size=BATCH_SIZE,
    max_train_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler="constant",
    lr_warmup_steps=0,
)

# Train
trainer.train()

# Save LoRA weights
trainer.save_lora_weights(OUTPUT_DIR)

print("Training complete. LoRA weights saved.")
