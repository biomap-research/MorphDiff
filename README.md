# MorphDiff


The codes in MorphDiff are modified based on the stable diffusion v1 main framework.

Original framework: [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)

---

## 📊 Training Process

The training process consists of two main stages:

### 1. Training the Variational AutoEncoder (VAE)

- **Example script**: `MorphDiff/vae.sh`
- **Configuration file**: `MorphDiff/configs/autoencoder/autoencoder_kl_32x32x4_5c.yaml`

Detailed information about input file paths and parameters can be found in the configuration file.

### 2. Training the Latent Diffusion Model

- **Example script**: `MorphDiff/dm.sh`
- **Configuration file**: `MorphDiff/configs/ldm/morph_5c.yaml`

Detailed information about input file paths and parameters can be found in the configuration file.

---

## 🚀 Inference Stage

Examples for both gene to image and image to image modes are provided as shell scripts: 
`MorphDiff/scripts/gene2img.sh` and `MorphDiff/scripts/img2img.sh`.

### 1. Gene to Image Mode
Example command from `gene2img.sh`:
```
python gene2img_dpm.py \
  --gene_path data/gene_expression.npy \
  --outdir result/g2i/ \
  --output_name /result/g2i.npy \
  --H 128 \
  --W 128 \
  --model_path /model/drug_ldm.ckpt \
  --config_path config/morph_5c.yaml \
  --scale 1.0 \
  --batch_size 2 \
  --ddim_steps 100
```
### 2. Image to Image Mode
Example command from `img2img.sh`:
```
python MorphDiff/scripts/gene_img2img_5c.py \
  --init_img_path data/drug_base_demo.npy \
  --gene_path data/gene_expression_demo.npy \
  --outdir ../results/drug_inference_demo/ \
  --output_name ../results/drug_inference_demo.npy \
  --model_path MorphDiff/checkpoints/drug_ldm.ckpt \
  --config_path MorphDiff/configs/ldm/morph_5c.yaml \
  --scale 1.0 \
  --strength 0.55 \
  --seed 10
```
---
