# MorphDiff

The codes in MorphDiff are modified based on the stable diffusion v1 main framework (https://github.com/CompVis/stable-diffusion).

There are two stages in training process.

training the Variantional AutoEncoder (VAE). An example script is provided as MorphDiff/vae.sh. The relative config file are provided as MorphDiff/configs/autoencoder/autoencoder_kl_32x32x4_5c.yaml. The detailed information of path of input files and parameters can be found in the config file.

training the latent diffusion model. An example script is provided as MorphDiff/dm.sh. The relative config file are provided as MorphDiff/configs/ldm/morph_5c.yaml. The detailed information of path of input files and parameters can be found in the config file.

For the inference stage: Examples of gene to image and image to image are provided as MorphDiff/scripts/gene2img.sh and MorphDiff/scripts/img2img.sh.

The gene to image mode can ran as: python gene2img_dpm.py --gene_path data/gene_expression.npy
--outdir result/g2i/ --output_name /result/g2i.npy --H 128 --W 128
--model_path /model/drug_ldm.ckpt
--config_path config/morph_5c.yaml --scale 1.0 --batch_size 2 --ddim_steps 100

The image to image mode can be ran as: python MorphDiff/scripts/gene_img2img_5c.py --init_img_path data/drug_base_demo.npy
--gene_path data/gene_expression_demo.npy
--outdir ../results/drug_inference_demo/ --output_name ../results/drug_inference_demo.npy
--model_path MorphDiff/checkpoints/drug_ldm.ckpt
--config_path MorphDiff/configs/ldm/morph_5c.yaml --scale 1.0 --strength 0.55
--seed 10

The analysis foler provides jupyter notebook to reproduce some figures in the paper.
