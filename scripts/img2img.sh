python MorphDiff/scripts/gene_img2img_5c.py  --init_img_path data/drug_base_demo.npy \
    --gene_path data/gene_expression_demo.npy \
    --outdir ../results/drug_inference_demo/  --output_name  ../results/drug_inference_demo.npy \
    --model_path MorphDiff/checkpoints/drug_ldm.ckpt \
    --config_path MorphDiff/configs/ldm/morph_5c.yaml --scale 1.0 --strength 0.55 \
    --seed 10
   
