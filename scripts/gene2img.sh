python gene2img_dpm.py --gene_path data/gene_expression.npy  \
        --outdir result/g2i/ --output_name /result/g2i.npy  --H 128 --W 128 \
        --model_path /model/drug_ldm.ckpt \
        --config_path config/morph_5c.yaml --scale 1.0 --batch_size 2 --ddim_steps 100