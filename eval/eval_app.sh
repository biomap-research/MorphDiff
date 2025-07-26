generated_output='/data2/wangxuesong/morphdata/data_1115/application/used_for_fig4/IMPA_inside_i2i/images'
ground_truth='/data2/wangxuesong/morphdata/data_1115/application/for_fig5_expand/inside_target/train_test/target_real_total_images'
output='target_IMPA'
stats_path='target_real.npz'
# python cell_type_classifier.py --out_channel ${channel} --data_dir /mnt/petrelfs/fanyimin/data_for_cpd/train $TRAIN_FLAGS $CLASSIFIER_FLAGS --val_data_dir ${val_data_dir} --noised ${noised} --resume_checkpoint /mnt/petrelfs/fanyimin/guided-diffusion/output/cpd_classifier_False/model00${resume}.pt
python calculate_dnc.py --output ${output} --generated_output ${generated_output} --ground_truth ${ground_truth}  --stats ${stats_path}
