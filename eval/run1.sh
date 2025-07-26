
#train_drug_name='BRD-K93632104'
train_drug_name='BRD-A02481876'
train_drug="/data2/wangxuesong/morphdata/data_1115/application/on_${train_drug_name}/${train_drug_name}"
nearest_10_drug_base="/data2/wangxuesong/morphdata/data_1115/application/on_${train_drug_name}/nearest_10"
#nearest_10_drug=('BRD-K04488512' 'BRD-K09108485' 'BRD-K22812176' 'BRD-K30103771' 'BRD-K31780647' 'BRD-K38710631' 'BRD-K48020855' 'BRD-K50718861' 'BRD-K56381552' 'BRD-K81997160')
nearest_10_drug=('BRD-A24369193' 'BRD-K38710631' 'BRD-K05261060' 'BRD-K09366011' 'BRD-K35879814' 'BRD-A01848933' 'BRD-K42797038' 'BRD-K56572622' 'BRD-K50718861' 'BRD-K61432982')
furtherest_10_drug_base="/data2/wangxuesong/morphdata/data_1115/application/on_${train_drug_name}/furtherest_10"
#furtherest_10_drug=('BRD-A57353172' 'BRD-A77440546' 'BRD-A94885304' 'BRD-K02078126' 'BRD-K05060234' 'BRD-K05392698' 'BRD-K37912617' 'BRD-K45073068' 'BRD-K49040541' 'BRD-K74817844')
furtherest_10_drug=('BRD-K49040541' 'BRD-K45073068' 'BRD-A77440546' 'BRD-K37912617' 'BRD-A57353172' 'BRD-K05060234' 'BRD-K02078126' 'BRD-K05392698' 'BRD-K74817844' 'BRD-A94885304')
#output='orf_ood_img2img'
stats_path="${train_drug_name}.npz"
# python cell_type_classifier.py --out_channel ${channel} --data_dir /mnt/petrelfs/fanyimin/data_for_cpd/train $TRAIN_FLAGS $CLASSIFIER_FLAGS --val_data_dir ${val_data_dir} --noised ${noised} --resume_checkpoint /mnt/petrelfs/fanyimin/guided-diffusion/output/cpd_classifier_False/model00${resume}.pt
for drug in "${nearest_10_drug[@]}"; do
    output="./on_${train_drug_name}/near/${drug}"  
    echo ${output}
    python calculate_dnc.py --output ${output} --generated_output ${nearest_10_drug_base}/${drug} --ground_truth ${train_drug}  --stats "./on_${train_drug_name}/near/${stats_path}"
    output="./on_${train_drug_name}/near/${drug}_generate"
    echo ${output}
    python calculate_dnc.py --output ${output} --generated_output "${nearest_10_drug_base}/${drug}_generate" --ground_truth "${nearest_10_drug_base}/${drug}"  --stats "./on_${train_drug_name}/near/${drug}.npz"
    output="./on_${train_drug_name}/near/${drug}_generate_i2i"
    echo ${output}
    python calculate_dnc.py --output ${output} --generated_output "${nearest_10_drug_base}/${drug}_generate_i2i" --ground_truth "${nearest_10_drug_base}/${drug}"  --stats "./on_${train_drug_name}/near/${drug}.npz"
done

for drug in "${furtherest_10_drug[@]}"; do
    output="./on_${train_drug_name}/far/${drug}"
    echo ${output}    
    python calculate_dnc.py --output ${output} --generated_output ${furtherest_10_drug_base}/${drug} --ground_truth ${train_drug}  --stats "./on_${train_drug_name}/far/${stats_path}"
    output="./on_${train_drug_name}/far/${drug}_generate"
    echo ${output}
    python calculate_dnc.py --output ${output} --generated_output "${furtherest_10_drug_base}/${drug}_generate" --ground_truth "${furtherest_10_drug_base}/${drug}"  --stats "./on_${train_drug_name}/far/${drug}.npz"
    output="./on_${train_drug_name}/far/${drug}_generate_i2i"
    echo ${output}
    python calculate_dnc.py --output ${output} --generated_output "${furtherest_10_drug_base}/${drug}_generate_i2i" --ground_truth "${furtherest_10_drug_base}/${drug}"  --stats "./on_${train_drug_name}/far/${drug}.npz"
done