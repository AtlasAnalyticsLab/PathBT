python main_clam.py  --image_size 224  --source_level 2 --target_level 2 --exp_code sup-410 --model_type clam_sb --drop_out --early_stopping --lr 2e-4 --k 1 --label_frac 1  --weighted_sample --bag_loss ce --inst_loss svm --task task_2_tumor_subtyping  --log_data

# change in main_clam the path to the KGH_slides.csv
# change emb_dim if benchmark or swin or else
# change result dir if necessary