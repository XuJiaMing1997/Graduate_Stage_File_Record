source activate pytorch36

# 6.18
python train_pro.py --lr_start 0.0001 --save_step 2 --eval_step 2 --dropout_rate 0.6 --if_weight_softmax --weight_softmax_value 4.0 --zoom_out_pad_prob 0.3 --if_load_SBP

# python train_pro.py --lr_start 0.0001 --save_step 2 --eval_step 2 --dropout_rate 0.6 --if_random_erase --zoom_out_pad_prob 0. --if_use_reid_dataset --reid_list_dir ./Market_train_OP_label.list --folder_dir ../Dataset_256x128/Market/bounding_box_train

# python train_pro.py --lr_start 0.0001 --save_step 2 --eval_step 2 --dropout_rate 0.6 --if_random_erase --zoom_out_pad_prob 0. --if_use_reid_dataset --reid_list_dir ./Market_train_OP_label.list --folder_dir ../Dataset_256x128/Market/bounding_box_train
# python train_pro.py --lr_start 0.0001 --save_step 2 --eval_step 2 --dropout_rate 0.6 --if_random_erase --zoom_out_pad_prob 0. --if_use_reid_dataset --reid_list_dir ./Market_train_OP_label.list --folder_dir ../Dataset_256x128/Market/bounding_box_train --if_load_SBP
