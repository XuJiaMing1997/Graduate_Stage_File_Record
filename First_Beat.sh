source activate pytorch36

# MIE_SBP
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --save_step 4 --eval_step 2 --target_dataset_name CUHK03
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --save_step 4 --eval_step 2 --target_dataset_name DUKE

# SBP
# python train.py --target_dataset_name CUHK03
# python train.py --target_dataset_name DUKE

# OP pretrain
# for CUHK03
# python train_pro.py --lr_start 0.0001 --save_step 2 --eval_step 2 --dropout_rate 0.6 --if_weight_softmax --weight_softmax_value 4.0 --zoom_out_pad_prob 0.3 --if_load_SBP --SBP_dir ./SBP/model_save_CUHK03/pytorch_SBP_112.pth
# for Duke
# python train_pro.py --lr_start 0.0001 --save_step 2 --eval_step 2 --dropout_rate 0.6 --if_weight_softmax --weight_softmax_value 4.0 --zoom_out_pad_prob 0.3 --if_load_SBP --SBP_dir ./SBP/model_save_DUKE/pytorch_SBP_100.pth

# 6.29
# Full, No MLoss
# CUHK03, SGD/Adam
# FAIL # python train_pro.py --optim_name SGD --Lr_Start 0.035 --addSTNPart --STN_lr_start 0.00001 --STN_init_value 1.0 --STN_freeze_end_ep 90 --if_load_STN_SBP_model --SBP_model_dir ./SBP/model_save_CUHK03/pytorch_SBP_112.pth --if_affine --addOrientationPart --OP_lr_start 0.001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./OP/model_save-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBP-SBP-model_save_CUHK03-pytorch_SBP_112.pthLy2/OP-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBP-SBP-model_save_CUHK03-pytorch_SBP_112.pthLy2_256x128_56.pth --if_OP_channel_wise --addChannelReduce --final_dim 2048 --addAttentionMaskPart --save_step 4 --eval_step 2 --target_dataset_name CUHK03
# SUCCESS # python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addSTNPart --STN_lr_start 0.0000001 --STN_init_value 1.0 --STN_freeze_end_ep 130 --if_load_STN_SBP_model --SBP_model_dir ./SBP/model_save_CUHK03/pytorch_SBP_112.pth --if_affine --addOrientationPart --OP_lr_start 0.00001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./OP/model_save-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBP-SBP-model_save_CUHK03-pytorch_SBP_112.pthLy2/OP-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBP-SBP-model_save_CUHK03-pytorch_SBP_112.pthLy2_256x128_56.pth --if_OP_channel_wise --addChannelReduce --final_dim 2048 --addAttentionMaskPart --save_step 4 --eval_step 2 --target_dataset_name CUHK03

# 6.30
# Full, No Triplet Loss, Mloss
# CUHK03
# FAIL # python train_pro.py --optim_name SGD --Lr_Start 0.035 --triplet_loss_switch --addSTNPart --STN_lr_start 0.00001 --STN_init_value 1.0 --STN_freeze_end_ep 90 --if_load_STN_SBP_model --SBP_model_dir ./SBP/model_save_CUHK03/pytorch_SBP_112.pth --if_affine --addOrientationPart --OP_lr_start 0.001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./OP/model_save-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBP-SBP-model_save_CUHK03-pytorch_SBP_112.pthLy2/OP-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBP-SBP-model_save_CUHK03-pytorch_SBP_112.pthLy2_256x128_56.pth --if_OP_channel_wise --addChannelReduce --final_dim 2048 --addAttentionMaskPart --save_step 4 --eval_step 2 --target_dataset_name CUHK03

# MSMT17
# SBP
# python train.py --target_dataset_name MSMT17
# OP pretrain
# python train_pro.py --lr_start 0.0001 --save_step 2 --eval_step 2 --dropout_rate 0.6 --if_weight_softmax --weight_softmax_value 4.0 --zoom_out_pad_prob 0.3 --if_load_SBP --SBP_dir ./SBP/model_save_
# Full, No MLoss
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addSTNPart --STN_lr_start 0.00001 --STN_init_value 1.0 --STN_freeze_end_ep 90 --if_load_STN_SBP_model --SBP_model_dir ./SBP --if_affine --addOrientationPart --OP_lr_start 0.001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./OP --if_OP_channel_wise --addChannelReduce --final_dim 2048 --addAttentionMaskPart --save_step 4 --eval_step 2 --target_dataset_name MSMT17



# ########################################################################
# CUHK03 full test

# ./CUHK_DUKE_pretrain/SBP/CUHK03_pytorch_SBP_112.pth
# ./CUHK_DUKE_pretrain/OP/OP_CUHK03_SBP_112_256x128_56.pth

# Full Model, No MLoss
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addSTNPart --STN_lr_start 0.0000001 --STN_init_value 1.0 --STN_freeze_end_ep 130 --if_load_STN_SBP_model --SBP_model_dir ./CUHK_DUKE_pretrain/SBP/CUHK03_pytorch_SBP_112.pth --if_affine --addOrientationPart --OP_lr_start 0.00001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./CUHK_DUKE_pretrain/OP/OP_CUHK03_SBP_112_256x128_56.pth --if_OP_channel_wise --addChannelReduce --final_dim 2048 --addAttentionMaskPart --save_step 4 --eval_step 2 --target_dataset_name CUHK03
# #####################################  START PARALLEL  ###################################


# Full Model + MLoss
# echo Full Model + MLoss >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addSTNPart --STN_lr_start 0.0000001 --STN_init_value 1.0 --STN_freeze_end_ep 90 --if_load_STN_SBP_model --SBP_model_dir ./CUHK_DUKE_pretrain/SBP/CUHK03_pytorch_SBP_112.pth --if_affine --addOrientationPart --OP_lr_start 0.00001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./CUHK_DUKE_pretrain/OP/OP_CUHK03_SBP_112_256x128_56.pth --if_OP_channel_wise --addChannelReduce --final_dim 2048 --addAttentionMaskPart --mask_loss_switch --binary_threshold 0.8 --area_constrain_proportion 0.2 --mask_loss_weight 0.001 --save_step 4 --eval_step 2 --target_dataset_name CUHK03


# Attention num_discuss
# CUHK03
# echo Attention num_discuss >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addAttentionMaskPart --mask_num 1 --save_step 4 --eval_step 2 --target_dataset_name CUHK03
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addAttentionMaskPart --mask_num 2 --save_step 4 --eval_step 2 --target_dataset_name CUHK03
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addAttentionMaskPart --mask_num 4 --save_step 4 --eval_step 2 --target_dataset_name CUHK03
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addAttentionMaskPart --mask_num 8 --save_step 4 --eval_step 2 --target_dataset_name CUHK03
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addAttentionMaskPart --mask_num 16 --save_step 4 --eval_step 2 --target_dataset_name CUHK03


# Orientation Sensitive Merge Strategy
# CUHK03
# echo Orientation Sensitive Merge Strategy >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addOrientationPart --OP_lr_start 0.00001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./CUHK_DUKE_pretrain/OP/OP_CUHK03_SBP_112_256x128_56.pth --if_OP_channel_wise --save_step 4 --eval_step 2 --target_dataset_name CUHK03
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addOrientationPart --OP_lr_start 0.00001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./CUHK_DUKE_pretrain/OP/OP_CUHK03_SBP_112_256x128_56.pth --if_OP_channel_wise --addChannelReduce --final_dim 2048 --save_step 4 --eval_step 2 --target_dataset_name CUHK03
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addOrientationPart --OP_lr_start 0.00001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./CUHK_DUKE_pretrain/OP/OP_CUHK03_SBP_112_256x128_56.pth --save_step 4 --eval_step 2 --target_dataset_name CUHK03
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addOrientationPart --OP_lr_start 0.00001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./CUHK_DUKE_pretrain/OP/OP_CUHK03_SBP_112_256x128_56.pth --addChannelReduce --final_dim 1024 --save_step 4 --eval_step 2 --target_dataset_name CUHK03


# Affine Transformation Init Value
# CUHK03
# echo Affine Transformation Init Value >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addSTNPart --STN_lr_start 0.0000001 --STN_init_value 1.0 --STN_freeze_end_ep 60 --if_load_STN_SBP_model --SBP_model_dir ./CUHK_DUKE_pretrain/SBP/CUHK03_pytorch_SBP_112.pth --if_affine --save_step 4 --eval_step 2 --target_dataset_name CUHK03
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addSTNPart --STN_lr_start 0.0000001 --STN_init_value 0.9 --STN_freeze_end_ep 60 --if_load_STN_SBP_model --SBP_model_dir ./CUHK_DUKE_pretrain/SBP/CUHK03_pytorch_SBP_112.pth --if_affine --save_step 4 --eval_step 2 --target_dataset_name CUHK03
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addSTNPart --STN_lr_start 0.0000001 --STN_init_value 0.8 --STN_freeze_end_ep 60 --if_load_STN_SBP_model --SBP_model_dir ./CUHK_DUKE_pretrain/SBP/CUHK03_pytorch_SBP_112.pth --if_affine --save_step 4 --eval_step 2 --target_dataset_name CUHK03


# Affine Transformation Affine vs. Perspective
# Perspective + b_1.0
# CUHK03
# echo Affine Transformation Affine vs. Perspective >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addSTNPart --STN_lr_start 0.0000001 --STN_init_value 1.0 --STN_freeze_end_ep 60 --if_load_STN_SBP_model --SBP_model_dir ./CUHK_DUKE_pretrain/SBP/CUHK03_pytorch_SBP_112.pth --save_step 4 --eval_step 2 --target_dataset_name CUHK03

# MIE_SBP
# echo MIE_SBP CUHK03 >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --save_step 4 --eval_step 2 --target_dataset_name CUHK03

# ########################################################################






# ########################################################################
# Market full test
# echo Market Start >> ~/Desktop/Roll_Run_Record.txt


# Attention num_discuss
# Market
# echo Attention num_discuss >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addAttentionMaskPart --mask_num 1 --save_step 4 --eval_step 2 --target_dataset_name MARKET
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addAttentionMaskPart --mask_num 2 --save_step 4 --eval_step 2 --target_dataset_name MARKET
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addAttentionMaskPart --mask_num 4 --save_step 4 --eval_step 2 --target_dataset_name MARKET
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addAttentionMaskPart --mask_num 8 --save_step 4 --eval_step 2 --target_dataset_name MARKET
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addAttentionMaskPart --mask_num 16 --save_step 4 --eval_step 2 --target_dataset_name MARKET



# Affine Transformation Init Value
# Market
# echo Affine Transformation Init Value >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addSTNPart --STN_lr_start 0.00001 --STN_init_value 0.9 --STN_freeze_end_ep 60 --if_load_STN_SBP_model --SBP_model_dir ./MIE/MIE_SBP/Market_SBP/model_save-CS-MARKET/pytorch_MIE_104.pth --if_affine --save_step 4 --eval_step 2
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addSTNPart --STN_lr_start 0.00001 --STN_init_value 0.8 --STN_freeze_end_ep 60 --if_load_STN_SBP_model --SBP_model_dir ./MIE/MIE_SBP/Market_SBP/model_save-CS-MARKET/pytorch_MIE_104.pth --if_affine --save_step 4 --eval_step 2



# Affine Transformation Affine vs. Perspective
# Perspective + b_1.0
# Market
# echo Affine Transformation Affine vs. Perspective >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addSTNPart --STN_lr_start 0.00001 --STN_init_value 1.0 --STN_freeze_end_ep 60 --if_load_STN_SBP_model --SBP_model_dir ./MIE/MIE_SBP/Market_SBP/model_save-CS-MARKET/pytorch_MIE_104.pth --save_step 4 --eval_step 2

# echo Market End >> ~/Desktop/Roll_Run_Record.txt
# ########################################################################







# ########################################################################
# DUKE full test
# echo Duke Start >> ~/Desktop/Roll_Run_Record.txt

# ./CUHK_DUKE_pretrain/SBP/DUKE_pytorch_SBP_100.pth
# ./CUHK_DUKE_pretrain/OP/OP-DUKE_SBP_100_256x128_52.pth

# Full Model, No MLoss
# echo Full Model, No MLoss >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addSTNPart --STN_lr_start 0.00001 --STN_init_value 1.0 --STN_freeze_end_ep 90 --if_load_STN_SBP_model --SBP_model_dir ./CUHK_DUKE_pretrain/SBP/DUKE_pytorch_SBP_100.pth --if_affine --addOrientationPart --OP_lr_start 0.001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./CUHK_DUKE_pretrain/OP/OP-DUKE_SBP_100_256x128_52.pth --if_OP_channel_wise --addChannelReduce --final_dim 2048 --addAttentionMaskPart --save_step 4 --eval_step 2 --target_dataset_name DUKE
# #####################################  START PARALLEL  ###################################


# Full Model + MLoss
# echo Full Model + MLoss >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addSTNPart --STN_lr_start 0.00001 --STN_init_value 1.0 --STN_freeze_end_ep 90 --if_load_STN_SBP_model --SBP_model_dir ./CUHK_DUKE_pretrain/SBP/DUKE_pytorch_SBP_100.pth --if_affine --addOrientationPart --OP_lr_start 0.001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./CUHK_DUKE_pretrain/OP/OP-DUKE_SBP_100_256x128_52.pth --if_OP_channel_wise --addChannelReduce --final_dim 2048 --addAttentionMaskPart --mask_loss_switch --binary_threshold 0.8 --area_constrain_proportion 0.2 --mask_loss_weight 0.001 --save_step 4 --eval_step 2 --target_dataset_name DUKE


# Attention num_discuss
# Duke
# echo Attention num_discuss >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addAttentionMaskPart --mask_num 1 --save_step 4 --eval_step 2 --target_dataset_name DUKE
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addAttentionMaskPart --mask_num 2 --save_step 4 --eval_step 2 --target_dataset_name DUKE
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addAttentionMaskPart --mask_num 4 --save_step 4 --eval_step 2 --target_dataset_name DUKE
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addAttentionMaskPart --mask_num 8 --save_step 4 --eval_step 2 --target_dataset_name DUKE
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addAttentionMaskPart --mask_num 16 --save_step 4 --eval_step 2 --target_dataset_name DUKE


# echo Duke End >> ~/Desktop/Roll_Run_Record.txt
# ########################################################################



# 7.6
# echo "7.6 18:45 run" >> ~/Desktop/Roll_Run_Record.txt
# echo "Adam Market EM1024, Full Model, No MLoss" >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addSTNPart --STN_lr_start 0.0000001 --STN_init_value 1.0 --STN_freeze_end_ep 130 --if_load_STN_SBP_model --SBP_model_dir ./MIE/MIE_SBP/Market_SBP/model_save-CS-MARKET/pytorch_MIE_104.pth --if_affine --addOrientationPart --OP_lr_start 0.00001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./OP/model_save-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2/OP-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2_256x128_59.pth --addChannelReduce --final_dim 1024 --addAttentionMaskPart --save_step 4 --eval_step 4 --target_dataset_name MARKET

# ########################################################################
# Addition test in Duke
# echo Addition test in Duke >> ~/Desktop/Roll_Run_Record.txt

# Orientation Sensitive Merge Strategy
# Duke
# echo Orientation Sensitive Merge Strategy in Duke >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addOrientationPart --OP_lr_start 0.001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./CUHK_DUKE_pretrain/OP/OP-DUKE_SBP_100_256x128_52.pth --if_OP_channel_wise --save_step 4 --eval_step 4 --target_dataset_name DUKE
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addOrientationPart --OP_lr_start 0.001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./CUHK_DUKE_pretrain/OP/OP-DUKE_SBP_100_256x128_52.pth --if_OP_channel_wise --addChannelReduce --final_dim 2048 --save_step 4 --eval_step 4 --target_dataset_name DUKE
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addOrientationPart --OP_lr_start 0.001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./CUHK_DUKE_pretrain/OP/OP-DUKE_SBP_100_256x128_52.pth --save_step 4 --eval_step 4 --target_dataset_name DUKE
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addOrientationPart --OP_lr_start 0.001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./CUHK_DUKE_pretrain/OP/OP-DUKE_SBP_100_256x128_52.pth --addChannelReduce --final_dim 1024 --save_step 4 --eval_step 4 --target_dataset_name DUKE


# Affine Transformation Init Value
# Duke
# echo Affine Transformation Init Value >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addSTNPart --STN_lr_start 0.00001 --STN_init_value 1.0 --STN_freeze_end_ep 60 --if_load_STN_SBP_model --SBP_model_dir ./CUHK_DUKE_pretrain/SBP/DUKE_pytorch_SBP_100.pth --if_affine --save_step 4 --eval_step 4 --target_dataset_name DUKE
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addSTNPart --STN_lr_start 0.00001 --STN_init_value 0.9 --STN_freeze_end_ep 60 --if_load_STN_SBP_model --SBP_model_dir ./CUHK_DUKE_pretrain/SBP/DUKE_pytorch_SBP_100.pth --if_affine --save_step 4 --eval_step 4 --target_dataset_name DUKE
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addSTNPart --STN_lr_start 0.00001 --STN_init_value 0.8 --STN_freeze_end_ep 60 --if_load_STN_SBP_model --SBP_model_dir ./CUHK_DUKE_pretrain/SBP/DUKE_pytorch_SBP_100.pth --if_affine --save_step 4 --eval_step 4 --target_dataset_name DUKE


# Affine Transformation Affine vs. Perspective
# Perspective + b_1.0
# Duke
# echo Affine Transformation Affine vs. Perspective >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name SGD --Lr_Start 0.035 --addSTNPart --STN_lr_start 0.00001 --STN_init_value 1.0 --STN_freeze_end_ep 60 --if_load_STN_SBP_model --SBP_model_dir ./CUHK_DUKE_pretrain/SBP/DUKE_pytorch_SBP_100.pth --save_step 4 --eval_step 4 --target_dataset_name DUKE

# echo End Addition test in Duke >> ~/Desktop/Roll_Run_Record.txt
# ##################################################################




# ##################################################################
# Adam Run Market + Duke
# Full Model, No MLoss
# !!!! FAIL No Need Train: Max rank-1:9376 mAP:8450 !!!!

# Market
# echo Adam Market Full Model, No MLoss >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addSTNPart --STN_lr_start 0.0000001 --STN_init_value 1.0 --STN_freeze_end_ep 130 --if_load_STN_SBP_model --SBP_model_dir ./MIE/MIE_SBP/Market_SBP/model_save-CS-MARKET/pytorch_MIE_104.pth --if_affine --addOrientationPart --OP_lr_start 0.00001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./OP/model_save-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2/OP-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2_256x128_59.pth --if_OP_channel_wise --addChannelReduce --final_dim 2048 --addAttentionMaskPart --save_step 4 --eval_step 4 --target_dataset_name MARKET

# Market
echo "7.9 21:20 run" >> ~/Desktop/Roll_Run_Record.txt
echo Adam Market Full Model, No MLoss, EM+CR1024 >> ~/Desktop/Roll_Run_Record.txt
python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addSTNPart --STN_lr_start 0.0000001 --STN_init_value 1.0 --STN_freeze_end_ep 130 --if_load_STN_SBP_model --SBP_model_dir ./MIE/MIE_SBP/Market_SBP/model_save-CS-MARKET/pytorch_MIE_104.pth --if_affine --addOrientationPart --OP_lr_start 0.00001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./OP/model_save-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2/OP-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2_256x128_59.pth --addChannelReduce --final_dim 1024 --addAttentionMaskPart --save_step 4 --eval_step 4 --target_dataset_name MARKET

# Duke
# echo Adam Duke Full Model, No MLoss >> ~/Desktop/Roll_Run_Record.txt
# python train_pro.py --optim_name Adam --Lr_Start 0.00035 --addSTNPart --STN_lr_start 0.0000001 --STN_init_value 1.0 --STN_freeze_end_ep 130 --if_load_STN_SBP_model --SBP_model_dir ./CUHK_DUKE_pretrain/SBP/DUKE_pytorch_SBP_100.pth --if_affine --addOrientationPart --OP_lr_start 0.00001 --OP_freeze_end_ep 0 --if_load_OP_model --OP_model_dir ./CUHK_DUKE_pretrain/OP/OP-DUKE_SBP_100_256x128_52.pth --if_OP_channel_wise --addChannelReduce --final_dim 2048 --addAttentionMaskPart --save_step 4 --eval_step 4 --target_dataset_name DUKE

# ##################################################################
