source activate pytorch36
python train_pro.py --addOrientationPart --if_load_OP_model --OP_model_dir ./OP/model_save-lr_0.0001-WT2.0-DP0.6/OP-lr_0.0001-WT2.0-DP0.6_256x128_40.pth --save_step 8 --eval_step 4
python train_pro.py --addOrientationPart --if_load_OP_model --OP_model_dir ./OP/model_save-lr_0.0001-RE-WT2.0-DP0.6/OP-lr_0.0001-RE-WT2.0-DP0.6_256x128_30.pth --save_step 8 --eval_step 4
python train_pro.py --addOrientationPart --if_load_OP_model --OP_model_dir ./OP/model_save-lr_6e-05-WT2.0-DP0.6/OP-lr_6e-05-WT2.0-DP0.6_256x128_50.pth --save_step 8 --eval_step 4
python train_pro.py --addOrientationPart --if_load_OP_model --OP_model_dir ./OP/model_save-lr_6e-05-RE-WT2.0-DP0.6/OP-lr_6e-05-RE-WT2.0-DP0.6_256x128_44.pth --save_step 8 --eval_step 4