source activate pytorch36
python train_pro.py --if_rerank --save_step 4 --eval_step 4
python train_pro.py --if_Euclidean --save_step 4 --eval_step 4
python train_pro.py --save_step 4 --eval_step 4
python train_pro.py --save_step 4 --eval_step 4 --target_dataset_name DUKE
python train_pro.py --if_rerank --save_step 4 --eval_step 4 --target_dataset_name DUKE