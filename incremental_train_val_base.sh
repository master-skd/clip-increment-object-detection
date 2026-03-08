export PORT=29505
export CUDA_VISIBLE_DEVICES=0,1,2,3

bash dist_train.sh configs/ov_coco/baseline_dist/task2.py 4 --work-dir runs/base_dist/test_train_task2_10_2d_decouple_distill_all_pseudo_label_sigmoid
