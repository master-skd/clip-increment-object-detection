# export PORT=29501
export CUDA_VISIBLE_DEVICES=4,5,6,7
# export NCCL_P2P_DISABLE=1
# unset LD_LIBRARY_PATH

# bash dist_train.sh configs/ov_coco/cat_seg_2d_pseudolabel/catseg_mask.py 4 --work-dir runs/cat-seg/test_train_task1_10_2d_moe_sigmoid
#     --resume-from runs/cat-seg/test_train_task1_10_2d_moe_sigmoid/epoch_15.pth
bash dist_train.sh configs/ov_coco/cat_seg_2d_pseudolabel/catseg_mask_task2.py 4 --work-dir runs/cat-seg/test_train_task2_10_2d_moe_sigmoid
# bash dist_test.sh configs/ov_coco/cat_seg_2d_pseudolabel/catseg_mask_task2.py \
#     runs/cat-seg/test_train_task2_10_2d_moe_sigmoid/epoch_3.pth \
#     2 \
#     --work-dir runs/test/eval \
#     --eval bbox \
