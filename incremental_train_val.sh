export PORT=29501
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NCCL_P2P_DISABLE=1
# unset LD_LIBRARY_PATH

bash dist_train.sh configs/ov_coco/cat_seg_2d_4tasks/catseg_mask_task4.py 4 --work-dir runs/cat-seg/ablation_ewc/test_train_task4_10_2d_moe_sigmoid
#     --resume-from runs/cat-seg/test_train_task1_10_2d_moe_sigmoid/epoch_18.pth
# bash dist_train.sh configs/ov_coco/cat_seg_2d_4tasks/catseg_mask_task2.py 8 --work-dir runs/cat-seg/ablation_ewc/test_train_task2_10_2d_moe_sigmoid
      # --resume-from runs/cat-seg/ablation_cls_aggre/test_train_task1_10_2d_moe_sigmoid/epoch_15.pth
      # --resume-from runs/cat-seg/test_train_task1_10_2d_moe_sigmoid/epoch_5.pth
# bash dist_test.sh configs/ov_coco/cat_seg_2d_pseudolabel/catseg_mask_task2.py \
#     runs/cat-seg/test_train_task2_10_2d_moe_sigmoid/epoch_3.pth \
#     2 \
#     --work-dir runs/test/eval \
#     --eval bbox \
