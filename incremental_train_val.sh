# source activate fvit
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export PORT=29500
export CUDA_VISIBLE_DEVICES=4,5,6,7
# export NCCL_P2P_DISABLE=1
unset LD_LIBRARY_PATH

# bash dist_train.sh configs/ov_coco/incremental/vit-l/fvit_incremental_task1.py 1 --work-dir runs/fineclipvit-l

# bash dist_train.sh configs/ov_coco/incremental/fg-clip-large-896/fvit_incremental_task1.py 1 --work-dir runs/incremental/fg-clip-large-896/train_task1
# bash dist_test.sh configs/ov_coco/incremental/fg-clip-large-896/fvit_incremental_task1.py runs/incremental/fg-clip-large-896/train_task1/latest.pth 1 --work-dir runs/incremental/fg-clip-large-896/eval_task1 --eval bbox
bash dist_train.sh configs/ov_coco/cat_seg_2d_pseudolabel/catseg_mask_task2.py 4 --work-dir runs/cat-seg/test_train_task2_10_2d_decouple_distill_pseudo_label
# bash dist_test.sh configs/ov_coco/cat_seg_pseudolabel/pretrained_catseg_mask.py runs/test/epoch_1.pth 1 --work-dir runs/test/eval --eval bbox
# bash dist_test.sh configs/ov_coco/incremental/fg-clip-large-896/fvit_incremental_task2.py runs/incremental/fg-clip-large-896/train_task2/latest.pth 1 --work-dir runs/incremental/fg-clip-large-896/eval_task2 --eval bbox
# bash dist_train.sh configs/ov_coco/incremental/fg-clip-large-896/fvit_incremental_task3.py 1 --work-dir runs/incremental/fg-clip-large-896/train_task3
# bash dist_test.sh configs/ov_coco/incremental/fg-clip-large-896/fvit_incremental_task3.py runs/incremental/fg-clip-large-896/train_task3/latest.pth 1 --work-dir runs/incremental/fg-clip-large-896/eval_task3 --eval bbox
# bash dist_train.sh configs/ov_coco/incremental/fg-clip-large-896/fvit_incremental_task4.py 1 --work-dir runs/incremental/fg-clip-large-896/train_task4
# bash dist_test.sh configs/ov_coco/incremental/fg-clip-large-896/fvit_incremental_task4.py runs/incremental/fg-clip-large-896/train_task4/latest.pth 1 --work-dir runs/incremental/fg-clip-large-896/eval_task4 --eval bbox
