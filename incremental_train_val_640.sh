# source activate fvit
export PORT=29501

bash dist_train.sh configs/ov_coco/incremental/vit-b/fvit_incremental_task1.py 1 --work-dir runs/test/fineclip
# bash dist_test.sh configs/ov_coco/incremental/fg-clip-base-640/fvit_incremental_task1.py runs/incremental/fg-clip-base-640/train_task1/latest.pth 1 --work-dir runs/incremental/fg-clip-base-640/eval_task1 --eval bbox
# bash dist_train.sh configs/ov_coco/incremental/fg-clip-base-640/fvit_incremental_task2.py 1 --work-dir runs/incremental/fg-clip-base-640/train_task2
# bash dist_test.sh configs/ov_coco/incremental/fg-clip-base-640/fvit_incremental_task2.py runs/incremental/fg-clip-base-640/train_task2/latest.pth 1 --work-dir runs/incremental/fg-clip-base-640/eval_task2 --eval bbox
# bash dist_train.sh configs/ov_coco/incremental/fg-clip-base-640/fvit_incremental_task3.py 1 --work-dir runs/incremental/fg-clip-base-640/train_task3
# bash dist_test.sh configs/ov_coco/incremental/fg-clip-base-640/fvit_incremental_task3.py runs/incremental/fg-clip-base-640/train_task3/latest.pth 1 --work-dir runs/incremental/fg-clip-base-640/eval_task3 --eval bbox
# bash dist_train.sh configs/ov_coco/incremental/fg-clip-base-640/fvit_incremental_task4.py 1 --work-dir runs/incremental/fg-clip-base-640/train_task4
# bash dist_test.sh configs/ov_coco/incremental/fg-clip-base-640/fvit_incremental_task4.py runs/incremental/fg-clip-base-640/train_task4/latest.pth 1 --work-dir runs/incremental/fg-clip-base-640/eval_task4 --eval bbox
