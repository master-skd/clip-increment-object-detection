CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29505 \
    --use_env \
    generate_pseudo_label.py \
    --launcher pytorch \
    --score-thr 0.5 \
    --out train_task4_with_pseudo_labels_2d.json