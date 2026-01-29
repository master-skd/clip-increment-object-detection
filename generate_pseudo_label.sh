CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29505 \
    --use_env \
    generate_pseudo_label.py \
    --launcher pytorch \
    --score-thr 0.5 \
    --out train_task2_with_pseudo_labels_2d.json