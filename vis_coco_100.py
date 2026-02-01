import json
import cv2
import os
import random
from tqdm import tqdm

def visualize_tasks(json_path, img_prefix, output_dir='vis_results', num_samples=100):
    # 加载 JSON 标注
    print("Loading annotations...")
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 建立索引
    images = {img['id']: img for img in coco_data['images']}
    img_to_anns = {}
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        # 过滤只属于 Task1 和 Task2 的标注
        if 0 <= cat_id <= 40:
            img_id = ann['image_id']
            img_to_anns.setdefault(img_id, []).append(ann)

    # 随机挑选 100 张有标注的图片
    available_ids = list(img_to_anns.keys())
    selected_ids = random.sample(available_ids, min(num_samples, len(available_ids)))

    print(f"Processing {len(selected_ids)} images...")
    for img_id in tqdm(selected_ids):
        img_info = images[img_id]
        file_name = img_info['file_name']
        img_path = os.path.join(img_prefix, file_name)
        
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        for ann in img_to_anns[img_id]:
            cat_id = ann['category_id']
            x, y, w, h = [int(v) for v in ann['bbox']]
            
            # 定义颜色 (BGR)
            if 0 <= cat_id <= 39:
                color = (0, 0, 255)  # Task 1: 红色
                label = f"T1_{cat_id}"
            elif 40 <= cat_id <= 60:
                color = (0, 255, 0)  # Task 2: 绿色
                label = f"T2_{cat_id}"
            else:
                continue

            # 画框和标签
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 保存结果
        cv2.imwrite(os.path.join(output_dir, f"vis_{file_name}"), img)

    print(f"Done! Results saved in '{output_dir}'")

# --- 配置参数 ---
JSON_PATH = 'train_task3_with_pseudo_labels_2d.json'
IMG_PREFIX = '/mnt/data14/yyg/datasets/MSCOCO/2017/train2017'

visualize_tasks(JSON_PATH, IMG_PREFIX)