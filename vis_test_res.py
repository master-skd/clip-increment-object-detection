import torch
import mmcv
import numpy as np
import cv2
import os
from models.cat_seg_detector import CatSegDetector
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO

# ================= 配置区域 (请修改这里) =================
# 1. 配置文件路径
CONFIG_FILE = 'configs/ov_coco/cat_seg_2d_pseudolabel/catseg_mask_task2.py'
# 2. 权重文件路径
CHECKPOINT_FILE = 'runs/cat-seg/test_train_task2_10_2d_decouple_distill_all_pseudo_label/epoch_1.pth'
# 3. 验证集的标注文件 (包含 Task 1 旧类的 GT)
ANN_FILE = '/mnt/data14/yyg/datasets/Incremental/test_task_12.json'
# 4. 图片根目录
IMG_ROOT = '/mnt/data14/yyg/datasets/MSCOCO/2017/val2017'
# 5. 输出目录
SAVE_DIR = 'debug_vis_results'

# 6. 关键参数
OLD_CLASSES_NUM = 19  # 只看前 19 个类
SCORE_THR = 0.05      # 【关键】设置极低阈值，查看是否有微弱响应
MAX_IMAGES = 50       # 只随机看 50 张图，避免跑太久
# =======================================================

def draw_bbox(img, bbox, label_name, color, score=None):
    """
    绘制单个边界框
    Color: (B, G, R) -> Green=(0, 255, 0), Red=(0, 0, 255)
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # 画框 (线宽 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # 准备标签文本
    if score is not None:
        text = f"{label_name}|{score:.2f}"
    else:
        text = f"{label_name}"
        
    # 画文字背景 (避免看不清)
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
    
    # 画文字 (白色)
    cv2.putText(img, text, (x1, y1 - 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    # 1. 初始化模型
    print(f"正在加载模型: {CHECKPOINT_FILE} ...")
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device='cuda:0')
    
    # 获取类别列表 (假设模型的前 OLD_CLASSES_NUM 个类就是 Task 1 的类)
    # 注意：MMDet 的 classes 是一个 list，索引即 label
    model_classes = model.CLASSES
    
    # 2. 加载 COCO 标注
    print(f"正在加载标注文件: {ANN_FILE} ...")
    coco = COCO(ANN_FILE)
    
    # 建立 COCO category_id 到 class_name 的映射
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}
    # 建立 class_name 到 COCO category_id 的映射
    name_to_cat_id = {v: k for k, v in cat_id_to_name.items()}

    # 3. 准备输出目录
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 4. 筛选包含旧类物体的图片
    # 我们只关心包含前 19 类物体的图片，不要看全是背景的图
    target_names = model_classes[:OLD_CLASSES_NUM]
    target_cat_ids = [name_to_cat_id[n] for n in target_names if n in name_to_cat_id]
    
    img_ids_with_old_class = []
    for cat_id in target_cat_ids:
        img_ids_with_old_class.extend(coco.getImgIds(catIds=[cat_id]))
    
    # 去重并随机采样
    img_ids_with_old_class = list(set(img_ids_with_old_class))
    np.random.shuffle(img_ids_with_old_class)
    sample_ids = img_ids_with_old_class[:MAX_IMAGES]

    print(f"开始处理 {len(sample_ids)} 张包含旧类的图片...")

    for i, img_id in enumerate(sample_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(IMG_ROOT, img_info['file_name'])
        
        # 读取图片
        img = mmcv.imread(img_path)
        vis_img = img.copy()
        
        # -------------------------------------------------------------
        # A. 绘制 Ground Truth (红色) - 只画前 19 类
        # -------------------------------------------------------------
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        has_gt_vis = False
        for ann in anns:
            cat_id = ann['category_id']
            name = cat_id_to_name[cat_id]
            
            # 过滤：只画旧类
            if name in target_names:
                bbox = ann['bbox'] # x, y, w, h
                # 转换 xywh -> xyxy
                x1, y1, w, h = bbox
                rect = [x1, y1, x1 + w, y1 + h]
                
                # 绘制红色框 (0, 0, 255)
                draw_bbox(vis_img, rect, f"GT:{name}", (0, 0, 255))
                has_gt_vis = True

        # -------------------------------------------------------------
        # B. 绘制 Prediction (绿色) - 只画前 19 类
        # -------------------------------------------------------------
        result = inference_detector(model, img_path)
        # result 是一个 list，长度为 num_classes，每个元素是 [N, 5] 的 array (x1, y1, x2, y2, score)
        
        has_pred_vis = False
        # 只遍历前 19 个类的预测结果
        for label_idx in range(OLD_CLASSES_NUM):
            if label_idx >= len(result): break
            
            dets = result[label_idx]
            class_name = model_classes[label_idx]
            
            # 筛选分数大于阈值的框
            valid_inds = np.where(dets[:, -1] > SCORE_THR)[0]
            
            for ind in valid_inds:
                bbox = dets[ind, :4]
                score = dets[ind, 4]
                
                # 绘制绿色框 (0, 255, 0)
                draw_bbox(vis_img, bbox, class_name, (0, 255, 0), score)
                has_pred_vis = True

        # -------------------------------------------------------------
        # C. 保存结果
        # -------------------------------------------------------------
        # 只有当图片里有 GT 或者有预测时才保存
        if has_gt_vis or has_pred_vis:
            save_path = os.path.join(SAVE_DIR, f"vis_{img_info['file_name']}")
            cv2.imwrite(save_path, vis_img)
            
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(sample_ids)} 张")

    print(f"\n可视化完成！结果保存在: {SAVE_DIR}")
    print("红色框 = 真实标注 (GT)")
    print("绿色框 = 模型预测 (Pred)")

if __name__ == '__main__':
    main()