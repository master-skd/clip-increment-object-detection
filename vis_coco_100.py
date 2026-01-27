import os
import random
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO

def color_from_id(idx: int):
    # 固定且“看得清”的颜色（按类别/实例 id 稳定生成）
    rng = np.random.RandomState(idx * 9973 % 2**32)
    return tuple(int(x) for x in rng.randint(30, 225, size=3))

def draw_one_image(coco: COCO, img_root: str, out_dir: str, img_id: int, alpha: float = 0.35):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_root, img_info["file_name"])
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Cannot read image: {img_path}")
        return

    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
    anns = coco.loadAnns(ann_ids)

    # overlay 用来画 mask（半透明）
    overlay = img.copy()

    # 类别 id -> name
    cats = coco.loadCats(coco.getCatIds())
    catid2name = {c["id"]: c["name"] for c in cats}

    for ann in anns:
        cat_id = ann.get("category_id", -1)
        name = catid2name.get(cat_id, str(cat_id))
        inst_id = ann.get("id", 0)
        color = color_from_id(cat_id if cat_id != -1 else inst_id)  # BGR

        # 1) segmentation mask（如果有）
        if "segmentation" in ann and ann["segmentation"]:
            try:
                mask = coco.annToMask(ann)  # (H, W) 0/1
                if mask is not None and mask.any():
                    overlay[mask.astype(bool)] = color
            except Exception:
                pass

        # 2) bbox（如果有）
        if "bbox" in ann and ann["bbox"] is not None:
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            label = f"{name}"
            cv2.putText(
                img, label, (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
            )

        # 3) keypoints（如果有）
        if "keypoints" in ann and ann["keypoints"]:
            kps = ann["keypoints"]
            # COCO keypoints: [x1,y1,v1, x2,y2,v2, ...]
            for i in range(0, len(kps), 3):
                xk, yk, v = kps[i], kps[i+1], kps[i+2]
                if v > 0:
                    cv2.circle(img, (int(xk), int(yk)), 3, color, -1)

    # mask 半透明叠加
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(img_info["file_name"]))
    cv2.imwrite(out_path, img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True, help="path to COCO annotations json")
    parser.add_argument("--img_root", required=True, help="root dir of images (where file_name is relative to)")
    parser.add_argument("--out", required=True, help="output dir for visualization images")
    parser.add_argument("--num", type=int, default=100, help="number of images to visualize")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    coco = COCO(args.ann)
    img_ids = coco.getImgIds()

    random.seed(args.seed)
    pick = random.sample(img_ids, k=min(args.num, len(img_ids)))

    for img_id in tqdm(pick, desc="Visualizing"):
        draw_one_image(coco, args.img_root, args.out, img_id)

    print(f"Done. Saved to: {args.out}")

if __name__ == "__main__":
    main()
