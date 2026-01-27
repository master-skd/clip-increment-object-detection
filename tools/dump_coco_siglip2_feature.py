import argparse
import json
import torch
import open_clip
from transformers import AutoTokenizer
from clip_utils import build_text_embedding_openclip

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default='/hy-tmp/liuzihan/datasets/MSCOCO/2017/annotations/instances_val2017.json')
    parser.add_argument('--out_path', default='datasets/embeddings/SigLIP2_vitl16_256.pt')
    parser.add_argument('--model_name', default='ViT-L-16-SigLIP2-256')
    parser.add_argument('--model_dir', default="/hy-tmp/liuzihan/CLIPSelf/F-ViT/checkpoints/siglip2")
    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    cat_names = [x['name'] for x in \
        sorted(data['categories'], key=lambda x: x['id'])]
    cat_names = cat_names + ['background']
    ori_cat_names = cat_names
    print('cat_names', cat_names)
    
    import os
    
    print(f'Loading SigLIP2 model: {args.model_name}')
    
    # 加载模型：指定模型架构和本地权重路径
    model_path = os.path.join(args.model_dir, 'open_clip_model.safetensors')
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name=args.model_name,
        pretrained=model_path
    )
    model.eval()
    
    # 从本地目录加载 tokenizer
    print(f'从本地加载 tokenizer: {args.model_dir}')
    tokenizer_hf = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    # 包装 tokenizer 使其返回 PyTorch tensor
    def tokenizer(texts):
        return tokenizer_hf(texts, return_tensors="pt", padding=True, truncation=True)
    print("成功加载本地模型和tokenizer")
    
    # 生成文本嵌入
    print('Building text embeddings...')
    text_embeddings = build_text_embedding_openclip(cat_names, model, tokenizer)
    text_embeddings = text_embeddings.cpu()
    text_embeddings = text_embeddings.to(torch.float32)
    print('text_embeddings.shape', text_embeddings.shape)
    
    # 保存为字典格式
    class_embed = {k: v for k, v in zip(ori_cat_names, text_embeddings)}
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True) # 确保目录存在
    torch.save(class_embed, args.out_path)
    print(f'Saved text embeddings to {args.out_path}')

