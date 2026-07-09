#!/usr/bin/env python3
"""
Pick-a-Pic 数据集使用示例
这个脚本展示了如何读取和使用 Pick-a-Pic 数据集
"""

import pandas as pd
import io
from PIL import Image
import os

def load_pickapic_data(parquet_file):
    """加载 Pick-a-Pic parquet 文件"""
    df = pd.read_parquet(parquet_file)
    return df

def get_image_from_bytes(jpg_bytes):
    """从字节数据转换为 PIL Image 对象"""
    return Image.open(io.BytesIO(jpg_bytes))

def save_comparison_pair(df, index, output_dir='examples'):
    """保存一对比较图像到磁盘"""
    os.makedirs(output_dir, exist_ok=True)

    row = df.iloc[index]

    # 保存图像
    img0 = get_image_from_bytes(row['jpg_0'])
    img1 = get_image_from_bytes(row['jpg_1'])

    img0.save(f"{output_dir}/sample_{index}_image0.jpg")
    img1.save(f"{output_dir}/sample_{index}_image1.jpg")

    # 保存元数据
    with open(f"{output_dir}/sample_{index}_metadata.txt", 'w') as f:
        f.write(f"Caption: {row['caption']}\n")
        f.write(f"\nImage 0:\n")
        f.write(f"  - Model: {row['model_0']}\n")
        f.write(f"  - Label: {row['label_0']}\n")
        f.write(f"  - UID: {row['image_0_uid']}\n")
        f.write(f"\nImage 1:\n")
        f.write(f"  - Model: {row['model_1']}\n")
        f.write(f"  - Label: {row['label_1']}\n")
        f.write(f"  - UID: {row['image_1_uid']}\n")
        f.write(f"\nBest Image: {'image_0' if row['best_image_uid'] == row['image_0_uid'] else 'image_1'}\n")
        f.write(f"Has Label: {row['has_label']}\n")
        f.write(f"Are Different: {row['are_different']}\n")

    print(f"Saved comparison pair {index} to {output_dir}/")

def analyze_dataset(df):
    """分析数据集统计信息"""
    print("=== Dataset Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"Labeled samples: {df['has_label'].sum()}")
    print(f"Average label_0: {df['label_0'].mean():.3f}")
    print(f"Average label_1: {df['label_1'].mean():.3f}")
    print(f"\nModels used:")
    models = pd.concat([df['model_0'], df['model_1']]).value_counts()
    for model, count in models.items():
        print(f"  {model}: {count}")

if __name__ == "__main__":
    # 加载数据
    print("Loading Pick-a-Pic dataset...")
    df = load_pickapic_data('test-00000-of-00010-a4b2a15ad966dd4c.parquet')

    # 分析数据集
    analyze_dataset(df)

    # 保存几个示例
    print("\n=== Saving example pairs ===")
    for i in range(3):
        save_comparison_pair(df, i)

    print("\n=== Example data access ===")
    # 展示如何访问数据
    row = df.iloc[0]
    print(f"Caption: {row['caption']}")
    print(f"Image 0 model: {row['model_0']}, Label: {row['label_0']}")
    print(f"Image 1 model: {row['model_1']}, Label: {row['label_1']}")

    # 展示如何加载图像
    img0 = get_image_from_bytes(row['jpg_0'])
    img1 = get_image_from_bytes(row['jpg_1'])
    print(f"\nImage 0 size: {img0.size}")
    print(f"Image 1 size: {img1.size}")
