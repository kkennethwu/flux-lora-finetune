import random
from pathlib import Path

import jsonlines


def process_single_file(caption_file, data_dir):
    """
    處理單個文件，返回樣本數據或 None
    """
    base_name = caption_file.stem
    jpg_file = data_dir / f"{base_name}.jpg"
    mask_file = data_dir / f"{base_name}_mask.png"

    # 檢查文件是否存在
    if not (jpg_file.exists() and mask_file.exists()):
        return None

    # 讀取 caption 內容
    try:
        with open(caption_file, 'r', encoding='utf-8') as f:
            caption = f.read().strip()

        sample = {
            "id": base_name,
            "image": str(jpg_file.absolute()),
            "caption": caption,
            "mask": str(mask_file.absolute())
        }
        return sample

    except Exception as e:
        print(f"處理文件 {caption_file} 時出錯: {e}")
        return None

def create_single_jsonl_with_jsonlines(data_dir, output_file="dataset.jsonl", train_ratio=0.8, random_state=42):
    """
    使用 jsonlines 庫寫入單個 JSONL 文件，包含 split 標記
    """
    data_dir = Path(data_dir)

    # 設置隨機種子
    random.seed(random_state)

    # 查找所有 caption 文件
    caption_files = list(data_dir.glob("cn_*.caption"))

    # 打亂順序用於分割
    random.shuffle(caption_files)

    # 計算分割點
    total_files = len(caption_files)
    train_count = int(total_files * train_ratio)

    print(f"找到 {total_files} 個 caption 文件")

    valid_count = 0
    train_samples = 0
    test_samples = 0

    # 使用 jsonlines 寫入單個文件
    with jsonlines.open(output_file, mode='w') as writer:
        # 寫入訓練樣本
        for caption_file in caption_files[:train_count]:
            sample = process_single_file(caption_file, data_dir)
            if sample:
                sample["split"] = "train"
                writer.write(sample)
                train_samples += 1
                valid_count += 1

        # 寫入測試樣本
        for caption_file in caption_files[train_count:]:
            sample = process_single_file(caption_file, data_dir)
            if sample:
                sample["split"] = "test"
                writer.write(sample)
                test_samples += 1
                valid_count += 1

    print(f"Output jsonl file: {output_file}")
    print(f"Train: {train_samples} 樣本")
    print(f"Test: {test_samples} 樣本")
    print(f"Total: {valid_count} 樣本")

    return output_file

# 使用示例
if __name__ == "__main__":
    # 你的數據目錄路徑
    DATA_DIR = "/teamspace/studios/this_studio/flux-lora-finetune/data/PosterArt-Text-small"  # 請修改為你的實際路徑

    jsonl_file = create_single_jsonl_with_jsonlines(
        DATA_DIR,
        output_file="text_dataset_small.jsonl",
        train_ratio=0.9,
        random_state=42
    )
