# -*- coding: utf-8 -*-
"""
HTGATFraud 离线推理脚本
加载训练好的模型checkpoint，对全图进行推理，输出JSON格式的预测结果。

用法:
    python inference.py --dataset yelp --checkpoint /path/to/model.pth
"""

import sys
import json
import argparse
from pathlib import Path

import torch

# 将异构图欺诈检测模型目录加入 sys.path，以便导入模型和工具模块
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "异构图欺诈检测模型"
sys.path.insert(0, str(MODEL_DIR))

from config import get_data_dir, get_model_config_for_dataset
from src.utils.data_loader import load_fraud_detection_data
from src.models.ht_gat_fraud import HTGATFraud


def parse_args():
    parser = argparse.ArgumentParser(description="HTGATFraud 离线推理")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["yelp", "amazon"],
        help="数据集名称 (yelp 或 amazon)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型 checkpoint 文件路径 (.pth)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"[ERROR] checkpoint 文件不存在: {checkpoint_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 设备选择
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # ------------------------------------------------------------------
    # 加载数据
    # ------------------------------------------------------------------
    print(f"[INFO] 正在加载 {dataset} 数据集 ...")
    data_dir = get_data_dir(dataset)
    data, class_weights = load_fraud_detection_data(data_dir, dataset=dataset)
    data = data.to(device)
    print(f"[INFO] 数据加载完成 — 节点数: {data.num_nodes}")

    # ------------------------------------------------------------------
    # 构建模型并加载权重
    # ------------------------------------------------------------------
    config = get_model_config_for_dataset(dataset)
    model = HTGATFraud(config)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("[INFO] 模型加载完成，开始推理 ...")

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    with torch.no_grad():
        logits = model(data)  # [N, 2]
        probs = torch.softmax(logits, dim=1)
        predictions = probs.argmax(dim=1)  # 0=fraud, 1=normal
        confidences = probs.max(dim=1).values

    # ------------------------------------------------------------------
    # 组装结果
    # ------------------------------------------------------------------
    labels = data.y.cpu().tolist()
    predictions = predictions.cpu().tolist()
    confidences = confidences.cpu().tolist()

    results = []
    for node_id in range(data.num_nodes):
        results.append(
            {
                "node_id": node_id,
                "prediction": predictions[node_id],
                "confidence": round(confidences[node_id], 6),
                "label": labels[node_id],
            }
        )

    # ------------------------------------------------------------------
    # 写出 JSON
    # ------------------------------------------------------------------
    output_dir = Path(__file__).resolve().parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predictions_{dataset}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 推理完成，共 {len(results)} 条预测结果已写入: {output_path}")


if __name__ == "__main__":
    main()
