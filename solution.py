"""从本地 train.csv 训练岭回归，提供线程安全的单样本预测接口。"""

from pathlib import Path

import numpy as np
import pandas as pd

from model import FEATURES, Model, Preprocessor


def load_training_data(path):
    """仅读取允许的特征和标签；无有效标签时无法监督训练，直接报错。"""
    frame = pd.read_csv(path, usecols=list(FEATURES) + ["age"])
    labels = pd.to_numeric(frame.pop("age"), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(labels)
    if not valid.all():
        raise ValueError(f"训练数据包含 {(~valid).sum()} 条无效年龄标签，请检查 CSV")
    return frame, labels


class Solution:
    def __init__(self):
        frame, labels = load_training_data(Path(__file__).resolve().parent / "train.csv")
        self.preprocessor = Preprocessor().fit(frame)
        self.model = Model(alpha=1.0)
        self.model.fit(self.preprocessor.transform(frame), labels)

    def forward(self, sample: dict) -> dict:
        features = self.preprocessor.transform_one(sample)
        prediction = float(features @ self.model.weights + self.model.bias)
        if not np.isfinite(prediction):
            raise ValueError("预测结果不是有限数值，请检查输入")
        return {"prediction": prediction}
