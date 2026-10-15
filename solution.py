from pathlib import Path

import pandas as pd
import numpy as np
from model import Model


class Cleaner:
    """执行固定的特征变换，不从预测样本重新学习统计参数。"""

    def __init__(self, X: pd.DataFrame):
        self.X = X
        self.cleanedX = None

    def _process_pdays(self, df: pd.DataFrame):
        """
        矛盾情况 A：pdays == -1（说自己从未联系过），但 previous = 2（历史联系了 2 次）或者 poutcome = 'failure'（记录了上次失败）。
        矛盾情况 B：pdays == 180（说半年前联系过），但 previous = 0（历史联系次数写着 0）"""
        # 1. 拆分 pdays
        df['has_previous_contact'] = np.where(df['pdays'] == -1, 0, 1)
        df['contact_gap'] = np.where(df['pdays'] == -1, 0, df['pdays'])

        # 2. 标记冲突行
        conflict_mask = (
            ((df['pdays'] == -1)
             & ((df['previous'] > 0) | (df['poutcome'] != 'unknown')))
            | ((df['pdays'] != -1) & (df['previous'] == 0))
        )
        df['is_contact_conflict'] = conflict_mask.astype(int)

    def _process_big_data(self, df: pd.DataFrame):
        # 1. balance 对称对数变换
        df['balance_symlog'] = np.sign(df['balance']) * np.log1p(
            np.abs(df['balance'])
        )

        # 2. 非负长尾字段 log1p
        for col in ['duration', 'campaign', 'previous']:
            df[f'{col}_log'] = np.log1p(df[col])

    def clean(self, feature_columns=None):
        """返回清洗后的特征；预测时传入训练列名，对齐列数和顺序。"""
        # 从原始输入重新处理，重复调用也不会再次编码已处理的列。
        # 测试脚本即使传入整张表，标签和编号也不会进入模型特征。
        self.cleanedX = self.X.drop(columns=['id', 'age'], errors='ignore').copy()

        # 处理很大的数据 见 ai_test/长尾数据处理.md
        self._process_big_data(self.cleanedX)

        # 处理 Pdays 等字段 见 ai_test/pdays处理.md
        """
        | 最终特征名 | 数据来源与处理 | 作用 |
        | --- | --- | --- |
        | `has_previous_contact` | 从 `pdays` 提取二值（0/1） | 充当门控，区分新老客户 |
        | `contact_gap` | `pdays` 将 -1 转为 0 | 衡量老客户的时间衰减 |
        | `previous_log` | 对 `previous` 取 `log1p` | 压缩联系次数的长尾极大值（如联系了几十次的情况） |
        | `poutcome` | 类别编码（如 One-Hot） | 保留上次营销结果，供模型学习其与年龄的关系 |
        | `is_contact_conflict` | 逻辑冲突判断（0/1） | 保留冲突标记，不擅自修正原始记录 |
        """

        self._process_pdays(self.cleanedX)

        # 某些字段映射为0，1
        col_01 = ['default', 'housing', 'loan']
        for col in col_01:
            self.cleanedX[col] = self.cleanedX[col].map({'yes': 1, 'no': 0})
            if self.cleanedX[col].isna().any():
                raise ValueError(f'{col} 必须为 yes 或 no')

        # 对某些字段独热编码
        dummy_col = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
        self.cleanedX = pd.get_dummies(self.cleanedX, columns=dummy_col, dtype=np.int8)

        if feature_columns is not None:
            # 样本中没出现的训练类别补 0；训练未见类别不新增列。
            # 字符串 unknown 若在训练中出现，仍有自己的独热编码列。
            self.cleanedX = self.cleanedX.reindex(columns=feature_columns, fill_value=0)

        return self.cleanedX


class Solution:
    def __init__(self):
        """只在初始化时读取训练集、确定预处理参数并训练模型。"""
        train_path = Path(__file__).resolve().parent / 'train.csv'
        df = pd.read_csv(train_path)
        y_train = df['age'].to_numpy(dtype=float)

        # 1. 清洗整个训练集，并记住每一列的含义和顺序。
        X_clean = Cleaner(df).clean()
        self.feature_columns = X_clean.columns.tolist()
        X_train = X_clean.to_numpy(dtype=float)
        del df, X_clean

        # 2. 只用训练集计算标准化参数，预测时复用。
        if not np.isfinite(X_train).all():
            raise ValueError('清洗后的训练特征包含 NaN 或无穷值')
        self.X_mean = X_train.mean(axis=0)
        self.X_std = X_train.std(axis=0)
        # 常量列标准差为 0，改用 1 作除数，中心化后该列仍全为 0。
        self.X_std[self.X_std == 0] = 1.0
        X_train -= self.X_mean
        X_train /= self.X_std

        # 3. 特征维数由清洗结果确定，不再写死为原始的 15 列。
        self.model = Model()
        self.model.train(X_train, y_train, alpha=1.0)

    def forward(self, sample: dict) -> dict:
        """
        单样本推理接口

        Args:
            sample: 包含单个样本特征的字典（不包含age列）

        Returns:
            dict: 包含'prediction'键的字典，值为预测的年龄
        """
        # 每个请求使用局部 Cleaner，避免并发预测时互相修改数据。
        cleaner = Cleaner(pd.DataFrame([sample]))
        X_clean = cleaner.clean(feature_columns=self.feature_columns)
        X = X_clean.to_numpy(dtype=float)
        if not np.isfinite(X).all():
            raise ValueError('清洗后的预测特征包含 NaN 或无穷值')

        # 这里只应用训练参数，不重新计算均值、标准差，也不重新训练。
        X_scaled = (X - self.X_mean) / self.X_std

        prediction = self.model.predict(X_scaled)[0]

        return {'prediction': float(prediction)}
