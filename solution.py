import math
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import numpy as np
from model import Model

# 需要独热编码的类别列（与 Cleaner.clean 保持一致）
DUMMY_COLS = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
# 名称以类别列前缀开头、但其实是派生数值列（不是独热列）的字段。
NON_DUMMY_PREFIXED = {'contact_gap'}

# 本地显示进度；教师环境未安装 tqdm 时仍可正常训练和预测。
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


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
        self.cleanedX = pd.get_dummies(self.cleanedX, columns=DUMMY_COLS, dtype=np.int8)

        if feature_columns is not None:
            # 样本中没出现的训练类别补 0；训练未见类别不新增列。
            # 字符串 unknown 若在训练中出现，仍有自己的独热编码列。
            self.cleanedX = self.cleanedX.reindex(columns=feature_columns, fill_value=0)

        return self.cleanedX


class Solution:
    def __init__(self):
        """只在初始化时读取训练集、确定预处理参数并训练模型。"""
        # 岭回归直接解方程，没有逐轮训练；这里显示四个准备阶段的完成进度。
        progress = (tqdm(total=4, desc='读取训练集', unit='阶段', dynamic_ncols=True)
                    if tqdm is not None else nullcontext())
        with progress as bar:
            train_path = Path(__file__).resolve().parent / 'train.csv'
            df = pd.read_csv(train_path)
            y_train = df['age'].to_numpy(dtype=float)
            if bar is not None:
                bar.update(1)
                bar.set_description('清洗与编码')

            # 1. 清洗整个训练集，并记住每一列的含义和顺序。
            X_clean = Cleaner(df).clean()
            self.feature_columns = X_clean.columns.tolist()
            X_train = X_clean.to_numpy(dtype=float)
            del df, X_clean
            if bar is not None:
                bar.update(1)
                bar.set_description('标准化特征')

            # 2. 只用训练集计算标准化参数，预测时复用。
            if not np.isfinite(X_train).all():
                raise ValueError('清洗后的训练特征包含 NaN 或无穷值')
            self.X_mean = X_train.mean(axis=0)
            self.X_std = X_train.std(axis=0)
            # 常量列标准差为 0，改用 1 作除数，中心化后该列仍全为 0。
            self.X_std[self.X_std == 0] = 1.0
            X_train -= self.X_mean
            X_train /= self.X_std
            if bar is not None:
                bar.update(1)
                bar.set_description('求解岭回归')

            # 3. 特征维数由清洗结果确定，不再写死为原始的 15 列。
            self.model = Model()
            self.model.train(X_train, y_train, alpha=1.0)
            if bar is not None:
                bar.update(1)
                bar.set_description('训练完成')

        # 预计算单样本推理所需的映射，避免预测时反复构造 DataFrame / get_dummies。
        self._col_index = {name: i for i, name in enumerate(self.feature_columns)}
        self._n_features = len(self.feature_columns)
        # 每个类别列在训练中出现的取值（只有这些取值才有对应的独热列）。
        self._onehot_values = {}
        for name in self.feature_columns:
            if name in NON_DUMMY_PREFIXED:
                continue
            for col in DUMMY_COLS:
                prefix = col + '_'
                if name.startswith(prefix):
                    self._onehot_values.setdefault(col, set()).add(name[len(prefix):])
                    break

    def forward(self, sample: dict) -> dict:
        """
        单样本推理接口

        Args:
            sample: 包含单个样本特征的字典（不包含age列）

        Returns:
            dict: 包含'prediction'键的字典，值为预测的年龄
        """
        # 直接用预计算的映射拼装特征向量，避免每行构造 DataFrame 和 get_dummies。
        # 逻辑与 Cleaner.clean() 完全等价（列名、顺序、编码方式均一致）。
        idx = self._col_index
        v = np.zeros(self._n_features, dtype=float)

        # 1. balance 对称对数变换（_process_big_data）
        balance = sample['balance']
        v[idx['balance']] = balance
        v[idx['balance_symlog']] = math.copysign(math.log1p(abs(balance)), balance)

        # 2. 非负长尾字段 log1p（_process_big_data）
        for col in ('duration', 'campaign', 'previous'):
            value = sample[col]
            v[idx[col]] = value
            v[idx[col + '_log']] = math.log1p(value)

        # 3. pdays 拆分与冲突标记（_process_pdays）
        pdays = sample['pdays']
        previous = sample['previous']
        poutcome = sample['poutcome']
        v[idx['pdays']] = pdays
        v[idx['day']] = sample['day']
        if pdays != -1:
            v[idx['has_previous_contact']] = 1
            v[idx['contact_gap']] = pdays
        if ((pdays == -1) and (previous > 0 or poutcome != 'unknown')) or \
                (pdays != -1 and previous == 0):
            v[idx['is_contact_conflict']] = 1

        # 4. yes/no 映射为 1/0（非法取值与 Cleaner 一样报错）
        for col in ('default', 'housing', 'loan'):
            value = sample[col]
            if value == 'yes':
                v[idx[col]] = 1
            elif value == 'no':
                v[idx[col]] = 0
            else:
                raise ValueError(f'{col} 必须为 yes 或 no')

        # 5. 独热编码：只置 1，训练未见类别自然保持全 0（等价 reindex(fill_value=0)）
        for col, seen in self._onehot_values.items():
            value = sample[col]
            if value in seen:
                v[idx[f'{col}_{value}']] = 1

        if not np.isfinite(v).all():
            raise ValueError('清洗后的预测特征包含 NaN 或无穷值')

        # 这里只应用训练参数，不重新计算均值、标准差，也不重新训练。
        X_scaled = (v - self.X_mean) / self.X_std
        prediction = self.model.predict(X_scaled[None, :])[0]

        return {'prediction': float(prediction)}
