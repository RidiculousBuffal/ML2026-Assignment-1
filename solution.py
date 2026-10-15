import pandas as pd
import numpy as np
from model import Model




class Cleaner:
    def __init__(self,X:pd.DataFrame):
        self.X = X
        self.cleanedX = self.X.copy()



    def _process_pdays(self,df:pd.DataFrame):

        """
        矛盾情况 A：pdays == -1（说自己从未联系过），但 previous = 2（历史联系了 2 次）或者 poutcome = 'failure'（记录了上次失败）。
        矛盾情况 B：pdays == 180（说半年前联系过），但 previous = 0（历史联系次数写着 0）"""
        # 1. 拆分 pdays
        df['has_previous_contact'] = np.where(df['pdays'] == -1, 0, 1)
        df['contact_gap'] = np.where(df['pdays'] == -1, 0, df['pdays'])

        # 2. 标记冲突行
        conflict_mask = ((df['pdays'] == -1) & ((df['previous'] > 0) | (df['poutcome'] != 'unknown'))) | ((df['pdays'] != -1) & (df['previous'] == 0))
        df['is_contact_conflict'] = conflict_mask.astype(int)


    def _process_big_data(self,df:pd.DataFrame):
        # 1. balance 对称对数变换
        df['balance_symlog'] = np.sign(df['balance']) * np.log1p(
            np.abs(df['balance'])
        )

        # 2. 非负长尾字段 log1p
        for col in ['duration', 'campaign', 'previous']:
            df[f'{col}_log'] = np.log1p(df[col])




    def clean(self):

        # 处理很大的数据 见 ai_test/长尾数据处理.md
        self._process_big_data(self.cleanedX)


        # 处理 Pdays 等字段 见 ai_test/pdays处理.md
        """
        | 最终特征名 | 数据来源与处理 | 作用 |
        | --- | --- | --- |
        | `has_previous_contact` | 从 `pdays` 提取二值（0/1） | 充当门控，区分新老客户 |
        | `contact_gap` | `pdays` 将 -1 转为 0 | 衡量老客户的时间衰减 |
        | `previous_log` | 对 `previous` 取 `log1p` | 压缩联系次数的长尾极大值（如联系了几十次的情况） |
        | `poutcome` | 类别编码（如 One-Hot） | 保留上次成败反馈（success 的转化率通常极高） |
        | `is_contact_conflict` | 逻辑冲突判断（0/1） | 标记脏数据行，防止噪声干扰主干规律 |
        """

        self._process_pdays(self.cleanedX)


        # 某些字段映射为0，1
        col_01 = ['default','housing','loan']
        for col in col_01:
            self.cleanedX[col] = self.cleanedX[col].map({'yes':1,'no':0})

        # 对某些字段独热编码
        dummy_col = ['job','marital','education','contact','month','poutcome']
        self.cleanedX = pd.get_dummies(self.cleanedX, columns=dummy_col,dtype=int)






class Solution:
    def __init__(self):
        """
        初始化随机权重示例，供你检查接口；它不是教师评分基线。
        """
        self.model = Model()
        np.random.seed(42)
        n_features = 15

        self.model.weights = np.random.randn(n_features) * 0.01
        self.model.bias = np.random.randn() * 0.01

        self.X_mean = np.zeros(n_features)
        self.X_std = np.ones(n_features)

    def forward(self, sample: dict) -> dict:
        """
        单样本推理接口

        Args:
            sample: 包含单个样本特征的字典（不包含age列）

        Returns:
            dict: 包含'prediction'键的字典，值为预测的年龄
        """
        sample_df = pd.DataFrame([sample])

        # The evaluator supplies the 15 features; id and age are excluded.
        X = sample_df.drop(columns=['id', 'age'], errors='ignore')

        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]

        X = X.values.astype(float)
        X = np.nan_to_num(X, nan=0.0)

        X_scaled = (X - self.X_mean) / self.X_std

        prediction = self.model.predict(X_scaled)[0]

        return {'prediction': float(prediction)}
