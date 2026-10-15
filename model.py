"""NumPy 岭回归及仅从训练数据拟合的表格预处理。"""

import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = (
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome",
)
NUMERIC_FEATURES = (
    "balance", "day", "duration", "campaign", "pdays", "previous",
)
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


class Preprocessor:
    """独热编码、数值中位数填充/标准化及缺失标记。

    保留业务类别 unknown；空类别与未见类别分别保留独立位置。
    pdays=-1 单独标记，其天数分量按缺失处理，不删除原始记录。
    """

    @staticmethod
    def _category(value):
        if pd.isna(value):
            return None
        value = str(value).strip().lower()
        return value if value else None

    @staticmethod
    def _number(value, name):
        try:
            value = float(value)
        except (ValueError, TypeError, OverflowError):
            return np.nan
        if not np.isfinite(value):
            return np.nan
        if name == "day" and not 1 <= value <= 31:
            return np.nan
        if name in ("duration", "campaign", "previous") and value < 0:
            return np.nan
        if name == "pdays" and value < 0:
            return np.nan
        return value

    def _numeric(self, frame):
        # 与单样本采用相同清洗规则，不根据验证集拟合任何阈值。
        return np.column_stack([
            frame[name].map(lambda value: self._number(value, name)).to_numpy(
                dtype=np.float64
            ) for name in NUMERIC_FEATURES
        ])

    def fit(self, frame):
        if frame.empty:
            raise ValueError("训练数据不能为空")
        missing = set(FEATURES) - set(frame.columns)
        if missing:
            raise ValueError(f"缺少训练字段: {sorted(missing)}")
        self.category_maps = {}
        self.category_offsets = {}
        self.feature_names = []
        offset = 0
        for name in CATEGORICAL_FEATURES:
            values = frame[name].map(self._category)
            categories = sorted(set(values.dropna()))
            # 0: 缺失；1: 未见类别；2 起: 训练中实际出现的类别。
            self.category_maps[name] = {
                category: i + 2 for i, category in enumerate(categories)
            }
            self.category_offsets[name] = offset
            self.feature_names.extend(
                [f"{name}=<missing>", f"{name}=<unseen>"]
                + [f"{name}={category}" for category in categories]
            )
            offset += len(categories) + 2
        self.numeric_offset = offset
        numeric = self._numeric(frame)
        self.medians = np.array([
            np.median(column[np.isfinite(column)])
            if np.isfinite(column).any() else 0.0
            for column in numeric.T
        ])
        filled = np.where(np.isfinite(numeric), numeric, self.medians)
        self.means = filled.mean(axis=0)
        self.scales = filled.std(axis=0)
        self.scales[self.scales < 1e-12] = 1.0
        self.feature_names.extend(NUMERIC_FEATURES)
        self.feature_names.extend(f"{name}_missing" for name in NUMERIC_FEATURES)
        self.feature_names.append("never_contacted")
        self.n_features = len(self.feature_names)
        return self

    def transform(self, frame):
        if not hasattr(self, "category_maps"):
            raise ValueError("请先拟合预处理器")
        X = np.zeros((len(frame), self.n_features), dtype=np.float64)
        rows = np.arange(len(frame))
        for name in CATEGORICAL_FEATURES:
            values = frame[name].map(self._category)
            codes = values.map(self.category_maps[name]).fillna(1).to_numpy(dtype=int)
            codes[values.isna().to_numpy()] = 0
            X[rows, self.category_offsets[name] + codes] = 1.0
        numeric = self._numeric(frame)
        missing = ~np.isfinite(numeric)
        start = self.numeric_offset
        end = start + len(NUMERIC_FEATURES)
        X[:, start:end] = (
            np.where(missing, self.medians, numeric) - self.means
        ) / self.scales
        X[:, end:end + len(NUMERIC_FEATURES)] = missing
        pdays = pd.to_numeric(frame["pdays"], errors="coerce").to_numpy(dtype=float)
        X[:, -1] = pdays == -1
        return X

    def transform_one(self, sample):
        """只分配当前请求的数组，不修改共享状态，避免单行 DataFrame 开销。"""
        if not hasattr(self, "category_maps"):
            raise ValueError("请先拟合预处理器")
        X = np.zeros(self.n_features, dtype=np.float64)
        for name in CATEGORICAL_FEATURES:
            value = self._category(sample.get(name))
            code = 0 if value is None else self.category_maps[name].get(value, 1)
            X[self.category_offsets[name] + code] = 1.0
        numeric = np.array([
            self._number(sample.get(name), name) for name in NUMERIC_FEATURES
        ])
        missing = ~np.isfinite(numeric)
        start = self.numeric_offset
        end = start + len(NUMERIC_FEATURES)
        X[start:end] = (
            np.where(missing, self.medians, numeric) - self.means
        ) / self.scales
        X[end:end + len(NUMERIC_FEATURES)] = missing
        try:
            X[-1] = float(sample.get("pdays")) == -1
        except (TypeError, ValueError, OverflowError):
            X[-1] = 0
        return X


class Model:
    """最小化 sum((y - Xw - b)**2) + alpha * sum(w**2)。

    alpha 必须为正，截距不正则化。通过中心化充分统计量求解，
    无需构造完整的中心化矩阵或显式求逆。
    """

    def __init__(self, alpha=1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha 必须为有限正数")
        self.alpha = float(alpha)
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X 应为二维矩阵，y 应为等长的一维数组")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("训练样本和特征不能为空")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("训练输入必须为有限数值")
        x_mean = X.mean(axis=0)
        y_mean = y.mean()
        gram = X.T @ X - len(X) * np.outer(x_mean, x_mean)
        gram.flat[::gram.shape[0] + 1] += self.alpha
        rhs = X.T @ y - len(X) * x_mean * y_mean
        self.weights = np.linalg.solve(gram, rhs)
        self.bias = float(y_mean - x_mean @ self.weights)
        return self

    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("请先训练模型")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.weights):
            raise ValueError("X 应为二维矩阵，且特征数量应与训练时一致")
        if not np.isfinite(X).all():
            raise ValueError("预测输入必须为有限数值")
        return X @ self.weights + self.bias
