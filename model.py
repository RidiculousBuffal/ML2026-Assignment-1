import numpy as np


class Model:
    def __init__(self):
        """
        初始化岭回归模型
        """
        self.weights = None
        self.bias = None

    def train(self, X, y, alpha=1.0):
        """最小化 sum((Xw + b - y)^2) + alpha * sum(w^2)，不惩罚截距。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
            raise ValueError('X 应为非空二维矩阵，y 应为等长一维数组')
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('岭回归的 alpha 必须为有限正数')
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('训练数据包含 NaN 或无穷值')

        # 中心化后先求权重，再还原截距，避免将截距一起正则化。
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean
        gram = X_centered.T @ X_centered
        gram += alpha * np.eye(X.shape[1])
        self.weights = np.linalg.solve(gram, X_centered.T @ y_centered)
        self.bias = float(y_mean - X_mean @ self.weights)
        return self

    def predict(self, X):
        """
        预测样本的值

        Args:
            X: 输入特征，形状为 (n_samples, n_features) 的numpy数组

        Returns:
            numpy.ndarray: 形状为 (n_samples,) 的预测值数组
        """
        return np.dot(X, self.weights) + self.bias
