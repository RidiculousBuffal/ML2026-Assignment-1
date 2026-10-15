"""真实 CSV 内部验证：python test_ridge.py（不读取公开验证集）。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

from model import Model, Preprocessor
from solution import Solution, load_training_data


def metrics(y, prediction):
    error = y - prediction
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": float(np.mean(error ** 2)),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, help="仅用于快速测试的随机子样本数")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    start = perf_counter()
    frame, labels = load_training_data(Path(__file__).resolve().parent / "train.csv")
    indices = np.random.default_rng(args.seed).permutation(len(frame))
    if args.rows is not None:
        if not 10 <= args.rows <= len(frame):
            parser.error("--rows 必须介于 10 与训练数据总行数之间")
        indices = indices[:args.rows]
    split = int(len(indices) * 0.8)
    train_indices, valid_indices = indices[:split], indices[split:]
    train, valid = frame.iloc[train_indices], frame.iloc[valid_indices]
    y_train, y_valid = labels[train_indices], labels[valid_indices]
    preprocessor = Preprocessor().fit(train)
    model = Model(alpha=args.alpha).fit(preprocessor.transform(train), y_train)
    prediction = model.predict(preprocessor.transform(valid))
    print(f"训练行数: {len(train)}, 验证行数: {len(valid)}, 特征数: {preprocessor.n_features}")
    print("均值基线:", metrics(y_valid, np.full(len(valid), y_train.mean())))
    print("岭回归:  ", metrics(y_valid, prediction))

    # 使用实际 forward 方法验证同一实例的多线程推理与批量预测一致。
    instance = Solution.__new__(Solution)
    instance.preprocessor, instance.model = preprocessor, model
    samples = valid.head(100).to_dict(orient="records")
    weights = model.weights.copy()
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(instance.forward, samples))
    np.testing.assert_allclose(
        [item["prediction"] for item in results], prediction[:len(samples)], atol=1e-9
    )
    np.testing.assert_array_equal(weights, model.weights)

    # 未见类别、空字段、无穷数和 pdays 特殊值的批量/单行一致性。
    dirty = valid.head(3).copy()
    dirty.loc[:, "job"] = [" new-job ", None, " ADMIN. "]
    dirty.loc[:, "balance"] = [np.inf, np.nan, -20]
    dirty.loc[:, "pdays"] = [-1, np.nan, 10]
    dirty.loc[:, "day"] = [0, 32, 15]
    batch = preprocessor.transform(dirty)
    single = np.vstack([
        preprocessor.transform_one(sample) for sample in dirty.to_dict(orient="records")
    ])
    np.testing.assert_allclose(batch, single)
    assert np.isfinite(batch).all()
    assert np.isfinite(model.predict(batch)).all()
    print("通过: 批量/单样本一致、10 线程推理、权重不变、异常值清洗")
    print(f"总耗时: {perf_counter() - start:.2f} 秒")


if __name__ == "__main__":
    main()
