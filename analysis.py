"""固定配置：保存 SHAP 图片，分析数据和特征工程对照结果打印到终端。"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from model import Model
from solution import Cleaner, DUMMY_COLS, Solution


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'analysis_output'
SEED = 42
BACKGROUND_SIZE = 512
SAMPLE_SIZE = 2000
LOCAL_INDEX = 0
VALIDATION_FRACTION = 0.2
ALPHA = 1.0  # 对照实验使用当前提交配置，不进行调参。
HISTORY_GROUP = 'history (pdays + previous + poutcome)'


def transform(solution, raw):
    """复用提交模型的列顺序及训练集标准化参数。"""
    cleaned = Cleaner(raw).clean(feature_columns=solution.feature_columns)
    values = (cleaned.to_numpy(dtype=float) - solution.X_mean) / solution.X_std
    if not np.isfinite(values).all():
        raise ValueError('分析输入含有 NaN 或无穷值')
    return values


def make_feature_groups(feature_names, raw_columns):
    """每列只归组一次；共同派生的历史联系字段合组。"""
    history = {
        'pdays', 'previous', 'previous_log', 'has_previous_contact',
        'contact_gap', 'is_contact_conflict',
    }
    derived = {
        'balance_symlog': 'balance', 'duration_log': 'duration',
        'campaign_log': 'campaign',
    }
    groups = []
    for name in feature_names:
        if name in history or name.startswith('poutcome_'):
            group = HISTORY_GROUP
        elif name in derived:
            group = derived[name]
        elif name in raw_columns:
            group = name
        else:
            matches = [col for col in DUMMY_COLS if name.startswith(col + '_')]
            if len(matches) != 1:
                raise ValueError(f'特征 {name} 没有明确的归组规则')
            group = matches[0]
        groups.append(group)
    return groups


def rank_contributions(values, names):
    """组级先有符号求和，再计算平均绝对贡献。"""
    return pd.DataFrame({
        'feature': names,
        'mean_abs_shap_years': np.abs(values).mean(axis=0),
        'mean_signed_shap_years': values.mean(axis=0),
    }).sort_values('mean_abs_shap_years', ascending=False, kind='stable')


def save_plots(explanation, grouped_values, group_names):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.xlabel('mean(|SHAP value|) (years)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'encoded_importance.png', dpi=150, bbox_inches='tight')
    plt.close('all')

    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.xlabel('SHAP value (years)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'encoded_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close('all')

    # 组包含多种数值/类别，没有单一的“特征值”，故不设置 data。
    grouped = shap.Explanation(
        values=grouped_values, base_values=explanation.base_values,
        feature_names=group_names,
    )
    shap.plots.bar(grouped, max_display=len(group_names), show=False)
    plt.xlabel('mean(|SHAP value|) (years)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'grouped_importance.png', dpi=150, bbox_inches='tight')
    plt.close('all')

    shap.plots.waterfall(grouped[LOCAL_INDEX], max_display=len(group_names), show=False)
    plt.gcf().set_size_inches(14, 9)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'local_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close('all')


def experiment_features(raw, variant):
    """生成实验特征，不改变 Solution 的提交配置。"""
    if variant == 'onehot_only':
        features = raw.drop(columns=['id', 'age']).copy()
        for col in ['default', 'housing', 'loan']:
            features[col] = features[col].map({'yes': 1, 'no': 0})
        return pd.get_dummies(features, columns=DUMMY_COLS, dtype=np.int8)

    features = Cleaner(raw).clean()
    if variant == 'job_marital_interaction':
        joint = (raw['job'] + '|' + raw['marital']).rename('job_marital')
        interaction = pd.get_dummies(joint, prefix='job_marital', dtype=np.int8)
        features = pd.concat([features, interaction], axis=1)
    elif variant == 'drop_low_shap':
        features = features.drop(columns=[
            'default', 'duration', 'duration_log', 'campaign', 'campaign_log',
        ])
    elif variant != 'current':
        raise ValueError(f'未知实验变体：{variant}')
    return features


def run_feature_experiments():
    """按 README 比较特征工程收益，只在训练集内部验证。"""
    frame = pd.read_csv(ROOT / 'train.csv')
    indices = np.random.default_rng(SEED).permutation(len(frame))
    split = int(len(frame) * (1 - VALIDATION_FRACTION))
    training = frame.iloc[indices[:split]].copy()
    validation = frame.iloc[indices[split:]].copy()
    del frame
    y_train = training['age'].to_numpy(dtype=float)
    y_valid = validation['age'].to_numpy(dtype=float)
    rows = []
    print(f'\n特征对照：训练 {len(training)} 条，内部验证 {len(validation)} 条，'
          f'seed={SEED}，alpha={ALPHA}', flush=True)
    for variant in ['onehot_only', 'current', 'job_marital_interaction', 'drop_low_shap']:
        clean_train = experiment_features(training, variant)
        clean_valid = experiment_features(validation, variant).reindex(
            columns=clean_train.columns, fill_value=0,
        )
        X_train = clean_train.to_numpy(dtype=float)
        X_valid = clean_valid.to_numpy(dtype=float)
        del clean_train, clean_valid
        # 类别列集合及标准化参数都仅由训练部分确定。
        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        std[std == 0] = 1
        X_train -= mean
        X_train /= std
        X_valid -= mean
        X_valid /= std
        model = Model().train(X_train, y_train, alpha=ALPHA)
        train_error = model.predict(X_train) - y_train
        valid_error = model.predict(X_valid) - y_valid
        rows.append({
            'variant': variant, 'features': X_train.shape[1],
            'train_rmse': np.sqrt(np.mean(train_error ** 2)),
            'valid_mae': np.mean(np.abs(valid_error)),
            'valid_rmse': np.sqrt(np.mean(valid_error ** 2)),
        })
        print(f'完成 {variant}', flush=True)
        del X_train, X_valid, model
    results = pd.DataFrame(rows)
    baseline = results.loc[results.variant.eq('current'), 'valid_rmse'].iloc[0]
    results['rmse_delta_vs_current'] = results['valid_rmse'] - baseline
    print(results.to_string(index=False, float_format=lambda x: f'{x:.8f}'))


def main():
    np.random.seed(SEED)  # 包括蜂群图中的随机抖动。
    frame = pd.read_csv(ROOT / 'train.csv')
    positions = np.random.default_rng(SEED).choice(
        len(frame), SAMPLE_SIZE + BACKGROUND_SIZE, replace=False,
    )
    background = frame.iloc[positions[:BACKGROUND_SIZE]].copy()
    samples = frame.iloc[positions[BACKGROUND_SIZE:]].copy()
    raw_columns = frame.drop(columns=['age', 'id']).columns.tolist()
    del frame

    solution = Solution()
    X_background = transform(solution, background)
    X_samples = transform(solution, samples)
    predictions = solution.model.predict(X_samples)
    forward_predictions = np.array([
        solution.forward(sample)['prediction']
        for sample in samples[raw_columns].to_dict('records')
    ])
    np.testing.assert_allclose(predictions, forward_predictions, rtol=0, atol=1e-8)

    masker = shap.maskers.Independent(X_background, max_samples=BACKGROUND_SIZE)
    explainer = shap.LinearExplainer(
        (solution.model.weights, solution.model.bias), masker,
        feature_names=solution.feature_columns,
    )
    explanation = explainer(X_samples)
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    reconstructed = base_values + values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, predictions, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        base_values, solution.model.predict(X_background).mean(), rtol=0, atol=1e-8,
    )
    if not np.isfinite(values).all():
        raise ValueError('SHAP 值包含 NaN 或无穷值')

    feature_groups = make_feature_groups(solution.feature_columns, raw_columns)
    group_names = list(dict.fromkeys(feature_groups))
    grouped_values = np.column_stack([
        values[:, np.array(feature_groups) == group].sum(axis=1)
        for group in group_names
    ])
    np.testing.assert_allclose(
        base_values + grouped_values.sum(axis=1), predictions, rtol=0, atol=1e-8,
    )
    importance = rank_contributions(values, solution.feature_columns)
    coefficients = dict(zip(solution.feature_columns, solution.model.weights))
    importance['coefficient_standardized'] = importance['feature'].map(coefficients)
    grouped_importance = rank_contributions(grouped_values, group_names)
    save_plots(explanation, grouped_values, group_names)

    print(f'\nSHAP {shap.__version__}，LinearExplainer + Independent masker')
    print(f'seed={SEED}，背景={BACKGROUND_SIZE}条，解释={SAMPLE_SIZE}条，'
          f'编码特征={len(solution.feature_columns)}列；两组从训练集不重叠抽样')
    print(f'背景平均预测（基准年龄）：{base_values[0]:.8f} 岁')
    print('\n编码特征贡献与标准化系数：')
    print(importance.to_string(index=False, float_format=lambda x: f'{x:.8f}'))
    print('\n分组贡献（先有符号求和，再取绝对值）：')
    print(grouped_importance.to_string(index=False, float_format=lambda x: f'{x:.8f}'))
    print(f'\n单样本解释：抽样序号 {LOCAL_INDEX}，训练集行号 {samples.index[LOCAL_INDEX]}')
    print(samples.iloc[LOCAL_INDEX].to_string())
    local = pd.DataFrame({'group': group_names, 'shap_years': grouped_values[LOCAL_INDEX]})
    local = local.iloc[np.argsort(-np.abs(local['shap_years'].to_numpy()))]
    print(local.to_string(index=False, float_format=lambda x: f'{x:+.8f}'))
    print(f'预测年龄：{predictions[LOCAL_INDEX]:.8f}；'
          f'SHAP还原年龄：{reconstructed[LOCAL_INDEX]:.8f}')
    print('forward一致性最大误差：', np.max(np.abs(predictions - forward_predictions)))
    print('SHAP加和最大误差：', np.max(np.abs(reconstructed - predictions)))
    print(f'\n4张分析图片已保存到 {OUTPUT_DIR}')
    run_feature_experiments()


if __name__ == '__main__':
    main()
