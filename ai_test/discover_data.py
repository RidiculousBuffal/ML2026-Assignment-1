from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = [
    'PingFang SC',       # macOS 苹方
    'Hiragino Sans GB',  # macOS 冬青黑体
    'Heiti SC',          # macOS 黑体-简
    'Arial Unicode MS',  # macOS 通用支持多语言字体
    'SimHei',            # Windows 黑体
    'Microsoft YaHei',   # Windows 微软雅黑
    'sans-serif'
]
plt.rcParams["axes.unicode_minus"] = False

def draw_age(df:pd.DataFrame, output_path='./age.png'):
    age_counts = df['age'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    plt.bar(age_counts.index, age_counts.values, color='steelblue', width=0.8)

    plt.title('各具体年龄频数分布', fontsize=14, pad=12)
    plt.xlabel('年龄', fontsize=12)
    plt.ylabel('人数', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def draw_age_by_job(df: pd.DataFrame, output_path='./age_by_job.png'):
    """按职业绘制各具体年龄的人数分布，每个职业占一个子图。"""
    job_groups = list(df.dropna(subset=['job', 'age']).groupby('job'))
    if not job_groups:
        return

    ncols = 3
    nrows = (len(job_groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(16, 3.5 * nrows), sharex=True, squeeze=False
    )

    for ax, (job, group) in zip(axes.flat, job_groups):
        age_counts = group['age'].value_counts().sort_index()
        ax.bar(age_counts.index, age_counts.values, color='steelblue', width=0.8)
        ax.set_title(f'{job}（n={len(group)}）', fontsize=12)
        ax.set_xlabel('年龄')
        ax.set_ylabel('人数')
        ax.tick_params(axis='x', labelbottom=True)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    for ax in list(axes.flat)[len(job_groups):]:
        ax.set_visible(False)

    fig.suptitle('各职业年龄分布', fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path)
    plt.close(fig)


CATEGORY_COLUMNS = [
    'job', 'marital', 'education', 'default', 'housing', 'loan',
    'contact', 'month', 'poutcome',
]
MONTH_ORDER = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
               'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
# 左开右闭。零余额、未联系过等特殊状态单独成组；长尾也保留在最后一组。
NUMERIC_BINS = {
    'balance': ([-float('inf'), -1, 0, 500, 1000, 2000, 5000, 10000, float('inf')],
                ['负余额', '零余额', '1–500', '501–1000', '1001–2000',
                 '2001–5000', '5001–10000', '>10000']),
    'day': ([0, 5, 10, 15, 20, 25, 31],
            ['1–5', '6–10', '11–15', '16–20', '21–25', '26–31']),
    'duration': ([-float('inf'), 60, 120, 180, 300, 600, 1200, float('inf')],
                 ['≤60秒', '61–120秒', '121–180秒', '181–300秒',
                  '301–600秒', '601–1200秒', '>1200秒']),
    'campaign': ([0, 1, 2, 3, 5, 10, float('inf')],
                 ['1次', '2次', '3次', '4–5次', '6–10次', '>10次']),
    'pdays': ([-2, -1, 30, 90, 180, 365, float('inf')],
              ['-1 未联系过', '0–30天', '31–90天', '91–180天', '181–365天', '>365天']),
    'previous': ([-1, 0, 1, 2, 3, 5, 10, float('inf')],
                 ['0次', '1次', '2次', '3次', '4–5次', '6–10次', '>10次']),
}


def summarize_group_age(df: pd.DataFrame, groups: pd.Series):
    """同时保留样本量和年龄分位数，避免只比较均值。"""
    grouped = df['age'].groupby(groups, observed=True, dropna=False)
    summary = grouped.agg(n='count', mean='mean', median='median', std='std')
    quantiles = grouped.quantile([0.1, 0.25, 0.75, 0.9]).unstack()
    quantiles.columns = ['p10', 'p25', 'p75', 'p90']
    summary = summary.join(quantiles)
    summary['fraction'] = summary['n'] / df['age'].count()
    return summary


def draw_grouped_age(summaries: dict, title: str, output_path, age_limits):
    """绘制分位数箱线图：箱体为中间50%，须为10%–90%，红菱形为均值。"""
    nrows = (len(summaries) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(19, 5.2 * nrows), squeeze=False)
    for ax, (column, summary) in zip(axes.flat, summaries.items()):
        stats = [dict(
            label=f'{label}  (n={int(row["n"]):,})',
            med=row['median'], q1=row['p25'], q3=row['p75'],
            whislo=row['p10'], whishi=row['p90'], mean=row['mean'],
        ) for label, row in summary.iterrows()]
        ax.bxp(stats, vert=False, showfliers=False, showmeans=True, patch_artist=True,
               boxprops={'facecolor': '#b4d3e8'},
               medianprops={'color': '#24445c'},
               meanprops={'marker': 'D', 'markerfacecolor': '#bf463b',
                          'markeredgecolor': '#bf463b', 'markersize': 4})
        ax.axvline(age_limits[2], color='gray', linestyle='--', alpha=0.7)
        ax.set_xlim(age_limits[0] - 2, age_limits[1] + 2)
        ax.set_title(column)
        ax.set_xlabel('年龄')
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.3)
    for ax in list(axes.flat)[len(summaries):]:
        ax.set_visible(False)
    fig.suptitle(title + '\n箱体：25%–75%；须：10%–90%；红菱形：均值；虚线：全体均龄',
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def draw_age_interactions(df: pd.DataFrame, output_path, min_count=100):
    """在同一职业内比较婚姻、房贷状态；小样本格只显示数量。"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), layout='constrained')
    tables = []
    for ax, column in zip(axes, ['marital', 'housing']):
        summary = df.groupby(['job', column])['age'].agg(n='count', mean='mean')
        tables.append(summary.reset_index().rename(columns={column: 'group'}).assign(
            feature=column))
        means = summary['mean'].unstack()
        counts = summary['n'].unstack().reindex_like(means)
        shown = means.where(counts >= min_count)
        mesh = ax.imshow(shown, cmap='YlOrRd', vmin=20, vmax=65, aspect='auto')
        ax.set_xticks(range(len(means.columns)), means.columns)
        ax.set_yticks(range(len(means.index)), means.index)
        ax.set_title(f'job × {column}')
        for i in range(len(means.index)):
            for j in range(len(means.columns)):
                n = counts.iloc[i, j]
                if pd.isna(n):
                    text = '无样本'
                elif n < min_count:
                    text = f'—\nn={int(n):,}'
                else:
                    text = f'{means.iloc[i, j]:.1f}岁\nn={int(n):,}'
                ax.text(j, i, text, ha='center', va='center', fontsize=9,
                        color='white' if shown.iloc[i, j] > 50 else 'black')
    fig.colorbar(mesh, ax=axes, label='平均年龄', shrink=0.8)
    fig.suptitle(f'职业内的年龄差异（n<{min_count} 的格子不显示均值）')
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return pd.concat(tables, ignore_index=True)


def explore_remaining_columns(df: pd.DataFrame, output_dir=None):
    """仅做训练集探索，输出全部字段概况、分组统计、图表，不训练预测模型。"""
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent / 'eda'
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = pd.DataFrame({
        'dtype': df.dtypes.astype(str), 'unique': df.nunique(),
        'missing': df.isna().sum(), 'unknown': df.eq('unknown').sum(),
    })
    overview.to_csv(output_dir / 'column_overview.csv', index_label='column')
    df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99]).to_csv(
        output_dir / 'numeric_overview.csv', index_label='statistic')
    df.select_dtypes('number').corr()['age'].to_csv(
        output_dir / 'age_pearson_correlation.csv', header=['pearson_r'])
    history = pd.DataFrame({
        'pdays_never': df['pdays'].eq(-1),
        'previous_zero': df['previous'].eq(0),
        'poutcome_unknown': df['poutcome'].eq('unknown'),
    })
    history.value_counts().rename('n').to_csv(output_dir / 'history_states.csv')
    quality = {
        'rows': len(df), 'columns': len(df.columns),
        'missing_cells': int(df.isna().sum().sum()),
        'duplicate_ids': int(df['id'].duplicated().sum()),
        'duplicates_excluding_id': int(df.drop(columns='id').duplicated().sum()),
        'history_pdays_previous_disagreements': int(
            history['pdays_never'].ne(history['previous_zero']).sum()),
        'balance_negative_count': int(df['balance'].lt(0).sum()),
        'balance_zero_count': int(df['balance'].eq(0).sum()),
    }
    (output_dir / 'quality_summary.json').write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    summaries = {}
    for column in CATEGORY_COLUMNS:
        summary = summarize_group_age(df, df[column])
        if column == 'month':
            summary = summary.reindex([m for m in MONTH_ORDER if m in summary.index])
        summaries[column] = summary
    for column, (edges, labels) in NUMERIC_BINS.items():
        groups = pd.cut(df[column], bins=edges, labels=labels)
        summaries[column] = summarize_group_age(df, groups)

    grouped_age = pd.concat([
        summary.rename_axis('group').reset_index().assign(feature=column)
        for column, summary in summaries.items()
    ], ignore_index=True)
    grouped_age.to_csv(output_dir / 'grouped_age.csv', index=False)

    # 描述性 eta²：分组均值解释的年龄方差比例。分箱会影响结果，不能当模型重要性。
    mean_age = df['age'].mean()
    total_ss = ((df['age'] - mean_age) ** 2).sum()
    association = pd.DataFrame([
        {'feature': column,
         'grouping': '类别' if column in CATEGORY_COLUMNS else '固定区间分箱',
         'eta_squared': (summary['n'] * (summary['mean'] - mean_age) ** 2).sum()
                        / total_ss if total_ss else 0.0}
        for column, summary in summaries.items()
    ]).sort_values('eta_squared', ascending=False)
    association.to_csv(output_dir / 'group_association.csv', index=False)

    age_limits = (df['age'].min(), df['age'].max(), mean_age)
    draw_grouped_age({c: summaries[c] for c in CATEGORY_COLUMNS if c != 'job'},
                     '类别特征对应的年龄分布', output_dir / 'age_by_category.png', age_limits)
    draw_grouped_age({c: summaries[c] for c in NUMERIC_BINS},
                     '数值特征分箱后的年龄分布', output_dir / 'age_by_numeric_bin.png', age_limits)
    interactions = draw_age_interactions(df, output_dir / 'age_interactions.png')
    interactions.to_csv(output_dir / 'age_interactions.csv', index=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(association['feature'], association['eta_squared'] * 100, color='steelblue')
    ax.invert_yaxis()
    ax.set_xlabel('分组均值解释的年龄方差（%）')
    ax.set_title('单字段的描述性关联（依赖分组方式，不是模型重要性）')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'group_association.png', dpi=140)
    plt.close(fig)

    print('\n字段概况\n', overview.to_string())
    print('\n单字段描述性关联\n', association.to_string(index=False))
    print(f'\n探索结果已保存至 {output_dir.resolve()}')
    return summaries, association


if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent
    df = pd.read_csv(script_dir.parent / 'train.csv', encoding='utf-8')
    print(df.head())
    print(f"一共有 {len(df.columns)} 列, 分别是 {list(df.columns)}")
#一共有 17 列, 分别是 ['id', 'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']

    """

    | 字段 | 类型 | 含义与取值说明 |
    |---|---|---|
    | `job` | 类别 | 职业，如 `admin.`、`blue-collar`、`entrepreneur` |
    | `marital` | 类别 | 婚姻状况，如 `married`、`single`、`divorced` |
    | `education` | 类别 | 教育程度，如 `primary`、`secondary`、`tertiary`、`unknown` |
    | `default` | 类别 | 是否有信贷违约记录，`yes` / `no` |
    | `balance` | 数值 | 年平均余额，单位为欧元 |
    | `housing` | 类别 | 是否有住房贷款，`yes` / `no` |
    | `loan` | 类别 | 是否有个人贷款，`yes` / `no` |
    | `contact` | 类别 | 联系方式，如 `unknown`、`telephone`、`cellular` |
    | `day` | 数值 | 最近联系的日期，取值为 1 至 31 |
    | `month` | 类别 | 最近联系的月份，如 `jan`、`feb`、`dec` |
    | `duration` | 数值 | 最近联系的持续时间，单位为秒 |
    | `campaign` | 数值 | 本次营销中的联系次数 |
    | `pdays` | 数值 | 距上次营销联系的天数；`-1` 表示此前未联系过 |
    | `previous` | 数值 | 本次营销之前的联系次数 |
    | `poutcome` | 类别 | 上次营销结果，如 `unknown`、`other`、`failure`、`success` |

    """


    # age
    print("年龄分布")
    print(df['age'].value_counts())
    draw_age(df, script_dir / 'age.png')

    # job
    """
    management       140255
    blue-collar      136432
    technician       110550
    admin.            65186
    services          51519
    retired           28174
    self-employed     15200
    entrepreneur      14226
    unemployed        14029
    housemaid         12718
    student            9384
    unknown            2327
    """
    print("职业")
    print(df['job'].value_counts())
    draw_age_by_job(df, script_dir / 'age_by_job.png')
    # explore_remaining_columns(df)
    print("pdays 不一致样本数")
    conflict_mask = (
                            (df['pdays'] == -1)
                            & ((df['previous'] > 0) | (df['poutcome'] != 'unknown'))
                    ) | ((df['pdays'] != -1) & (df['previous'] == 0))

    # 1. 打印查看这 21 行的原始分布
    inconsistent_rows = df[conflict_mask]
    print(f'不一致样本数: {len(inconsistent_rows)}')
    print(
        inconsistent_rows[['pdays', 'previous', 'poutcome', 'campaign']].head(10)
    )
    """
    
    Name: count, dtype: int64
    pdays 不一致样本数
    不一致样本数: 30
            pdays  previous poutcome  campaign
    1016       -1         0  success         1
    3308       -1         0  failure         1
    60261      -1         1  failure         1
    68768     342         0  unknown         5
    83973      -1         0  failure         1
    101043     -1         3  unknown         2
    151018     -1         0    other         2
    151043    194         0  failure         5
    151677     -1         0  failure         3
    168210     -1         1  unknown         3
    """