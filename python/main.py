import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from statsmodels.stats.multitest import multipletests


# загрузка и проверка качества данных
df = pd.read_csv('cookie_cats.csv')
assert df.notna().all().all(), 'Есть пропущенные значения'
assert not df['userid'].duplicated().all(), 'Есть повторяющиеся пользователи'
assert (df['sum_gamerounds'] >= 0).all(), 'Есть отрицательное количество раундов'
for column in ('retention_1', 'retention_7'):
    assert set(df[column].unique()).issubset({True, False, 0, 1}), 'Есть некорректные значения удержания'
    df[column] = df[column].astype(bool)

# разделение таблицы на контрольную и тестовую выборки
control = df[df['version'] == 'gate_30']
treatment = df[df['version'] == 'gate_40']

# расчёт метрик
summary = df.groupby('version').agg(
    users=('userid', 'size'),  # количество игроков
    retained_d1=('retention_1', 'sum'),  # количество вернувшихся на 1-й день
    retained_d7=('retention_7', 'sum'),  # количество вернувшихся на 7-й день
    rounds_mean=('sum_gamerounds', 'mean'),  # среднее количество раундов
    rounds_median=('sum_gamerounds', 'median'),  # медианное количество раундов
    rounds_p95=('sum_gamerounds', lambda s: s.quantile(0.95)),  # 95-й процентиль количества раундов
    rounds_p99=('sum_gamerounds', lambda s: s.quantile(0.99)),  # 99-й процентиль количества раундов
).reindex(['gate_30', 'gate_40'])
summary['retention_d1'] = summary['retained_d1'] / summary['users']  # доля удержания на 1-й день
summary['retention_d7'] = summary['retained_d7'] / summary['users']  # доля удержания на 7-й день
summary.to_csv('outputs/group_summary.csv', float_format='%.4f')

# проверка статистической значимости различий удержания между выборками
results = []
for name, col in [('D1', 'retention_1'), ('D7', 'retention_7')]:

    # проверка гипотез с помощью двустороннего z-теста
    ct, cc = int(treatment[col].sum()), int(control[col].sum())  # количество удержанных пользователей выборок
    nt, nc = len(treatment), len(control)  # количество всех пользователей выборок
    pt, pc = ct / nt, cc / nc  # доля удержания пользователей выборок
    z, p = proportions_ztest(
        [ct, cc],
        [nt, nc],
        alternative='two-sided'
    )  # z-тест

    # расчёт границ доверительного интервала
    lo, hi = confint_proportions_2indep(
        ct, nt,
        cc, nc,
        method='newcomb',
        compare='diff'
    )

    # все результаты
    results.append(dict(
        metric=name, n_control=nc, n_test=nt,
        successes_control=cc, successes_test=ct,
        rate_control=pc, rate_test=pt,
        diff_pp=(pt - pc) * 100,  # разница между долями удержания пользователей в выборках, процентные пункты
        relative_change_pct=(pt / pc - 1) * 100,  # изменение доли удержания пользователей в выборках, проценты
        ci_low_pp=lo * 100, ci_high_pp=hi * 100,  # границы доверительного интервала, процентные пункты
        z_stat=z,  # z-критерий
        p_value=p)  # p-value для z-теста
    )
tests = pd.DataFrame(results)

# поправка Холма на множественные проверки гипотез
tests['p_holm'] = multipletests(
    tests['p_value'],
    alpha=0.05,
    method='holm'
)[1]
tests['significant_holm_05'] = tests['p_holm'] < 0.05  # наличие статистической значимости с учётом поправки Холма
tests.to_csv('outputs/retention_tests.csv', index=False, float_format='%.4f')

# анализ активности игроков (количество раундов)
# изменилось ли количество раундов после переноса ограничения с 30 на 40
rounds_control = control['sum_gamerounds']
rounds_test = treatment['sum_gamerounds']
u, p_mwu = mannwhitneyu(
    rounds_test,
    rounds_control,
    alternative='two-sided',
    method='asymptotic'
)  # двусторонний тест Манна - Уитни для сравнения распределений
rounds = pd.DataFrame([{
    'mean_control': rounds_control.mean(), 'mean_test': rounds_test.mean(),  # среднее количество раундов
    'median_control': rounds_control.median(), 'median_test': rounds_test.median(),  # медианное количество раундов
    'p95_control': rounds_control.quantile(0.95), 'p95_test': rounds_test.quantile(0.95),  # 95-й процентиль количеств раундов
    'max_control': rounds_control.max(), 'max_test': rounds_test.max(),  # максимальное количество раундов
    'mann_whitney_u': u,  # U-статистика
    'mann_whitney_p_exploratory': p_mwu < 0.05  # наличие статистической значимости распределений в выборках
}])  # результаты
rounds.to_csv('outputs/rounds_exploratory.csv', index=False)

# визуализация
x = np.arange(2)
fig, ax = plt.subplots()
w = 0.33
ax.bar(
    x - w / 2,
    [summary.loc['gate_30','retention_d1'] * 100,
     summary.loc['gate_30','retention_d7'] * 100],
    w,
    label='Контрольная выборка'
)
ax.bar(
    x + w / 2,
    [summary.loc['gate_40','retention_d1'] * 100,
     summary.loc['gate_40','retention_d7'] * 100],
    w,
    label='Тестовая выборка'
)
ax.set_xticks(x, ['Удержание на 1-й день', 'Удержание на 7-й день'])
ax.set_ylabel('Удержание, %')
ax.set_title('Удержание в выборках')
fig.savefig('images/retention_comparison.png')
plt.close(fig)

fig, ax = plt.subplots()
limit = df['sum_gamerounds'].quantile(0.99)
ax.hist(
    [rounds_control.clip(upper=limit), rounds_test.clip(upper=limit)],
    label=['Контрольная', 'Тестовая']
)
ax.set_xlabel(f'Количество раундов')
ax.set_ylabel('Игроки')
ax.set_title('Распределение количества раундов в выборках')
fig.savefig('images/rounds_distribution.png')
plt.close(fig)
