import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm, skew, kurtosis
from linearmodels.panel import PanelOLS, BetweenOLS, FirstDifferenceOLS, compare, PooledOLS
from linearmodels.iv import IV2SLS
from statsmodels.tsa.stattools import adfuller

# Configuration
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams.update({'figure.figsize': (12, 8), 'axes.titlesize': 14})

# Chemins des fichiers
PATH_SECTOR = '/Data/country_sector_cleaned.dta'
PATH_COUNTRY = '/Data/country_cleaned.dta'

df_sector = pd.read_stata(PATH_SECTOR)
df_country = pd.read_stata(PATH_COUNTRY)

print("--- Structure Analysis ---")
n_countries = df_country['country'].nunique()
n_sectors = df_sector['industry'].nunique()

# Création ID unique
df_sector['id'] = df_sector['country'].astype(str) + "_" + df_sector['industry'].astype(str)
obs_counts = df_sector.groupby('id')['year'].count()

# Vérification consécutive
df_sector_sorted = df_sector.sort_values(['id', 'year'])
df_sector_sorted['year_diff'] = df_sector_sorted.groupby('id')['year'].diff()
consecutive = df_sector_sorted[df_sector_sorted['year_diff'] == 1]['id'].nunique()

print(f"Countries: {n_countries}, Sectors: {n_sectors}")
print(f"Total Individuals: {df_sector['id'].nunique()}")
print(f"Single Obs Individuals: {(obs_counts == 1).sum()}")
print(f"Consecutive Obs Individuals: {consecutive}")

print("\n--- Q4: Variance Decomposition ---")
cols_numeric = df_country.select_dtypes(include=np.number).columns.drop('year', errors='ignore')
results_var = []

for col in cols_numeric:
    subset = df_country[['country', col]].dropna()
    if len(subset) < 2: continue

    pooled_var = subset[col].var()
    if pooled_var == 0: continue

    means = subset.groupby('country')[col].transform('mean')
    between_var = means.var()
    within_var = (subset[col] - means).var()

    results_var.append({
        'Variable': col,
        'Pooled Var': pooled_var,
        'Between Var': between_var,
        'Within Var': within_var,
        'Between Share (%)': (between_var / pooled_var) * 100,
        'Within Share (%)': (within_var / pooled_var) * 100
    })

df_variance = pd.DataFrame(results_var)
print(df_variance.round(4).to_string(index=False))

print("\n--- Q5: Distributions ---")
df_q5 = df_country[['country', 'year', 'cotwo_total_per_cap', 'log_gdp_per_cap']].dropna()
for col in ['cotwo_total_per_cap', 'log_gdp_per_cap']:
    mean_i = df_q5.groupby('country')[col].transform('mean')
    df_q5[f'{col}_between'] = mean_i
    df_q5[f'{col}_within'] = df_q5[col] - mean_i

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

plot_list = [
    (df_q5['cotwo_total_per_cap_within'], "Within: CO2/Cap", 0, 'blue'),
    (df_q5['cotwo_total_per_cap_between'], "Between: CO2/Cap", 1, 'silver'),
    (df_q5['log_gdp_per_cap_within'], "Within: Log GDP/Cap", 2, 'blue'),
    (df_q5['log_gdp_per_cap_between'], "Between: Log GDP/Cap", 3, 'silver')
]

for data, title, idx, color in plot_list:
    mu, sigma = data.mean(), data.std()
    sns.histplot(data, kde=False, stat="density", ax=axes[idx], color='lightgray', alpha=0.5)
    sns.kdeplot(data, ax=axes[idx], color=color, linewidth=2)
    x = np.linspace(data.min(), data.max(), 200)
    axes[idx].plot(x, norm.pdf(x, mu, sigma), color='red', linestyle='--')
    axes[idx].set_title(title)

plt.tight_layout()
plt.savefig('Question_5_distribution_analysis.png')
plt.show()

print("\n--- Q6: First Differences ---")
df_fd = df_country.sort_values(['country', 'year']).copy()
for col in cols_numeric:
    df_fd[f'{col}_fd'] = df_fd.groupby('country')[col].diff()

df_fd.to_csv('Question_6_country_first_differences.csv', index=False)
print("FD CSV Exported.")

print("\n--- Q7: Comparative Distributions ---")
df_q7 = df_country[['country', 'year', 'cotwo_total_per_gdp', 'fin_str2']].dropna().sort_values(['country', 'year'])

# Transformations
for col in ['cotwo_total_per_gdp', 'fin_str2']:
    df_q7[f'{col}_within'] = df_q7[col] - df_q7.groupby('country')[col].transform('mean')
    df_q7[f'{col}_fd'] = df_q7.groupby('country')[col].diff()

# Plotting Grid
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plot_config = [
    (df_q7['cotwo_total_per_gdp_within'].dropna(), "Within: CO2/GDP", 0, 0, 'blue'),
    (df_q7['cotwo_total_per_gdp_fd'].dropna(), "FD: CO2/GDP", 0, 1, 'green'),
    (df_q7['fin_str2_within'].dropna(), "Within: Equity Share", 1, 0, 'blue'),
    (df_q7['fin_str2_fd'].dropna(), "FD: Equity Share", 1, 1, 'green')
]

for data, title, r, c, color in plot_config:
    sns.histplot(data, kde=False, stat="density", ax=axes[r, c], color='lightgray', alpha=0.5)
    sns.kdeplot(data, ax=axes[r, c], color=color, linewidth=2)
    x = np.linspace(data.min(), data.max(), 200)
    axes[r, c].plot(x, norm.pdf(x, data.mean(), data.std()), color='red', ls='--')
    axes[r, c].set_title(title)

plt.tight_layout()
plt.savefig('Question7_Comparative_Distributions.png')
plt.show()

print("\n--- Q8: Correlation ---")
g = sns.jointplot(x='fin_str2_fd', y='cotwo_total_per_gdp_fd', data=df_q7, kind='reg',
                  height=10, color='green', joint_kws={'scatter_kws': {'alpha': 0.4}})
corr_val = df_q7[['fin_str2_fd', 'cotwo_total_per_gdp_fd']].corr().iloc[0, 1]
g.fig.suptitle(f'Correlation: {corr_val:.4f}', y=1.02)
plt.savefig('Question8_FD_Correlation.png')
plt.show()

print("\n--- Q9: Balanced Panel ---")
counts = df_country.groupby('year')['country'].nunique()
balanced_years = counts[counts == counts.max()].index
country_counts = df_country[df_country['year'].isin(balanced_years)].groupby('country').size()
balanced_countries = country_counts[country_counts == len(balanced_years)].index

df_bal = df_country[(df_country['country'].isin(balanced_countries)) & (df_country['year'].isin(balanced_years))].copy()
print(f"Balanced Panel: N={len(balanced_countries)}, T={len(balanced_years)}")

dep_var = 'cotwo_total_per_gdp'
grand_mean = df_bal[dep_var].mean()
time_component = -df_bal.groupby('year')[dep_var].mean() + grand_mean

plt.figure(figsize=(10, 6))
plt.plot(time_component.index, time_component.values, marker='o', color='purple')
plt.title('Time-Specific Component')
plt.savefig('Question9_Time_Component.png')
plt.show()

print("\n--- Q10: Boxplots ---")
variances = df_bal.groupby('country')[dep_var].var().sort_values()
plt.figure(figsize=(16, 8))
sns.boxplot(x='country', y=dep_var, data=df_bal, order=variances.index, palette='viridis')
plt.xticks(rotation=90)
plt.title('Boxplots ordered by Variance')
plt.savefig('Question10_Boxplots_by_Variance.png')
plt.show()

print("\n--- Q11: TWFE Correlations ---")
# Transformation TWFE
for col in [dep_var, 'log_gdp_per_cap']:
    grand = df_bal[col].mean()
    m_i = df_bal.groupby('country')[col].transform('mean')
    m_t = df_bal.groupby('year')[col].transform('mean')
    df_bal[f'{col}_twfe'] = df_bal[col] - m_i - m_t + grand

stats_q11 = []
for c, group in df_bal.groupby('country'):
    rho = group[[f'{dep_var}_twfe', 'log_gdp_per_cap_twfe']].corr().iloc[0, 1]
    stats_q11.append({'Country': c, 'Correlation': rho})

df_stats_q11 = pd.DataFrame(stats_q11).sort_values('Correlation', ascending=False)
print(df_stats_q11.head().to_string(index=False))
df_stats_q11.to_csv('Question11_TWFE_Correlations.csv', index=False)

print("\n--- Q13: Unbalanced Transformations ---")
obs_counts_unbal = df_country.groupby('country').size()
valid_countries = obs_counts_unbal[obs_counts_unbal > 1].index
df_unbal = df_country[df_country['country'].isin(valid_countries)].copy().sort_values(['country', 'year'])

# Calculer Between, Within, TWFE, FD pour les variables clés
vars_target = ['cotwo_total_per_gdp', 'log_gdp_per_cap', 'fin_str2']
for col in vars_target:
    # Between
    df_unbal[f'{col}_between'] = df_unbal.groupby('country')[col].transform('mean')
    # Within
    df_unbal[f'{col}_within'] = df_unbal[col] - df_unbal[f'{col}_between']
    # FD
    df_unbal[f'{col}_fd'] = df_unbal.groupby('country')[col].diff()
    # TWFE (Residuals)
    temp = df_unbal[['country', 'year', col]].dropna()
    temp['w'] = temp[col] - temp.groupby('country')[col].transform('mean')
    dummies = pd.get_dummies(temp['year'], prefix='year').astype(float)
    resid = sm.OLS(temp['w'], dummies).fit().resid
    df_unbal.loc[temp.index, f'{col}_twfe'] = resid

# Plotting Q13 (Simplifié)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
idx = 0
for var in ['cotwo_total_per_gdp', 'log_gdp_per_cap']:
    for trans, color in zip(['between', 'within', 'twfe'], ['silver', 'skyblue', 'gold']):
        data = df_unbal[f'{var}_{trans}'].dropna()
        sns.histplot(data, kde=True, stat="density", ax=axes[idx], color=color)
        axes[idx].set_title(f'{trans}: {var}')
        idx += 1
plt.tight_layout()
plt.savefig('Question13_Distribution_Comparison.png')
plt.show()

print("\n--- Q20: Regressions ---")
df_reg = df_unbal.set_index(['country', 'year'])
exog_vars = ['fin_str2', 'log_gdp_per_cap', 'credit', 'stock', 'pop_mil']
df_clean = df_reg[[dep_var] + exog_vars].dropna()

y = df_clean[dep_var]
X = sm.add_constant(df_clean[exog_vars])

res_be = BetweenOLS(y, X).fit()
res_fe = PanelOLS(y, X, entity_effects=True).fit(cov_type='clustered', cluster_entity=True)
res_twfe = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(cov_type='clustered', cluster_entity=True)
res_fd = FirstDifferenceOLS(y, df_clean[exog_vars]).fit(cov_type='clustered', cluster_entity=True)

comparison = compare({'Between': res_be, 'FE': res_fe, 'TWFE': res_twfe, 'FD': res_fd})
print(comparison)
with open('Question20_Regression_Table.txt', 'w') as f:
    f.write(str(comparison))

print("\n--- Q21: ARDL & IV ---")

df_ah = df_country.sort_values(['country', 'year']).copy()
y_raw = 'cotwo_total_per_gdp'
x_raw = 'fin_str2'

# Calcul Différences et Lags
for v in [y_raw, x_raw] + exog_vars:
    df_ah[f'{v}_fd'] = df_ah.groupby('country')[v].diff()

# Lags spécifiques pour modèle AH
df_ah['dy_lag1'] = df_ah.groupby('country')[f'{y_raw}_fd'].shift(1)
df_ah['dx_lag1'] = df_ah.groupby('country')[f'{x_raw}_fd'].shift(1)
df_ah['y_lev_lag2'] = df_ah.groupby('country')[y_raw].shift(2)
df_ah['x_lev_lag2'] = df_ah.groupby('country')[x_raw].shift(2)

# Q21.3 Fisher ADF Test
print("Running Fisher ADF...")
for label, col in [('Dep', f'{y_raw}_fd'), ('Exp', f'{x_raw}_fd')]:
    p_values = []
    for c in df_ah['country'].unique():
        s = df_ah[df_ah['country'] == c][col].dropna()
        if len(s) > 10:
            try:
                p_values.append(adfuller(s, autolag='AIC')[1])
            except:
                pass
    fisher_stat = -2 * np.sum(np.log(p_values))
    print(f"{label}: Fisher Stat={fisher_stat:.2f}")

# Q21.4 & Q21.5: Setup des Matrices
controls_fd = [f'{c}_fd' for c in exog_vars if c != x_raw]
# Liste complète pour nettoyage
cols_model = [f'{y_raw}_fd', 'dy_lag1', f'{x_raw}_fd', 'dx_lag1', 'y_lev_lag2', 'x_lev_lag2'] + controls_fd

df_model = df_ah[['country', 'year'] + cols_model].dropna().set_index(['country', 'year'])

# Définition Y
Y_final = df_model[f'{y_raw}_fd']

year_dummies = pd.get_dummies(df_model.index.get_level_values('year'), prefix='y', drop_first=True)
year_dummies.index = df_model.index

# --- OLS (Q21.4) ---
X_ols_cols = ['dy_lag1', f'{x_raw}_fd', 'dx_lag1'] + controls_fd
X_ols = sm.add_constant(pd.concat([df_model[X_ols_cols], year_dummies], axis=1))

res_ols_ardl = PooledOLS(Y_final, X_ols).fit(cov_type='clustered', cluster_entity=True)
print("\nOLS ARDL Results:")
print(res_ols_ardl.summary.tables[1])
with open('Question21.4_Regression_Table.txt', 'w') as f:
    f.write(str(res_ols_ardl.summary.tables[1]))

# --- IV (Q21.5) ---
X_endog = df_model[['dy_lag1', f'{x_raw}_fd']]

# Exogenous: Constant + dx_lag1 + controls + year_dummies
X_exog_cols = ['dx_lag1'] + controls_fd
X_exog = sm.add_constant(pd.concat([df_model[X_exog_cols], year_dummies], axis=1))

# Instruments: Lag 2 levels
Z_instr = df_model[['y_lev_lag2', 'x_lev_lag2']]

res_iv = IV2SLS(Y_final, X_exog, X_endog, Z_instr).fit(cov_type='clustered',
                                                       clusters=df_model.index.get_level_values('country'))
print("\nIV Results:")
print(res_iv.summary.tables[1])
with open('Question21.5_Regression_Table.txt', 'w') as f:
    f.write(str(res_iv.summary.tables[1]))

# Q21.7 IRF
print("\n--- Q21.7 IRF ---")
beta_y = res_iv.params['dy_lag1']
beta_1 = res_iv.params[f'{x_raw}_fd']
beta_2 = res_iv.params['dx_lag1']

irf_vals = [
    beta_1,
    beta_y * beta_1 + beta_2,
    (beta_y ** 2) * beta_1 + beta_y * beta_2,
    (beta_y ** 3) * beta_1 + (beta_y ** 2) * beta_2
]

plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4], irf_vals, marker='o', color='navy')
plt.axhline(0, color='red', ls='--')
plt.title('IRF: Delta CO2/GDP to Delta Equity Shock')
plt.savefig('Question21_7_IRF_Plot.png')
plt.show()

# Q21.8 Long Run
lrp = (beta_1 + beta_2) / (1 - beta_y)
print(f"\nLong Run Propensity: {lrp:.6f}")