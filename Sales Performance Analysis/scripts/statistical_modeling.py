
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('../data/sales_performance_data.csv')

# Aggregate data for modeling
agg_data = df.groupby(['Date']).agg({
    'Units_Sold': 'sum',
    'Revenue': 'sum',
    'Marketing_Spend': 'sum'
}).reset_index()

# Linear regression: Revenue vs Marketing Spend
X = agg_data[['Marketing_Spend']]
y = agg_data['Revenue']
model = LinearRegression()
model.fit(X, y)
print(f"R^2 Score: {model.score(X, y):.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")

# A/B Testing Simulation (Marketing Spend high vs low)
threshold = agg_data['Marketing_Spend'].median()
group_A = agg_data[agg_data['Marketing_Spend'] >= threshold]['Revenue']
group_B = agg_data[agg_data['Marketing_Spend'] < threshold]['Revenue']

from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(group_A, group_B)
print(f"A/B Test T-Stat: {t_stat:.2f}, P-Value: {p_value:.4f}")

# Visualization
sns.scatterplot(x='Marketing_Spend', y='Revenue', data=agg_data)
plt.title('Revenue vs Marketing Spend')
plt.xlabel('Marketing Spend')
plt.ylabel('Revenue')
plt.savefig('../assets/revenue_vs_marketing.png')
