import pandas as pd
import io

# Load the CSV
df = pd.read_csv('/workspace/FATBETA/paper/order.csv')

# Clean column names (strip whitespace)
df.columns = df.columns.str.strip()

# Numeric conversion for '订单金额' (Order Amount)
# It might contain string characters or be clean, let's coerce
df['订单金额'] = pd.to_numeric(df['订单金额'], errors='coerce').fillna(0.0)

# 1. Total Spend
total_spend = df['订单金额'].sum()
print(f"=== 总支出 (Total Spending) ===\n{total_spend:.2f} 元\n")

# 2. Spend by Resource ID
# We also take '资源类型' and '新资源配置' (or '老资源配置' if new is missing) to describe the resource
# Fill generic NaN for mapping
df['ResourceDesc'] = df['新资源配置'].fillna(df['老资源配置']).fillna('')
df['ResourceName'] = df['资源标识'].fillna('')

# Group by Resource ID
resource_group = df.groupby(['资源ID', '资源类型', 'ResourceName', 'ResourceDesc'])['订单金额'].sum().reset_index()
resource_group = resource_group.sort_values(by='订单金额', ascending=False)

print("=== 按资源汇总 (Spending by Resource) ===")
# Set pandas display options to print simplified columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.float_format', '{:.2f}'.format)

# Select columns to show
result_df = resource_group[['资源ID', '资源类型', '订单金额', 'ResourceName']]
print(result_df)
print("\n")

# 3. Top Expensive Orders
print("=== 金额最大的前 10 笔订单 (Top 10 Expensive Orders) ===")
top_orders = df.sort_values(by='订单金额', ascending=False).head(10)
print(top_orders[['订单号', '订单金额', '订单类型', '订单创建时间(北京时间)', '资源ID', 'ResourceName']])
