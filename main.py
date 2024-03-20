import pandas as pd

cash_csv_path = '~/tax_csv/txn_csv/cash_app_report_btc.csv'
t21_csv_path = '~/tax_csv/txn_csv/transactions21.csv'
f22_csv_path = '~/tax_csv/txn_csv/fold_app_report_2022.csv'

df_cash = pd.read_csv(cash_csv_path)
df_t21 = pd.read_csv(t21_csv_path)
df_f22 = pd.read_csv(f22_csv_path)
##df_cash['Date'] = pd.to_datetime(df_cash['Date'], format='%Y-%m-%d')
##df_cash = pd.read_csv(cash_csv_path, parse_dates=['Date'])

## DATA CLEANING
# Removing timezone information from the string
df_cash['Date'] = df_cash['Date'].str.slice(stop=-4)  # Assuming the timezone abbreviation is always 3 characters long, preceded by a space
df_cash['Date'] = pd.to_datetime(df_cash['Date'], format='%Y-%m-%d %H:%M:%S')

df_t21['Date'] = df_t21['Date'].str.slice(stop=-4)  # Assuming the timezone abbreviation is always 3 characters long, preceded by a space
df_t21['Date'] = pd.to_datetime(df_t21['Date'], format='%Y-%m-%d %H:%M:%S')

df_f22['Date'] = df_f22['Date'].str.slice(stop=-4)  # Assuming the timezone abbreviation is always 3 characters long, preceded by a space
df_f22['Date'] = pd.to_datetime(df_f22['Date'], format='%Y-%m-%d %H:%M:%S')

##diagnostics
#print(df_cash.head(20))
#print(df_cash[['Date','Amount','Fee','Net Amount','Asset Amount']].head(20))
#print(df_cash['Amount'].apply(type).unique())

# Removing commas, $ , whitespace and convert to positive numbers
columns_to_clean = ['Amount', 'Fee', 'Net Amount', 'Asset Price']
    
for column in columns_to_clean:
    df_cash[column] = df_cash[column].str.replace(',', '')
    df_cash[column] = df_cash[column].str.replace('$', '')
    df_cash[column] = df_cash[column].str.strip()  # Removes leading/trailing spaces
    df_cash[column] = pd.to_numeric(df_cash[column], errors='coerce').abs()

    df_t21[column] = df_t21[column].str.replace(',', '')
    df_t21[column] = df_t21[column].str.replace('$', '')
    df_t21[column] = df_t21[column].str.strip()  # Removes leading/trailing spaces
    df_t21[column] = pd.to_numeric(df_t21[column], errors='coerce').abs()

    df_f22[column] = pd.to_numeric(df_f22[column], errors='coerce').abs()

##diagnostics
#print(df_cash[['Date','Amount','Fee','Net Amount','Asset Price','Asset Amount']].head(20))
#print(df_cash.head(20))
    
#print(df_t21[['Date','Amount','Fee','Net Amount','Asset Price','Asset Amount']].head(20))
#print(df_t21.head(20))

print(df_f22[['Date','Amount','Fee','Net Amount','Asset Price','Asset Amount']].head(20))
print(df_f22.head(20))

#remove NaN entry (deposit from wallet, accounted for in other wallets)
df_cash = df_cash[df_cash['Transaction ID'] != 'ukknp3']

##diagnostics
#print(df_cash[['Date','Amount','Fee','Net Amount','Asset Price','Asset Amount']])
#df_subset = df_cash.iloc[149:181]
#print(df_subset[['Date','Amount','Fee','Net Amount','Asset Price','Asset Amount']])
#print(df_subset[['Transaction ID']])
# Remove deposit with NaN, accounted for in other utxos; 'Transaction ID' is a column with unique identifiers

##categorize transactions into 'buy' 'sale' and 'received' conditions
def categorize_transaction(row):
    # Lowercase the transaction type for case-insensitive matching
    transaction_type = row['Transaction Type'].lower()
    
    if 'purchase' in transaction_type or 'buy' in transaction_type:
        return 'Buys'
    if 'sale' in transaction_type:
        return 'Sales'
    if 'reward' in transaction_type or 'bonus' in transaction_type or 'boost' in transaction_type or 'p2p' in transaction_type:
        return 'Recieve'
    # Add more conditions as necessary
    else:
        return 'Other'

df_cash['Category'] = df_cash.apply(categorize_transaction, axis=1)
df_t21['Category'] = df_t21.apply(categorize_transaction, axis=1)
df_f22['Category'] = df_f22.apply(categorize_transaction, axis=1)

###chart diagnostics
"""
import matplotlib.pyplot as plt

df_cash['Amount'].hist(bins=50)
plt.title('Distribution of Amount')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

df_cash.boxplot(column=['Amount', 'Fee', 'Net Amount'])
plt.title('Box Plot of Financial Amounts')
plt.ylabel('Value')
plt.show()

plt.scatter(df_cash['Asset Price'], df_cash['Asset Amount'])
plt.title('Asset Price vs. Asset Amount')
plt.xlabel('Asset Price')
plt.ylabel('Asset Amount')
plt.show()

df_cash['Category'].value_counts().plot(kind='bar')
plt.title('Category')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.show()

df_cash['Transaction Type'].value_counts().plot(kind='bar')
plt.title('Transaction Types')
plt.xlabel('Type')
plt.ylabel('Frequency')
plt.show()



import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(df_cash.isnull(), cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Data')
plt.show()
"""
"""
import matplotlib.pyplot as plt

df_t21['Category'].value_counts().plot(kind='bar')
plt.title('Category')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.show()

df_t21['Transaction Type'].value_counts().plot(kind='bar')
plt.title('Transaction Types')
plt.xlabel('Type')
plt.ylabel('Frequency')
plt.show()

import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(df_t21.isnull(), cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Data')
plt.show()
"""

import matplotlib.pyplot as plt

df_f22['Category'].value_counts().plot(kind='bar')
plt.title('Category')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.show()

df_f22['Transaction Type'].value_counts().plot(kind='bar')
plt.title('Transaction Types')
plt.xlabel('Type')
plt.ylabel('Frequency')
plt.show()

import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(df_f22.isnull(), cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Data')
plt.show()


##diagnostics
#print(df_cash[df_cash['Category'] == 'Other'])

#print(df_cash[df_cash['Category'] == 'Other']['Notes'].unique())
#print(df_cash[df_cash['Category'] == 'Other']['Notes'].value_counts())
#print(df_cash[df_cash['Notes'] == "#CashAppBitcoin 🐝🍵🌊"])
