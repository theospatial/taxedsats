import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
"""
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
        return 'Receives'
    # Add more conditions as necessary
    else:
        return 'Other'

df_cash['Category'] = df_cash.apply(categorize_transaction, axis=1)
df_t21['Category'] = df_t21.apply(categorize_transaction, axis=1)
df_f22['Category'] = df_f22.apply(categorize_transaction, axis=1)

###chart diagnostics
""""""

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




plt.figure(figsize=(10, 6))
sns.heatmap(df_cash.isnull(), cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Data')
plt.show()
"""
"""

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


plt.figure(figsize=(10, 6))
sns.heatmap(df_t21.isnull(), cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Data')
plt.show()
""""""


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

plt.figure(figsize=(10, 6))
sns.heatmap(df_f22.isnull(), cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Data')
plt.show()


##diagnostics
#print(df_cash[df_cash['Category'] == 'Other'])

#print(df_cash[df_cash['Category'] == 'Other']['Notes'].unique())
#print(df_cash[df_cash['Category'] == 'Other']['Notes'].value_counts())
#print(df_cash[df_cash['Notes'] == "#CashAppBitcoin 🐝🍵🌊"])
"""

def plot_source_distribution(df, title_suffix):
    source_counts = df['Source'].value_counts()
    source_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, title=f'Distribution of Transactions by Source for {title_suffix}')
    plt.ylabel('')  # Hide the y-label for clarity
    plt.show()

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df['Date'] = df['Date'].str.slice(stop=-4)  # Assuming the timezone abbreviation is always 3 characters long, preceded by a space
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    columns_to_clean = ['Amount', 'Fee', 'Net Amount', 'Asset Price', 'Asset Amount']

    for column in columns_to_clean:
        if column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].str.replace(',', '').str.replace('$', '').str.strip()
            df[column] = pd.to_numeric(df[column], errors='coerce').abs()
    return df

csv_paths = {
    'cash_app_22': '~/tax_csv/txn_csv/cash_app_report_btc.csv',
    'cash_app_21': '~/tax_csv/txn_csv/transactions21.csv',
    'fold_app_22': '~/tax_csv/txn_csv/fold_app_report_2022.csv',
    'swan_app_22': '~/tax_csv/txn_csv/swan_app_report_2022.csv'
}
print("TESTING DUPLICATES")
# Load CSV files
df1 = pd.read_csv(csv_paths['cash_app_22'])
df2 = pd.read_csv(csv_paths['cash_app_21'])

# Combine the DataFrames
combined_df = pd.concat([df1, df2])

# Identify duplicates based on 'Transaction ID'
duplicates = combined_df[combined_df.duplicated(subset='Transaction ID', keep=False)]

# Show duplicates
print(duplicates)
duplicates_by_id = combined_df[combined_df.duplicated(subset='Transaction ID', keep=False)]
print("Duplicates based on Transaction ID:\n", duplicates_by_id)
duplicates_by_amount = combined_df[combined_df.duplicated(subset='Asset Amount', keep=False)]
print("Duplicates based on Asset Amount:\n", duplicates_by_amount)

print("END OF TEST")


dataframes = []
for source, path in csv_paths.items():
    df = load_and_preprocess(path)
    df['Source'] = source
    dataframes.append(df)

unified_ledger = pd.concat(dataframes, ignore_index=True)
unified_ledger = unified_ledger.sort_values(by='Date')

#remove NaN entry (deposit from wallet, accounted for in other wallets)
unified_ledger = unified_ledger[unified_ledger['Transaction ID'] != 'ukknp3']

##categorize transactions into 'buy' 'sale' and 'received' conditions
def categorize_transaction(row):
    # Lowercase the transaction type for case-insensitive matching
    transaction_type = row['Transaction Type'].lower()
    
    if 'purchase' in transaction_type or 'buy' in transaction_type:
        return 'Buys'
    if 'sale' in transaction_type:
        return 'Sales'
    if 'reward' in transaction_type or 'bonus' in transaction_type or 'boost' in transaction_type or 'p2p' in transaction_type:
        return 'Receives'
    # Add more conditions as necessary
    else:
        return 'Other'


unified_ledger['Category'] = unified_ledger.apply(categorize_transaction, axis=1)


#diagnostics and visualizations:
print(unified_ledger.head())


unified_ledger['Date'].value_counts().sort_index().plot(kind='line')
plt.title('Transaction Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Transactions')
plt.show()

unified_ledger['Amount'].plot(kind='hist', bins=50, alpha=0.6)
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.show()

unified_ledger['Category'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Transaction Type Breakdown')
plt.ylabel('')  # Hide the y-label as it's redundant for pie charts
plt.show()

plt.scatter(unified_ledger['Date'], unified_ledger['Asset Price'], c=unified_ledger['Amount'], cmap='viridis', alpha=0.5)
plt.colorbar(label='Transaction Amount')
plt.title('Asset Price Over Time')
plt.xlabel('Date')
plt.ylabel('Asset Price')
plt.show()

"""
# Select only numeric columns for correlation matrix
numeric_cols = unified_ledger.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_cols.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Assuming 'Date' is already in datetime format and set as index
daily_summaries = unified_ledger.groupby([unified_ledger['Date'].dt.date, 'Category'])['Amount'].sum().unstack().fillna(0)

# Plotting
daily_summaries.plot(kind='area', stacked=True, figsize=(10, 6))
plt.title('Transaction Amounts by Type Over Time')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(unified_ledger.isnull(), cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Data')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Amount', data=unified_ledger)
plt.title('Transaction Amounts by Category')
plt.xlabel('Category')
plt.ylabel('Amount')
plt.xticks(rotation=45)  # Rotate category names for better readability
plt.show()
"""

# Assuming unified_ledger is already loaded and contains the necessary columns
"""
# Initialize the FIFO queue and tax ledger
fifo_queue = []
tax_ledger = []
gainloss_ledger = []

def populate_fifo_queue(unified_ledger):
    fifo_queue = []
    # Loop through the unified ledger and add 'buy' and 'receive' transactions to the FIFO queue
    for index, transaction in unified_ledger.iterrows():
        if transaction['Category'] in ['Buys', 'Receives']:
            fifo_queue.append({
                'Transaction ID': transaction['Transaction ID'],
                'Date Acquired': transaction['Date'],
                'Remaining Amount': transaction['Asset Amount'],
                'Cost Basis per Unit': transaction['Asset Price'],
                'Taxed on Receipt': False if transaction['Category'] == 'Buys' else True
            })
    return fifo_queue

# Function to process the FIFO queue for sales
def process_fifo_queue(fifo_queue, sale_amount, sale_price, sale_date):
    cost_basis = 0
    gains_losses = []

    while sale_amount > 0 and fifo_queue:
        # Get the first buy/receive transaction
        asset = fifo_queue.pop(0)
        print(fifo_queue[0])
        # Determine the amount to use from this transaction
        used_amount = min(sale_amount, asset['remaining_amount'])
        sale_amount -= used_amount
        cost_basis_for_this_amount = used_amount * asset['cost_basis_per_unit']
        cost_basis += cost_basis_for_this_amount

        # Calculate gain or loss for this portion
        gain_loss = (used_amount * sale_price) - cost_basis_for_this_amount
        gains_losses.append({
            'date_acquired': asset['date_acquired'],
            'date_sold': sale_date,
            'amount': used_amount,
            'cost_basis': cost_basis_for_this_amount,
            'sale_proceeds': used_amount * sale_price,
            'gain_loss': gain_loss,
            'holding_period': 'Long' if (sale_date - asset['date_acquired']).days > 365 else 'Short'
        })

        # If there is a remaining amount, put it back at the start of the FIFO queue
        if asset['remaining_amount'] > used_amount:
            asset['remaining_amount'] -= used_amount
            fifo_queue.insert(0, asset)  # Put the remaining asset back for the next sale

    return gains_losses, fifo_queue

# Function to add taxable 'receive' transactions to the FIFO queue and tax ledger
def add_receive_to_fifo_and_tax(transaction, fifo_queue, tax_ledger):
    fair_market_value = transaction['Asset Amount'] * transaction['Asset Price']
    tax_ledger.append({
        'date': transaction['Date'],
        'type': 'Income',
        'income_type': 'Bitcoin',
        'amount': fair_market_value,
    })
    fifo_queue.append({
        'date_acquired': transaction['Date'],
        'remaining_amount': transaction['Asset Amount'],
        'cost_basis_per_unit': transaction['Asset Price'],
    })
    return fifo_queue, tax_ledger

# Main function to assign gains and losses
def assign_gain_loss(unified_ledger, fifo_queue, tax_ledger):
    for index, transaction in unified_ledger.iterrows():
        if transaction['Category'] == 'Sales':
            # Process sale transactions using the FIFO queue
            sale_date = transaction['Date']
            sale_price = transaction['Asset Price']
            sale_amount = transaction['Asset Amount']
            
            # Process the FIFO queue to get the gain/loss for this sale
            gains_losses, fifo_queue = process_fifo_queue(fifo_queue, sale_amount, sale_price, sale_date)
            
            # Add the gains/losses to the gainloss ledger
            for gl in gains_losses:
                gainloss_ledger.append(gl)
        if transaction['Category'] == 'Receives':
            # Add receive transactions to FIFO queue and record them as taxable events
            fifo_queue, tax_ledger = add_receive_to_fifo_and_tax(transaction, fifo_queue, tax_ledger)

    # Return the gainloss ledger and tax ledger
    return gainloss_ledger, tax_ledger

# Call the main function with the unified ledger, FIFO queue, and tax ledger
gainloss_ledger, tax_ledger = assign_gain_loss(unified_ledger, fifo_queue, tax_ledger)

# Convert the ledgers to DataFrame for easier handling
gainloss_ledger_df = pd.DataFrame(gainloss_ledger)
tax_ledger_df = pd.DataFrame(tax_ledger)

# Sort the gainloss ledger by date sold
#gainloss_ledger_df.sort_values(by='date_sold', inplace=True)
print(gainloss_ledger_df.head())
print(tax_ledger_df.head())
print(unified_ledger.head())
print(unified_ledger['Category'].unique())

fifo_queue = populate_fifo_queue(unified_ledger)
print(fifo_queue[20])
#gainloss_ledger, tax_ledger = assign_gain_loss(unified_ledger, fifo_queue, tax_ledger)

""""""

def assign_gain_loss(unified_ledger):
    fifo_queue = []
    gainloss_ledger = []
    tax_ledger = []

    # Sort unified_ledger by date to ensure chronological order
    unified_ledger.sort_values(by='Date', inplace=True)

    for index, sale_transaction in unified_ledger.iterrows():
        if sale_transaction['Category'] == 'Sales':
            sale_date = sale_transaction['Date']
            sale_amount = sale_transaction['Asset Amount']
            sale_price = sale_transaction['Asset Price']
            gains_losses = []

            # Ensure there are enough buys/receives before processing a sale
            while sale_amount > 0 and any(buy_receive['Remaining Amount'] > 0 for buy_receive in fifo_queue):
                for asset in fifo_queue:
                    if sale_amount <= 0:
                        break  # Break if sale amount has been fully allocated

                    if asset['Remaining Amount'] <= 0:
                        continue  # Skip this asset if it has no remaining amount

                    used_amount = min(sale_amount, asset['Remaining Amount'])
                    sale_amount -= used_amount
                    asset['Remaining Amount'] -= used_amount

                    # Calculate and record gain/loss for this portion of the sale
                    cost_basis_for_this_amount = used_amount * asset['Cost Basis per Unit']
                    gain_loss = (used_amount * sale_price) - cost_basis_for_this_amount
                    gains_losses.append({
                        'Transaction ID': asset['Transaction ID'],
                        'Date Sold': sale_date,
                        'Used Amount': used_amount,
                        'Cost Basis': cost_basis_for_this_amount,
                        'Sale Proceeds': used_amount * sale_price,
                        'Gain/Loss': gain_loss
                    })

                    if asset['Remaining Amount'] > 0:
                        # Adjust asset dictionary as needed before re-adding
                        modified_asset = {
                        'Transaction ID': asset['Transaction ID'],
                        'Date Acquired': asset['Date Acquired'],
                        'Remaining Amount': asset['Remaining Amount'],
                        'Cost Basis per Unit': asset['Cost Basis per Unit'],
                        'Category': asset['Category']  # Include other necessary fields
                        }
                        fifo_queue.insert(0, modified_asset) # potentially by re-adding it to the FIFO queue or handling it separately.

            # Add gains/losses for this sale to the ledger
            gainloss_ledger.extend(gains_losses)

        elif sale_transaction['Category'] in ['Buys', 'Receives']:
            # Populate FIFO queue with buys and receives as they appear chronologically
            fifo_queue.append({
                'Transaction ID': sale_transaction['Transaction ID'],
                'Date Acquired': sale_transaction['Date'],
                'Remaining Amount': sale_transaction['Asset Amount'],
                'Cost Basis per Unit': sale_transaction['Asset Price'],
                'Category': sale_transaction['Category']
            })
            if sale_transaction['Category'] == 'Receives':
                # Handle immediate taxation for receives
                # Similar to previous logic, add to tax_ledger

        # At this point, gainloss_ledger contains the gain/loss information for sales transactions
        # fifo_queue contains unsold assets, and tax_ledger contains taxable events from receives
        return pd.DataFrame(gainloss_ledger), pd.DataFrame(tax_ledger)
"""

def assign_gain_loss(unified_ledger):
    fifo_queue = []
    gainloss_ledger = []
    tax_ledger = []

    # Sort unified_ledger by date to ensure chronological order
    unified_ledger.sort_values(by='Date', inplace=True)

    for index, transaction in unified_ledger.iterrows():
        if transaction['Category'] == 'Sales':
            sale_date = transaction['Date']
            sale_amount = transaction['Asset Amount']
            sale_price = transaction['Asset Price']
            gains_losses = []

            while sale_amount > 0 and fifo_queue:
                asset = fifo_queue.pop(0)
                used_amount = min(sale_amount, asset['Remaining Amount'])
                sale_amount -= used_amount
                asset['Remaining Amount'] -= used_amount

                cost_basis_for_this_amount = used_amount * asset['Cost Basis per Unit']
                gain_loss = (used_amount * sale_price) - cost_basis_for_this_amount
                gains_losses.append({
                    'Transaction ID': asset['Transaction ID'],
                    'Date Sold': sale_date,
                    'Date Acquired': asset['Date Acquired'],  # Ensure this line is added
                    'Used Amount': used_amount,
                    'Cost Basis': cost_basis_for_this_amount,
                    'Sale Proceeds': used_amount * sale_price,
                    'Gain/Loss': gain_loss,
                    'Source': transaction['Source']
                })


                if asset['Remaining Amount'] > 0:
                    fifo_queue.insert(0, asset)

            gainloss_ledger.extend(gains_losses)

        elif transaction['Category'] in ['Buys', 'Receives']:
            fifo_queue.append({
                'Transaction ID': transaction['Transaction ID'],
                'Date Acquired': transaction['Date'],
                'Remaining Amount': transaction['Asset Amount'],
                'Cost Basis per Unit': transaction['Asset Price'],
                'Category': transaction['Category'],
                'Source': transaction['Source']
            })
            
            if transaction['Category'] == 'Receives':
                # Calculate the fair market value of the received asset
                income = transaction['Asset Amount'] * transaction['Asset Price']
                tax_ledger.append({
                    'Date': transaction['Date'],
                    'Type': 'Income',
                    'Category': 'Bitcoin',
                    'Amount': income,
                    'Source': transaction['Source']
                })

    return pd.DataFrame(gainloss_ledger), pd.DataFrame(tax_ledger), fifo_queue

gainloss_ledger_df, tax_ledger_df, fifo_queue_df = assign_gain_loss(unified_ledger)

def calculate_holding_period(df):
    """
    Prepares the gain/loss ledger DataFrame by adding 'holding_period'.
    
    Parameters:
    - df: DataFrame containing gain and loss ledger information with 'Date Sold' and 'Date Acquired' columns.
    
    Returns:
    - DataFrame with added 'holding_period'.
    """
    # Ensure 'Date Sold' and 'Date Acquired' are in datetime format
    df['Date Sold'] = pd.to_datetime(df['Date Sold'])
    df['Date Acquired'] = pd.to_datetime(df['Date Acquired'])
    
    # Calculate 'holding_period'
    df['holding_period'] = df.apply(
        lambda row: 'Long-term' if (row['Date Sold'] - row['Date Acquired']).days > 365 else 'Short-term', axis=1)
    
    return df

calculate_holding_period(gainloss_ledger_df)

# Quick overview of the ledgers
print("Gain/Loss Ledger Overview:")
print(gainloss_ledger_df.head())
print("\nTotal Gain/Loss Entries:", gainloss_ledger_df.shape[0])

print("\nTax Ledger Overview:")
print(tax_ledger_df.head())
print("\nTotal Tax Entries:", tax_ledger_df.shape[0])

# Check for any anomalies, like negative remaining amounts
print("\nAny Negative Remaining Amounts in FIFO Queue?", any(asset['Remaining Amount'] < 0 for asset in fifo_queue_df))


# Gain/Loss Distribution
plt.figure(figsize=(10, 6))
sns.histplot(gainloss_ledger_df['Gain/Loss'], kde=True, bins=30, color='blue')
plt.title('Distribution of Gain/Loss')
plt.xlabel('Gain/Loss')
plt.ylabel('Frequency')
plt.show()

# Income Over Time
plt.figure(figsize=(10, 6))
tax_ledger_df[tax_ledger_df['Type'] == 'Income'].groupby(tax_ledger_df['Date'].dt.to_period('M'))['Amount'].sum().plot(kind='bar')
plt.title('Monthly Income from Receives Transactions')
plt.xlabel('Month')
plt.ylabel('Total Income')
plt.xticks(rotation=45)
plt.show()

# Only select numeric columns
numeric_cols_gainloss = gainloss_ledger_df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_cols_gainloss.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Gain/Loss Ledger')
plt.show()


# Filter the DataFrame for gains only
gains_df = gainloss_ledger_df[gainloss_ledger_df['Gain/Loss'] > 0]

# Plot comparison of long-term vs short-term gains
plt.figure(figsize=(10, 6))
sns.boxplot(x='holding_period', y='Gain/Loss', data=gains_df)
plt.title('Comparison of Long-term vs Short-term Gains')
plt.xlabel('Holding Period')
plt.ylabel('Gain Amount')
plt.show()

# Convert 'Date Sold' to datetime format if not already
gainloss_ledger_df['Date Sold'] = pd.to_datetime(gainloss_ledger_df['Date Sold'])

# Extract the year and month for grouping
gainloss_ledger_df['Year-Month'] = gainloss_ledger_df['Date Sold'].dt.to_period('M')

# Group by 'Year-Month' and 'holding_period', then sum the 'Gain/Loss'
monthly_gains_losses = gainloss_ledger_df.groupby(['Year-Month', 'holding_period'])['Gain/Loss'].sum().unstack().fillna(0)

# Plotting the monthly gains and losses
monthly_gains_losses.plot(kind='bar', figsize=(14, 8), width=0.8)
plt.title('Monthly Gains and Losses by Holding Period')
plt.xlabel('Month')
plt.ylabel('Total Gain/Loss')
plt.xticks(rotation=45)
plt.legend(title='Holding Period')
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(unified_ledger['Date'], unified_ledger['Amount'], marker='o', linestyle='', alpha=0.5)
plt.title('Transaction Amounts Over Time')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

unified_ledger['Category'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
plt.title('Distribution of Transaction Categories')
plt.ylabel('')  # Hide the y-label for clarity
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Amount', data=unified_ledger)
plt.title('Transaction Amounts by Category')
plt.xlabel('Category')
plt.ylabel('Amount')
plt.xticks(rotation=45)
plt.show()

# Assuming 'holding_period' is calculated in days as a numeric value
plt.figure(figsize=(10, 6))
sns.histplot(data=gainloss_ledger_df, x='holding_period', bins=30, kde=False)
plt.title('Distribution of Holding Periods')
plt.xlabel('Holding Period (Days)')
plt.ylabel('Count')
plt.show()

"""# Assuming gainloss_ledger_df for this visualization
plt.figure(figsize=(10, 8))
sns.heatmap(gainloss_ledger_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numeric Variables')
plt.show()"""
"""
plt.figure(figsize=(10, 6))
sns.scatterplot(data=gainloss_ledger_df, x='Sale Proceeds', y='Gain/Loss', hue='holding_period')
plt.title('Sale Proceeds vs. Gain/Loss by Holding Period')
plt.xlabel('Sale Proceeds')
plt.ylabel('Gain/Loss')
plt.legend(title='Holding Period')
plt.show()
"""
def calc_8949_holding_period(df):
    """
    Prepares the 8949 DataFrame by adding 'Holding Period'.
    
    Parameters:
    - df: DataFrame containing 8949 information with 'Date Sold or Disposed' and 'Date Acquired' columns.
    
    Returns:
    - DataFrame with added 'Holding Period'.
    """
    # Ensure 'Date Sold' and 'Date Acquired' are in datetime format
    df['Date Sold or Disposed'] = pd.to_datetime(df['Date Sold or Disposed'])
    df['Date Acquired'] = pd.to_datetime(df['Date Acquired'])
    
    # Calculate 'holding_period'
    df['Holding Period'] = df.apply(
        lambda row: 'Long-term' if (row['Date Sold or Disposed'] - row['Date Acquired']).days > 365 else 'Short-term', axis=1)
    
    return df


def prepare_form_8949(gainloss_ledger_df, tax_ledger_df):
    # Rename and select columns for the gain/loss transactions
    form_8949_sales = gainloss_ledger_df.rename(columns={
        'Date Acquired': 'Date Acquired',
        'Date Sold': 'Date Sold or Disposed',
        'Sale Proceeds': 'Proceeds',
        'Cost Basis': 'Cost or Other Basis',
        'Gain/Loss': 'Gain or Loss'
    })[['Date Acquired', 'Date Sold or Disposed', 'Proceeds', 'Cost or Other Basis', 'Gain or Loss']]

    # Add description column (assuming the property is "Bitcoin" for this example)
    form_8949_sales['Description of Property'] = 'Bitcoin'
    form_8949_sales['Source'] = gainloss_ledger_df['Source']

    # Prepare the receive transactions treated as income
    # This assumes tax_ledger_df includes Date, Type (as "Income"), and Amount columns
    form_8949_income = tax_ledger_df.rename(columns={
        'Date': 'Date Acquired',
        'Amount': 'Proceeds'
    })[['Date Acquired', 'Proceeds']]
    form_8949_income['Date Sold or Disposed'] = form_8949_income['Date Acquired']  # Same date for acquired and sold
    form_8949_income['Cost or Other Basis'] = form_8949_income['Proceeds']  # FMV at time of receipt
    form_8949_income['Gain or Loss'] = 0  # No gain/loss at the point of receipt
    form_8949_income['Description of Property'] = 'Bitcoin (Received)'
    form_8949_income['Source'] = tax_ledger_df['Source']

    # Combine both DataFrames for full 8949 reporting
    full_form_8949 = pd.concat([form_8949_sales, form_8949_income], ignore_index=True, sort=False).fillna(0)

    # Sort by sale date for chronological order
    full_form_8949 = full_form_8949.sort_values(by='Date Sold or Disposed')

    # Add holding period information
    calc_8949_holding_period(full_form_8949)
    return full_form_8949

# Call the function with your DataFrames
form_8949_df = prepare_form_8949(gainloss_ledger_df, tax_ledger_df)

# Show the prepared Form 8949 DataFrame
print(form_8949_df)

row_count = len(form_8949_df)

#8949 diagnostics:


sns.histplot(data=form_8949_df, x='Gain or Loss', kde=True, hue='Holding Period', multiple='stack')
plt.title('Distribution of Gains and Losses on Form 8949')
plt.xlabel('Gain/Loss ($)')
plt.ylabel('Count')
plt.legend(title='Holding Period')
plt.show()

form_8949_df['Holding Period'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Holding Periods in Form 8949')
plt.ylabel('')  # Hide y-label for clarity
plt.show()

form_8949_df['Month'] = form_8949_df['Date Sold or Disposed'].dt.to_period('M')
monthly_gains_losses = form_8949_df.groupby(['Month', 'Holding Period'])['Gain or Loss'].sum().unstack()

monthly_gains_losses.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Monthly Gains and Losses from Form 8949')
plt.xlabel('Month')
plt.ylabel('Gain/Loss ($)')
plt.xticks(rotation=45)
plt.legend(title='Holding Period')
plt.show()

sns.scatterplot(data=form_8949_df, x='Proceeds', y='Gain or Loss', hue='Holding Period')
plt.title('Sale Proceeds vs. Gain/Loss in Form 8949')
plt.xlabel('Sale Proceeds ($)')
plt.ylabel('Gain/Loss ($)')
plt.legend(title='Holding Period')
plt.show()

form_8949_df.sort_values('Date Sold or Disposed', inplace=True)
form_8949_df['Cumulative Gain/Loss'] = form_8949_df['Gain or Loss'].cumsum()

plt.plot(form_8949_df['Date Sold or Disposed'], form_8949_df['Cumulative Gain/Loss'])
plt.title('Cumulative Gain/Loss from Form 8949')
plt.xlabel('Date')
plt.ylabel('Cumulative Gain/Loss ($)')
plt.xticks(rotation=45)
plt.show()


print(unified_ledger.head())
print(gainloss_ledger_df.head())
print(tax_ledger_df.head())
print(form_8949_df.head())

def prepare_taxslayer_csv(form_8949_df):
    # Map your form 8949 DataFrame to the TaxSlayer template structure
    taxslayer_master_df = form_8949_df.rename(columns={
        'Date Acquired': 'DtAcquired',
        'Date Sold or Disposed': 'DtSold',
        'Proceeds': 'SalesPrice',
        'Cost or Other Basis': 'Cost',
        'Description of Property': 'Description'
    })
    taxslayer_master_df['Owner'] = 'T'  # 'T' for taxpayer

    # Select the necessary columns as per the TaxSlayer template
    taxslayer_master_df = taxslayer_master_df[[
        'Owner', 'Description', 'DtAcquired', 'DtSold', 'SalesPrice', 'Cost', 'Source'
    ]]

    # Now split the DataFrame into separate ones based on the 'Source' column
    cash_app_21_df = taxslayer_master_df[taxslayer_master_df['Source'] == 'cash_app_21'].drop('Source', axis=1)
    cash_app_22_df = taxslayer_master_df[taxslayer_master_df['Source'] == 'cash_app_22'].drop('Source', axis=1)
    fold_app_22_df = taxslayer_master_df[taxslayer_master_df['Source'] == 'fold_app_22'].drop('Source', axis=1)
    swan_app_22_df = taxslayer_master_df[taxslayer_master_df['Source'] == 'swan_app_22'].drop('Source', axis=1)

    # Return the master DataFrame and the individual source DataFrames
    return taxslayer_master_df, cash_app_21_df, cash_app_22_df, fold_app_22_df, swan_app_22_df

# Assuming form_8949_df is your DataFrame with all the Form 8949 information
taxslayer_master_df, cash_app_21_df, cash_app_22_df, fold_app_22_df, swan_app_22_df = prepare_taxslayer_csv(form_8949_df)

# Now, you can save each DataFrame to a CSV file
taxslayer_master_df.to_csv('taxslayer_master.csv', index=False)
cash_app_21_df.to_csv('cash_app_21.csv', index=False)
cash_app_22_df.to_csv('cash_app_22.csv', index=False)
fold_app_22_df.to_csv('fold_app_22.csv', index=False)
swan_app_22_df.to_csv('swan_app_22.csv', index=False)


# Pie chart for the distribution of transactions across different sources
taxslayer_master_df['Source'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Transactions by Source')
plt.ylabel('')  # Hide y-label for clarity
plt.show()

# Histogram for the distribution of sales prices within the 2021 Cash App source
cash_app_21_df['SalesPrice'].plot(kind='hist', bins=30, alpha=0.7)
plt.title('Distribution of Sales Prices for Cash App 2021')
plt.xlabel('Sales Price')
plt.ylabel('Frequency')
plt.show()


cash_app_21_gl = cash_app_21_df['SalesPrice'] - cash_app_21_df['Cost']
# Cumulative gain/loss over time for the 2021 Cash App source
cash_app_21_df['Cumulative Gain/Loss'] = cash_app_21_gl.cumsum()
cash_app_21_df.sort_values('DtSold', inplace=True)
plt.plot(cash_app_21_df['DtSold'], cash_app_21_df['Cumulative Gain/Loss'])
plt.title('Cumulative Gain/Loss for Cash App 2021')
plt.xlabel('Date')
plt.ylabel('Cumulative Gain/Loss')
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to fit the x-axis labels
plt.show()

plot_source_distribution(unified_ledger, 'Unified Ledger')
plot_source_distribution(gainloss_ledger_df, 'Gain/Loss')
plot_source_distribution(tax_ledger_df, 'Tax/Receive')
plot_source_distribution(form_8949_df, 'Form 8949')
plot_source_distribution(taxslayer_master_df, 'TaxSlayer Master')

