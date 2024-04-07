import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


# Ensure that dataframes is a dictionary
dataframes = {}

# Populate dataframes dictionary
for source, path in csv_paths.items():
    df = load_and_preprocess(path)
    df['Source'] = source
    dataframes[source] = df

# Convert dates to just the year for comparison purposes
dataframes['cash_app_22']['Year'] = dataframes['cash_app_22']['Date'].dt.year
dataframes['cash_app_21']['Year'] = dataframes['cash_app_21']['Date'].dt.year
dataframes['swan_app_22']['Year'] = dataframes['swan_app_22']['Date'].dt.year

# Identify CashApp 2022 transactions that are before 2022 and not in Transactions 2021
transactions_to_exclude_cashapp = dataframes['cash_app_22'][~((dataframes['cash_app_22']['Year'] >= 2022) | 
                                        (dataframes['cash_app_22']['Transaction ID'].isin(dataframes['cash_app_21']['Transaction ID'])))]

# Identify Swan 2022 transactions that are before 2022
transactions_to_exclude_swan = dataframes['swan_app_22'][~(dataframes['swan_app_22']['Year'] >= 2022)]

# Exclude these transactions
dataframes['cash_app_22'] = dataframes['cash_app_22'].drop(transactions_to_exclude_cashapp.index)
dataframes['swan_app_22'] = dataframes['swan_app_22'].drop(transactions_to_exclude_swan.index)

# Identify duplicates in cash_app_22 already present in cash_app_21
duplicates_in_cash_app_21 = dataframes['cash_app_21'][dataframes['cash_app_21']['Transaction ID'].isin(dataframes['cash_app_22']['Transaction ID'])]

# Remove duplicates from cash_app_22
dataframes['cash_app_22'] = dataframes['cash_app_22'][~dataframes['cash_app_22']['Transaction ID'].isin(duplicates_in_cash_app_21['Transaction ID'])]

# Verify removal
print("Duplicates removed from cash_app_22:")
print(dataframes['cash_app_22'].head())

# Update transactions_to_exclude after removing duplicates
transactions_to_exclude = dataframes['cash_app_22'][~((dataframes['cash_app_22']['Year'] >= 2022) | 
                                        (dataframes['cash_app_22']['Transaction ID'].isin(dataframes['cash_app_21']['Transaction ID'])))]

# Exclude these transactions
dataframes['cash_app_22'] = dataframes['cash_app_22'].drop(transactions_to_exclude.index)

# Verify exclusion
print("Excluded transactions from cash_app_22:")
print(transactions_to_exclude.head())

unified_ledger = pd.concat(dataframes, ignore_index=True)
unified_ledger = unified_ledger.sort_values(by='Date')

#remove NaN entry (deposit from wallet, accounted for in other wallets)
unified_ledger = unified_ledger[unified_ledger['Transaction ID'] != 'ukknp3']



##categorize transactions into 'buy' 'sale' and 'received' conditions
def categorize_transaction(row):
    # Explicitly check for transaction IDs to classify as 'Sales'
    if row['Transaction ID'] == 'a' or row['Transaction ID'] == 'b':
        return 'Sales'

    transaction_type = row['Transaction Type'].lower()
    if 'purchase' in transaction_type or 'buy' in transaction_type:
        return 'Buys'
    if 'sale' in transaction_type:
        #islhsi
        #ofqavt

        return 'Sales'
    if 'reward' in transaction_type or 'bonus' in transaction_type or 'boost' in transaction_type or 'p2p' in transaction_type:
        return 'Receives'
    # Add more conditions as necessary
    else:
        return 'Other'


unified_ledger['Category'] = unified_ledger.apply(categorize_transaction, axis=1)


print(unified_ledger[unified_ledger['Category'] == 'Other']['Notes'].unique())
print(unified_ledger[unified_ledger['Category'] == 'Other']['Notes'].value_counts())
print("cash app bitcoin:")
      

#diagnostics and visualizations:
print(unified_ledger.head())


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

def round_to_nearest_cent(df):
    """
    Round gain/loss and income from transactions to the nearest cent.
    
    Parameters:
    - df: DataFrame containing Form 8949 information.
    
    Returns:
    - DataFrame with gain/loss and income rounded to the nearest cent.
    """
    df['Gain or Loss'] = df['Gain or Loss'].round(2)
    df['Proceeds'] = df['Proceeds'].round(2)
    df['Cost or Other Basis'] = df['Cost or Other Basis'].round(2)  # Round cost portion

    return df

def round_to_nearest_dollar(df):
    """
    Round gain/loss and income from transactions to the nearest dollar.
    
    Parameters:
    - df: DataFrame containing Form 8949 information.
    
    Returns:
    - DataFrame with gain/loss and income rounded to the nearest dollar.
    """
    df['Gain or Loss'] = df['Gain or Loss'].round()
    df['Proceeds'] = df['Proceeds'].round()
    df['Cost or Other Basis'] = df['Cost or Other Basis'].round()  # Round cost portion
    return df

# Assuming 'form_8949_df' is your DataFrame containing the Form 8949 information

# Ensure the date columns are in datetime format (if not already)
form_8949_df['Date Acquired'] = pd.to_datetime(form_8949_df['Date Acquired'])
form_8949_df['Date Sold or Disposed'] = pd.to_datetime(form_8949_df['Date Sold or Disposed'])

# Format the date columns into 'MM/DD/YYYY'
form_8949_df['Date Acquired'] = form_8949_df['Date Acquired'].dt.strftime('%m/%d/%Y')
form_8949_df['Date Sold or Disposed'] = form_8949_df['Date Sold or Disposed'].dt.strftime('%m/%d/%Y')

# Display the DataFrame to verify the changes
print(form_8949_df[['Date Acquired', 'Date Sold or Disposed']].head())

# Apply rounding to the Form 8949 dataset
form_8949_df_rounded_cent = round_to_nearest_cent(form_8949_df.copy())
form_8949_df_rounded_dollar = round_to_nearest_dollar(form_8949_df.copy())



# Apply rounding to the Form 8949 dataset
### using 8949_rounded_cent
form_8949_df = form_8949_df_rounded_cent = round_to_nearest_cent(form_8949_df.copy())
form_8949_df_rounded_dollar = round_to_nearest_dollar(form_8949_df.copy())


print(unified_ledger.head())
print(gainloss_ledger_df.head())
print(tax_ledger_df.head())
print(form_8949_df.head())

print("cents")
print(form_8949_df_rounded_cent.head())

print("dollars")
print(form_8949_df_rounded_dollar.head())

# Assuming your DataFrame is named 'form_8949_df_rounded_cent'
cent_sample_transactions = form_8949_df_rounded_cent['Proceeds'].sample(n=10)  # Change 'n' to the number of transactions you want to sample

# Print the sampled transactions
print("Sampled Transactions:")
print(cent_sample_transactions)

# Assuming your DataFrame is named 'form_8949_df_rounded_cent'
dollar_sample_transactions = form_8949_df_rounded_dollar['Proceeds'].sample(n=10)  # Change 'n' to the number of transactions you want to sample

# Print the sampled transactions
print("Sampled Transactions:")
print(dollar_sample_transactions)


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
    cash_app_21_df = taxslayer_master_df[taxslayer_master_df['Source'] == 'cash_app_21']#.drop('Source', axis=1)
    cash_app_22_df = taxslayer_master_df[taxslayer_master_df['Source'] == 'cash_app_22']#.drop('Source', axis=1)
    fold_app_22_df = taxslayer_master_df[taxslayer_master_df['Source'] == 'fold_app_22']#.drop('Source', axis=1)
    swan_app_22_df = taxslayer_master_df[taxslayer_master_df['Source'] == 'swan_app_22']#.drop('Source', axis=1)

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

# Calculate the split index (if odd number of rows, the first part will have one less row)
split_index = len(cash_app_22_df) // 2

# Split the DataFrame into two parts
df_part_1 = cash_app_22_df.iloc[:split_index]
df_part_2 = cash_app_22_df.iloc[split_index:]

# Save the two parts to new CSV files
df_part_1.to_csv('~/cash_app_22_part_1.csv', index=False)
df_part_2.to_csv('~/cash_app_22_part_2.csv', index=False)

print(fifo_queue_df[0])
print(gainloss_ledger_df.iloc[-1])
print(tax_ledger_df.iloc[-1])
"""def save_fifo_queue_to_csv(fifo_queue, filename):

"""
"""
    Saves remaining transactions in the FIFO queue to a CSV file.
    
    Parameters:
    - fifo_queue: The FIFO queue containing the transactions.
    - filename: Name of the CSV file to save the data to.
    """
"""
    # Convert the FIFO queue to a DataFrame
    df_fifo = pd.DataFrame(fifo_queue)
    
    # Ensure the DataFrame is not empty
    if not df_fifo.empty:
        # Save to CSV, without the index
        df_fifo.to_csv(filename, index=False)
    else:
        print("The FIFO queue is empty. No CSV file was created.")

save_fifo_queue_to_csv(fifo_queue_df, "transactions22.csv")
"""

import pandas as pd

def filter_and_update_ledger(fifo_queue, unified_ledger, output_filename=None):
    """
    Filters the unified ledger to include only transactions present in the FIFO queue,
    updates the amount of the first matched transaction with the unspent amount,
    and optionally saves the result to a CSV file.
    
    Parameters:
    - fifo_queue: List of dictionaries, each representing a FIFO transaction.
    - unified_ledger: DataFrame containing the full set of transactions.
    - output_filename: Optional; name of the CSV file to save the filtered ledger.
    """
    # Extract Transaction IDs from FIFO queue
    fifo_transaction_ids = [transaction['Transaction ID'] for transaction in fifo_queue]

    # Filter the unified ledger
    filtered_ledger = unified_ledger[unified_ledger['Transaction ID'].isin(fifo_transaction_ids)].copy()

    if not filtered_ledger.empty:
        # Assuming the first transaction in the FIFO queue has the unspent amount
        unspent_amount = fifo_queue[0]['Remaining Amount']
        transaction_id_with_unspent = fifo_queue[0]['Transaction ID']

        # Find the index of the transaction in the filtered ledger
        index_to_update = filtered_ledger.index[filtered_ledger['Transaction ID'] == transaction_id_with_unspent][0]

        # Update the amount in the filtered ledger to reflect the unspent amount
        filtered_ledger.at[index_to_update, 'Amount'] = unspent_amount

        # Verify the updated transaction (optional)
        print("Updated Transaction:", filtered_ledger.loc[index_to_update])

        # Save to CSV if an output filename is provided
        if output_filename:
            filtered_ledger.to_csv(output_filename, index=False)
            print(f"Filtered ledger saved to {output_filename}.")
    else:
        print("No matching transactions found in the unified ledger.")

# Example usage
filter_and_update_ledger(fifo_queue_df, unified_ledger, "transactions22.csv")

"""
plot_source_distribution(unified_ledger, 'Unified Ledger')
plot_source_distribution(gainloss_ledger_df, 'Gain/Loss')
plot_source_distribution(tax_ledger_df, 'Tax/Receive')
plot_source_distribution(form_8949_df, 'Form 8949')
plot_source_distribution(taxslayer_master_df, 'TaxSlayer Master')
#<!---REMOVE SOURCE BEFORE TRANSMISSION--->#
plot_source_distribution(cash_app_21_df, 'CashApp 2021')
plot_source_distribution(cash_app_22_df, 'CashApp 2022')
plot_source_distribution(fold_app_22_df, 'FoldApp 2022')
plot_source_distribution(swan_app_22_df, 'SwanApp 2022')

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

#form_8949_df['Month'] = form_8949_df['Date Sold or Disposed'].dt.to_period('M')
#monthly_gains_losses = form_8949_df.groupby(['Month', 'Holding Period'])['Gain or Loss'].sum().unstack()

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
"""