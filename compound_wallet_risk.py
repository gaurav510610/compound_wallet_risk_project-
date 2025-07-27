


import pandas as pd
import requests
import time 





## api configuration 
API_KEY ="cqt_rQGXq6fwCBDXV4ckvD7mP4HPmk7q"
base_url = "https://api.covalenthq.com/v1"





## load wallets
df=pd.read_csv("Wallet_id .csv")
wallets =df.iloc[:, 0].tolist()  
print(f"Loaded {len(wallets)} wallet addresses.")





###  fetch transaction from compound v2 (using covalent api)
## created function to fetch wallet transactions from compound 
def fetch_wallet_transactions(wallet_address,chain_id=1):
    url=f"{base_url}/{chain_id}/address/{wallet_address}/transactions_v2/"
    params={
        "key":API_KEY
    }
    response=requests.get(url, params=params)
    if response.status_code == 200:
        data=response.json()
        items = data.get('data', {}).get('items', [])
        if not items:
            print(f"No transactions found for {wallet_address} ")
            return pd.DataFrame()
        return pd.DataFrame(items)
    else:
        
        print(f"Error {response.status_code} for wallet: {wallet_address}")
        return pd.DataFrame()





## fetch all transactions for all wallets
all_transactions = pd.DataFrame()
for wallet in wallets:
    print(f"Fetching data for: {wallet}")
    tx_df = fetch_wallet_transactions(wallet)
    if not tx_df.empty:
        tx_df['wallet_id'] = wallet
        all_transactions = pd.concat([all_transactions, tx_df], ignore_index=True)
    time.sleep(0.5)  
 ## save to csv    
all_transactions.to_csv("wallet_transactions_compound.csv", index=False)


# ### Data Preparation  and feature engineering 




## loading transactional data that i fetched before 
df2=pd.read_csv("wallet_transactions_compound.csv")
df2.head()





print("Total unique wallets:", df2['wallet_id'].nunique())





df2.info()





## converting block_signed_at datetime 
df2['block_signed_at'] = pd.to_datetime(df2['block_signed_at'])





## aggregate useful feratures per wallet 
wallet_features=df2.groupby("wallet_id").agg(
    total_transactions=("tx_hash", "count"),
    total_gas_spent=("gas_spent", "sum"),
    avg_gas_price=("gas_price", "mean"),
    total_fees_paid=("fees_paid", "sum"),
    unique_destinations=("to_address", "nunique"),
    total_successful_transaction=("successful", "sum"),
    active_days=("block_signed_at", "nunique")
).reset_index()
wallet_features.head()





## replace 0 to avoid division errors
wallet_features['active_days'] = wallet_features['active_days'].replace(0, 1)
wallet_features['total_transactions'] = wallet_features['total_transactions'].replace(0, 1)
##deriving custom  features for wallet risk 
wallet_features['avg_gas_spent_per_tx'] = wallet_features['total_gas_spent'] / wallet_features['total_transactions']
wallet_features['avg_tx_per_day'] = wallet_features['total_transactions'] / wallet_features['active_days']
wallet_features['fees_paid_per_tx'] = wallet_features['total_fees_paid'] / wallet_features['total_transactions']
wallet_features['success_ratio'] = wallet_features['total_successful_transaction'] / wallet_features['total_transactions']
wallet_features['destination_diversity'] = wallet_features['unique_destinations'] / wallet_features['total_transactions']
wallet_features['avg_fee_per_day'] = wallet_features['total_fees_paid'] / wallet_features['active_days']





###  checking final feature 
wallet_features[['wallet_id','avg_gas_spent_per_tx','avg_tx_per_day','fees_paid_per_tx',
                 'success_ratio','destination_diversity','avg_fee_per_day' ]].head()
               





import os
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt





##  select relavant features  for modeling 
model_features = wallet_features[[
    "avg_tx_per_day",
    "avg_gas_spent_per_tx",
    "success_ratio",
    "fees_paid_per_tx",
    "destination_diversity",
    "avg_fee_per_day",
    "total_transactions",
    "total_gas_spent",
    "avg_gas_price",
    "total_fees_paid",
    "unique_destinations",
    "total_successful_transaction",
    "active_days"
]]

## normalizing features 
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(model_features)
## kmeans clustering 
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
wallet_features["cluster"] = kmeans.fit_predict(scaled_features)
## score  clisuter using summary metrics
cluster_summary = wallet_features.groupby("cluster").agg({
    "avg_tx_per_day": "mean",
    "success_ratio": "mean",
    "fees_paid_per_tx": "mean",
    "destination_diversity": "mean",
    "avg_fee_per_day": "mean"
})
cluster_summary["score_metric"] = (
    cluster_summary["success_ratio"] +
    cluster_summary["fees_paid_per_tx"] +
    cluster_summary["avg_fee_per_day"] -
    cluster_summary["destination_diversity"]
).round(2)
# Rank clusters and assign scores
cluster_summary["rank"] = cluster_summary["score_metric"].rank(ascending=False, method='min').astype(int)
rank_to_score = {1: 1000, 2: 750, 3: 500, 4: 250, 5: 100}
cluster_to_rank = cluster_summary["rank"].to_dict()
wallet_features["score"] = wallet_features["cluster"].map(lambda c: rank_to_score[cluster_to_rank[c]])
# Export final scores
wallet_scores = wallet_features[["wallet_id", "score"]]
wallet_scores.to_csv("wallet_risk_scores.csv", index=False)
print(wallet_scores.head())





print(wallet_features[["cluster", "score"]].value_counts().sort_index())
print(wallet_features["score"].value_counts().sort_index())





# Plotting Risk Score Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=wallet_scores, x='score', hue='score', palette='viridis', 
              order=sorted(wallet_scores['score'].unique(), reverse=True))
plt.title('Risk Score Distribution of Wallets', fontsize=14)
plt.xlabel('Risk Score', fontsize=12)
plt.ylabel('Number of Wallets', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend([],[], frameon=False)  
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("risk score distribution of wallets.png", dpi=300)
plt.show()





merged = wallet_features.merge(
    wallet_scores,   
    how="left"
)

bucket_means = (merged.groupby("score")["avg_tx_per_day"].mean().sort_index())
plt.figure(figsize=(8, 4))
plt.bar(bucket_means.index.astype(str), bucket_means.values, color="C0")
plt.title("Average Transactions per Day by Risk Score")
plt.xlabel("Risk Score")
plt.ylabel("Avg. Tx per Day")
plt.tight_layout()  
plt.savefig("avg_tx_per_day_by_scor.png", dpi=300)
plt.show()





print(bucket_means)







