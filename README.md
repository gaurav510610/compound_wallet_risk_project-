# Compound Protocol – Wallet Risk Scoring 

This project focuses on analyzing on-chain transaction behavior of wallet addresses interacting with the Compound V2 Protocol to assess their risk levels. The objective is to develop a risk scoring system from scratch that evaluates wallet activities and assigns a risk score between 0 and 1000.
## Objective
- Extract transaction data for provided wallet addresses.
- Fetching transaction data from the Compound V2 Protocol.
- Engineer meaningful features indicative of wallet risk.
- Develop a scalable scoring methodology.
- Deliver a final CSV file mapping wallet IDs to their respective risk scores.

## Tools & Technologies Used
- **Python (Pandas, NumPy,)**
- **Jupyter Notebook / Python Script**
- **Seaborn, Matplotlib**
- **Scikit-learn (KMeans, MinMaxScaler)**
## Folder Structure
**├── README.md
  ├── analysis.md
  ├── compound_wallet_risk.ipynb / .py
  ├── wallet_id.csv
  ├── wallet_transactions_compound.csv
  ├── wallet_risk_scores.csv
  └── figures/
    ├── risk_score_distribution OF WalletsS.png
    └── avg_tx_per_day_by_scor**
## How to Run the Code
- **1.** Clone this repository to your local machine.
- **2.** Open the compound_wallet_risk.ipynb in Jupyter Notebook or run the .py file.
- **3.**  Execute the script to generate wallet risk scores.
- **4.**  The output CSV wallet_risk_scores.csv will be created in the root folder.

## Data Collection Method
- **Wallet List:** The assignment provided a Google Sheet containing 103 wallet addresses.
- **Protocol**: Compound V2
- **API Used:** Covalent API (Chain ID: 1 - Ethereum Mainnet)
-  **Tool Used:** Python (`requests` library) to loop through wallet addresses and fetch transactional data
- **Approach**: For each wallet, we fetched historical on-chain transaction data using Covalent's /transactions_v2/ endpoint.
- **Fields Collected:** `tx_hash, block_signed_at, gas_spent, gas_price, fees_paid, to_address, successful` (and more).
- `Data was collected and saved locally to prevent repeated API calls when restarting the kernel.`
[Download Fetched_transactional_data  (Google Drive)](https://drive.google.com/file/d/1UdOQQMa9KnaXaYJceH4Q29HROtKcD7WJ/view?usp=drive_link)

# Feature Engineering
Per wallet, the following behavioral features were engineered:

| Feature                      | Description                                                   |
| ---------------------------- | ------------------------------------------------------------- |
| **avg\_tx\_per\_day**        | Average number of transactions per active day.                |
| **avg\_gas\_spent\_per\_tx** | Average gas spent per transaction.                            |
| **fees\_paid\_per\_tx**      | Average transaction fees paid per transaction.                |
| **success\_ratio**           | Ratio of successful transactions to total transactions.       |
| **destination\_diversity**   | Ratio of unique destination addresses per total transactions. |
| **avg\_fee\_per\_day**       | Average transaction fees paid per active day.                 |
| **total_transactions**       | Total number of transactions made.                            |
| **total_gas_spent**          | Total gas used across all transactions.                       |   
| **avg_gas_price**            | Average gas price used.                                       |
| **total_fees_paid**          |  Total fees paid across transactions.                         |
| **unique_destinations**      | Number of unique recipients.                                  |
|**total_successful_transaction** | Count of successful transactions.                          |
| **active_days**              | Unique days the wallet was active.                            |
## Feature Scaling
- **Method:** Min-Max Normalization.
- Applied to ensure that features with large values don't dominate clustering.

## Clustering & Scoring Method
- **Algorithm Used:** KMeans Clustering (n_clusters = 5).
- KMeans was applied on engineered features to group wallets by behavioral
- A score metric was computed as:

 `score_metric = success_ratio + fees_paid_per_tx + avg_fee_per_day - destination_diversity`
- Clusters were ranked based on this metric and assigned scores:
  - **Rank 1 → 1000**(most reliable cluster)
  - **Rank 2 → 750**
  - **Rank 3 → 500**
  - **Rank 4 → 250**
  - **Rank 5 → 100** (most risky cluster)
## Final Output
- Wallet-wise risk scores exported to **wallet_risk_scores.csv.**
- **Output**




