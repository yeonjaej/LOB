import pandas as pd
from huggingface_hub import hf_hub_download

# 1. Define the repository and the target files
repo_id = "totalorganfailure/lobster-data"
#subfolder = "LOBSTER_SampleFile_AAPL_2012-06-21_10"
#subfolder = "LOBSTER_SampleFile_AMZN_2012-06-21_10"
#subfolder = "LOBSTER_SampleFile_GOOG_2012-06-21_10"
#subfolder = "LOBSTER_SampleFile_INTC_2012-06-21_10"
subfolder = "LOBSTER_SampleFile_MSFT_2012-06-21_10"

#msg_filename = "AAPL_2012-06-21_34200000_57600000_message_10.csv"
#ob_filename = "AAPL_2012-06-21_34200000_57600000_orderbook_10.csv"
#msg_filename = "AMZN_2012-06-21_34200000_57600000_message_10.csv"
#ob_filename = "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
#msg_filename = "GOOG_2012-06-21_34200000_57600000_message_10.csv"
#ob_filename = "GOOG_2012-06-21_34200000_57600000_orderbook_10.csv"
#msg_filename = "INTC_2012-06-21_34200000_57600000_message_10.csv"
#ob_filename = "INTC_2012-06-21_34200000_57600000_orderbook_10.csv"
msg_filename = "MSFT_2012-06-21_34200000_57600000_message_10.csv"
ob_filename = "MSFT_2012-06-21_34200000_57600000_orderbook_10.csv"

# 2. Download the specific message and orderbook pair
print("Downloading files...")

msg_path = hf_hub_download(
    repo_id=repo_id,
    filename=f"{subfolder}/{msg_filename}",
    repo_type="dataset",
    local_dir=".",
)

ob_path = hf_hub_download(
    repo_id=repo_id,
    filename=f"{subfolder}/{ob_filename}",
    repo_type="dataset",
    local_dir=".",
)


# 3. Define the column names (LOBSTER files do not include headers)
msg_cols = ["Time", "Type", "OrderID", "Size", "Price", "Direction"]

# Dynamically generate orderbook columns for exactly 10 levels
depth = 10
ob_cols = []
for i in range(1, depth + 1):
    ob_cols.extend([f"AskPrice_{i}", f"AskSize_{i}", f"BidPrice_{i}", f"BidSize_{i}"])

# 4. Load the files into pandas DataFrames
print("Loading DataFrames...")
df_msg = pd.read_csv(msg_path, names=msg_cols)
df_ob = pd.read_csv(ob_path, names=ob_cols)

# 5. Scale the prices
# LOBSTER stores prices as integers scaled by 10,000 to avoid floating-point inaccuracies.
df_msg["Price"] = df_msg["Price"] / 10000.0

price_cols = [col for col in df_ob.columns if "Price" in col]
df_ob[price_cols] = df_ob[price_cols] / 10000.0

# 6. Verify the alignment
print("\nFirst Message Event:")
print(df_msg.iloc[0])

print("\nCorresponding Orderbook State (Top Level):")
print(df_ob[["AskPrice_1", "AskSize_1", "BidPrice_1", "BidSize_1"]].iloc[0])