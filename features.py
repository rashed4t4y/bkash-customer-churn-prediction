import os
import glob
import pandas as pd
import numpy as np

# Path configurations
BASE_PATH = r"d:\nsu contest\bkash-presents-nsucec-datathon\public"

def load_base_data(sample_ratio=0.1, random_state=42):
    """
    Phase 1: Ingestion and Sampling
    Loads train labels, test set, and KYC data.
    Implements a reproducible stratified sampling strategy on customer IDs.
    """
    print("Loading train, test, and KYC data...")
    train_labels = pd.read_csv(os.path.join(BASE_PATH, "train_labels.csv"))
    test_df = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
    kyc_df = pd.read_parquet(os.path.join(BASE_PATH, "kyc.parquet"))
    
    # Filter KYC to Customer accounts only
    kyc_df = kyc_df[kyc_df["ACCOUNT_TYPE"] == "Customer"]
    
    # Create the list of target customers
    train_custs = train_labels["ACCOUNT_ID"].unique()
    test_custs = test_df["ACCOUNT_ID"].unique()
    
    if sample_ratio < 1.0:
        print(f"Applying sampling strategy: {sample_ratio*100}% sample size.")
        # Reproducible stratified sampling on train set (based on Churn label)
        pos_sampled = train_labels[train_labels["CHURN"] == 1].sample(frac=sample_ratio, random_state=random_state)
        neg_sampled = train_labels[train_labels["CHURN"] == 0].sample(frac=sample_ratio, random_state=random_state)
        train_sampled = pd.concat([pos_sampled, neg_sampled]).sample(frac=1.0, random_state=random_state)
        
        # Random sampling on test set
        test_sampled = test_df.sample(frac=sample_ratio, random_state=random_state)
        
        active_train_custs = set(train_sampled["ACCOUNT_ID"])
        active_test_custs = set(test_sampled["ACCOUNT_ID"])
        
        train_labels_active = train_sampled
        test_df_active = test_sampled
    else:
        active_train_custs = set(train_custs)
        active_test_custs = set(test_custs)
        train_labels_active = train_labels
        test_df_active = test_df
        
    active_custs = active_train_custs.union(active_test_custs)
    print(f"Total active customers in analysis: {len(active_custs)} (Train: {len(active_train_custs)}, Test: {len(active_test_custs)})")
    
    return train_labels_active, test_df_active, kyc_df, active_custs

def load_transactions_and_balances(active_custs):
    """
    Loads transaction and balance files, filtering for target customer accounts.
    """
    # Load transactions
    print("Loading transaction parquet files...")
    trx_files = glob.glob(os.path.join(BASE_PATH, "transactions", "*.parquet"))
    trx_parts = []
    for f in trx_files:
        print(f" Reading {os.path.basename(f)}...")
        part = pd.read_parquet(f)
        # Column pruning and row filtering
        part = part[["TrxID", "TRX_DATETIME", "SRC_ACCOUNT", "DST_ACCOUNT", "TRX_TYPE", "TRX_AMT"]]
        part = part[part["SRC_ACCOUNT"].isin(active_custs) | part["DST_ACCOUNT"].isin(active_custs)]
        trx_parts.append(part)
    trx_df = pd.concat(trx_parts, ignore_index=True)
    trx_df["TRX_DATETIME"] = pd.to_datetime(trx_df["TRX_DATETIME"])
    print(f"Loaded {len(trx_df)} relevant transaction rows.")
    
    # Load balances
    print("Loading daily balance parquet files...")
    bal_files = glob.glob(os.path.join(BASE_PATH, "dayend_balance", "*.parquet"))
    bal_parts = []
    for f in bal_files:
        print(f" Reading {os.path.basename(f)}...")
        part = pd.read_parquet(f)
        part = part[["ACCOUNT_ID", "DATE", "AVAILABLE_BALANCE"]]
        part = part[part["ACCOUNT_ID"].isin(active_custs)]
        bal_parts.append(part)
    bal_df = pd.concat(bal_parts, ignore_index=True)
    bal_df["DATE"] = pd.to_datetime(bal_df["DATE"])
    print(f"Loaded {len(bal_df)} relevant balance rows.")
    
    return trx_df, bal_df

def extract_features(kyc_df, trx_df, bal_df, customer_ids):
    """
    Phase 2: Feature Engineering
    Extracts 15 behavioral features for the specified customer IDs.
    """
    print("Extracting features for customers...")
    cust_set = set(customer_ids)
    
    # 1. Base KYC features
    kyc_active = kyc_df[kyc_df["ACCOUNT_ID"].isin(cust_set)].copy()
    ref_date = pd.to_datetime("2024-03-31")
    
    # Feature 1: Tenure
    kyc_active["tenure_days"] = (ref_date - pd.to_datetime(kyc_active["ACCOUNT_OPEN_DATE"])).dt.days
    
    # Feature 2: Gender encoding
    kyc_active["gender_encoded"] = kyc_active["GENDER"].map({"Male": 1, "Female": 0}).fillna(-1)
    
    # Feature 3: Region mapping (label encoding)
    regions = sorted(kyc_active["REGION"].dropna().unique())
    region_map = {r: idx for idx, r in enumerate(regions)}
    kyc_active["region_encoded"] = kyc_active["REGION"].map(region_map).fillna(-1)
    
    # Prepare features dataframe
    features_df = kyc_active[["ACCOUNT_ID", "tenure_days", "gender_encoded", "region_encoded"]].copy()
    features_df.set_index("ACCOUNT_ID", inplace=True)
    
    # 2. Transaction outgoing features
    trx_out = trx_df[trx_df["SRC_ACCOUNT"].isin(cust_set)].copy()
    trx_out_30d = trx_out[trx_out["TRX_DATETIME"] >= "2024-03-02"]
    
    # Outgoing aggregation (90d)
    out_agg_90 = trx_out.groupby("SRC_ACCOUNT").agg(
        total_amt_90d=("TRX_AMT", "sum"),
        trx_count_90d=("TrxID", "count"),
        max_date_out=("TRX_DATETIME", "max")
    )
    
    # Outgoing aggregation (30d)
    out_agg_30 = trx_out_30d.groupby("SRC_ACCOUNT").agg(
        trx_count_30d=("TrxID", "count"),
        mean_amt_30d=("TRX_AMT", "mean")
    )
    
    # Feature 4: Recency
    out_agg_90["recency_days"] = (ref_date - out_agg_90["max_date_out"]).dt.days
    
    # 3. Transaction incoming features
    trx_in = trx_df[trx_df["DST_ACCOUNT"].isin(cust_set)].copy()
    in_agg_90 = trx_in.groupby("DST_ACCOUNT").agg(
        total_amt_in_90d=("TRX_AMT", "sum")
    )
    
    # 4. Type Ratios (90d outgoing)
    type_amt_90 = trx_out.groupby(["SRC_ACCOUNT", "TRX_TYPE"])["TRX_AMT"].sum().unstack(fill_value=0.0)
    
    # 5. Weekend transaction count (Friday & Saturday in Bangladesh)
    # Friday=4, Saturday=5
    trx_out["is_weekend"] = trx_out["TRX_DATETIME"].dt.weekday.isin([4, 5])
    weekend_counts = trx_out.groupby("SRC_ACCOUNT")["is_weekend"].sum()
    
    # 6. Service diversity (Outgoing transaction types)
    service_diversity = trx_out.groupby("SRC_ACCOUNT")["TRX_TYPE"].nunique()
    
    # 7. Balance features
    bal_active = bal_df[bal_df["ACCOUNT_ID"].isin(cust_set)].copy()
    
    # Last balance (closest to 2024-03-31)
    last_bal_df = bal_active.sort_values("DATE").groupby("ACCOUNT_ID").last()
    
    # Mean and Std of balance (90d)
    bal_stats = bal_active.groupby("ACCOUNT_ID").agg(
        mean_balance_90d=("AVAILABLE_BALANCE", "mean"),
        balance_std_90d=("AVAILABLE_BALANCE", "std")
    )
    
    # Balance Trend: Mean last 30d vs Mean first 30d
    bal_last_30 = bal_active[bal_active["DATE"] >= "2024-03-02"].groupby("ACCOUNT_ID")["AVAILABLE_BALANCE"].mean()
    bal_first_30 = bal_active[bal_active["DATE"] <= "2024-01-30"].groupby("ACCOUNT_ID")["AVAILABLE_BALANCE"].mean()
    
    # Zero Balance Days (Available balance <= 0)
    bal_active["is_zero"] = bal_active["AVAILABLE_BALANCE"] <= 0
    zero_days = bal_active.groupby("ACCOUNT_ID")["is_zero"].sum()
    
    # Combine all engineered features
    features_df = features_df.join(out_agg_90[["total_amt_90d", "trx_count_90d", "recency_days"]], how="left")
    features_df = features_df.join(out_agg_30[["trx_count_30d", "mean_amt_30d"]], how="left")
    features_df = features_df.join(in_agg_90["total_amt_in_90d"], how="left")
    
    # Fill transaction frequency and monetary defaults
    features_df["total_amt_90d"] = features_df["total_amt_90d"].fillna(0.0)
    features_df["trx_count_90d"] = features_df["trx_count_90d"].fillna(0.0)
    features_df["recency_days"] = features_df["recency_days"].fillna(90.0) # Maximum window length
    features_df["trx_count_30d"] = features_df["trx_count_30d"].fillna(0.0)
    features_df["mean_amt_30d"] = features_df["mean_amt_30d"].fillna(0.0)
    features_df["total_amt_in_90d"] = features_df["total_amt_in_90d"].fillna(0.0)
    
    # Service diversity & weekend ratios
    features_df = features_df.join(weekend_counts.to_frame("weekend_trx_count"), how="left")
    features_df["weekend_trx_count"] = features_df["weekend_trx_count"].fillna(0.0)
    features_df["weekend_trx_ratio"] = (features_df["weekend_trx_count"] / (features_df["trx_count_90d"] + 1e-5)).fillna(0.0)
    features_df.drop(columns=["weekend_trx_count"], inplace=True)
    
    features_df = features_df.join(service_diversity.to_frame("service_diversity"), how="left")
    features_df["service_diversity"] = features_df["service_diversity"].fillna(0)
    
    # Type Ratios
    for col in ["P2P", "MerchantPay", "BillPay", "CashOut"]:
        ratio_col = f"{col.lower()}_ratio"
        if col in type_amt_90.columns:
            features_df = features_df.join(type_amt_90[col].to_frame(col), how="left")
            features_df[col] = features_df[col].fillna(0.0)
            features_df[ratio_col] = (features_df[col] / (features_df["total_amt_90d"] + 1e-5)).fillna(0.0)
            features_df.drop(columns=[col], inplace=True)
        else:
            features_df[ratio_col] = 0.0
            
    # Balance features join
    features_df = features_df.join(last_bal_df["AVAILABLE_BALANCE"].to_frame("last_balance"), how="left")
    features_df["last_balance"] = features_df["last_balance"].fillna(0.0)
    
    features_df = features_df.join(bal_stats, how="left")
    features_df["mean_balance_90d"] = features_df["mean_balance_90d"].fillna(0.0)
    features_df["balance_std_90d"] = features_df["balance_std_90d"].fillna(0.0)
    
    # Balance Trend
    features_df = features_df.join(bal_last_30.to_frame("mean_bal_last"), how="left")
    features_df = features_df.join(bal_first_30.to_frame("mean_bal_first"), how="left")
    features_df["mean_bal_last"] = features_df["mean_bal_last"].fillna(0.0)
    features_df["mean_bal_first"] = features_df["mean_bal_first"].fillna(0.0)
    features_df["balance_trend"] = (features_df["mean_bal_last"] + 1.0) / (features_df["mean_bal_first"] + 1.0)
    features_df.drop(columns=["mean_bal_last", "mean_bal_first"], inplace=True)
    
    # Zero balance days
    features_df = features_df.join(zero_days.to_frame("zero_balance_days"), how="left")
    features_df["zero_balance_days"] = features_df["zero_balance_days"].fillna(0)
    
    # Velocity
    features_df["velocity_freq_30_90"] = features_df["trx_count_30d"] / (features_df["trx_count_90d"] + 1.0)
    
    # Ensure there are no NaNs in final dataframe
    features_df = features_df.fillna(0.0)
    
    return features_df

def apply_feature_quality(df, is_train=True, train_stats=None):
    """
    Phase 3: Feature Quality & Distribution adjustments.
    Applies Winsorization (clipping) and log1p transformations for skewed features.
    Adds indicator flags for zero-inflated features (>20% zeros).
    """
    df_transformed = df.copy()
    
    # List of numeric features to audit for skewness
    skewed_features = [
        "tenure_days", "total_amt_90d", "trx_count_90d", "recency_days",
        "trx_count_30d", "mean_amt_30d", "total_amt_in_90d",
        "last_balance", "mean_balance_90d", "balance_std_90d",
        "balance_trend", "zero_balance_days", "velocity_freq_30_90"
    ]
    
    stats = {} if train_stats is None else train_stats
    
    for feat in skewed_features:
        if feat in df_transformed.columns:
            # 1. Audit skewness on train
            if is_train:
                skew_val = df_transformed[feat].skew()
                zero_frac = (df_transformed[feat] == 0).mean()
                
                # Check for Winsorization bounds (99th percentile for positive skew)
                p99 = df_transformed[feat].quantile(0.99)
                p01 = df_transformed[feat].quantile(0.01)
                
                stats[feat] = {
                    "skew_raw": skew_val,
                    "zero_frac": zero_frac,
                    "p01": p01,
                    "p99": p99,
                    "apply_log1p": abs(skew_val) > 1.0,
                    "apply_zero_flag": zero_frac > 0.20
                }
            
            # Retrieve stats
            feat_stats = stats[feat]
            
            # Apply zero flag first
            if feat_stats["apply_zero_flag"]:
                df_transformed[f"{feat}_is_zero"] = (df_transformed[feat] == 0).astype(int)
            
            # Apply Winsorization (Clipping)
            df_transformed[feat] = np.clip(df_transformed[feat], feat_stats["p01"], feat_stats["p99"])
            
            # Apply log1p transformation if skew > 1
            if feat_stats["apply_log1p"]:
                # If negative values exist after clipping, shift before log1p
                min_val = df_transformed[feat].min()
                if min_val < 0:
                    df_transformed[feat] = np.log1p(df_transformed[feat] - min_val)
                else:
                    df_transformed[feat] = np.log1p(df_transformed[feat])
                    
                # Re-calculate skew after
                if is_train:
                    stats[feat]["skew_after"] = df_transformed[feat].skew()
    
    if is_train:
        return df_transformed, stats
    else:
        return df_transformed
