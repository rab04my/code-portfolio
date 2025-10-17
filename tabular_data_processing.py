# Helper functions for general tabular data cleaning and processing for classification- and regression-based ML tasks.
# Last Modified: 07/15/2025

import pandas as pd
import pandas.api.types as ptypes
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def summarize_data(df, label):
    """
    Print a summary of the dataset including shape, data types,
    missing values (sorted), and unique counts for categorical variables.
    """
    print(f'Summary for {label} data.\n')
    print("=================================")
    print("Size:", df.shape)

    print("\n=================================")
    print("\nData types:\n", df.dtypes)

    print("\n=================================")
    print("\nMissing values (n):\n", df.isnull().sum())
    missing = (df.isnull().mean() * 100).round(2)
    missing = missing[missing > 0].sort_values(ascending=False)
    print("\nMissing values (%):\n", missing)
    
    print("\n=================================")
    print("\nUnique value counts (factors):\n")
    for col in df.select_dtypes(include='object').columns:
        print(f"{col}: {df[col].nunique()} unique values")


def cleanup_data(df):
    """
    Clean up the dataset by standardizing column names, removing duplicates,
    and replacing blank strings with NaN.
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
        .str.replace(r'[^\w]', '', regex=True)
    )
    df.drop_duplicates(inplace=True)
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    return df

def find_datetime_cols(df, threshold=0.9):
    """
    Detect columns that are likely datetime values, even if stored as strings/objects. 
    Really SLOW... need to optimize...
    
    threshold (numeric): proportion of rows that must parse to datetime
    """
    dt_cols = []

    for col in df.columns:
        dtype_str = str(df[col].dtype)

        if 'string[pyarrow]' in dtype_str:
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                success_ratio = parsed.notna().mean()
                if success_ratio >= threshold:
                    dt_cols.append(col)
            except Exception:
                continue

    return dt_cols

def impute_missing(df, perc_missing=0.5, cat_fill_value="unknown"):
    """
    Fill missing values--numeric columns with median and categorical with a placeholder string.
    Drops columns with more than `perc_missing` missing data.
    """
    df_filled = df.copy()
    
    missing_ratio = df_filled.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > perc_missing].index.tolist()
    if cols_to_drop:
        print(f"Dropping columns with {perc_missing * 100}% missing values: {cols_to_drop}")
        df_filled.drop(columns=cols_to_drop, inplace=True)
    
    for col in df_filled.columns:
        if ptypes.is_integer_dtype(df_filled[col]) or ptypes.is_float_dtype(df_filled[col]):
            median_val = df_filled[col].median()
            df_filled[col] = df_filled[col].fillna(median_val)
        elif ptypes.is_string_dtype(df_filled[col]) or ptypes.is_object_dtype(df_filled[col]):
            df_filled[col] = df_filled[col].fillna(cat_fill_value)

    return df_filled

def find_missing_datetime(df):
    """
    Print and return the missing datetime stamps for relevant columns.
    """
    df_datetime = df.select_dtypes(include=["datetime64[ns]"])
    dt_missing_dict = {}

    for col in df_datetime.columns:
        nmiss = df[col].isna().sum()
        if nmiss > 0:
            print(f"{col}: {nmiss} missing datetime values")
            dt_missing_dict[col] = nmiss

    return dt_missing_dict

def low_var_filter(df, threshold=0.0):
    """
    Remove numerical columns with variance below the given threshold.

    threshold (float): variance threshold.
    """
    df_numeric = df.select_dtypes(include=[np.number])
    selector = VarianceThreshold(threshold)
    selector.fit(df_numeric)
    selected_numeric_cols = df_numeric.columns[selector.get_support()]
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    filtered_cols = list(selected_numeric_cols) + list(non_numeric_cols)
    removed_cols = df_numeric.columns[~selector.get_support()]
    print("Dropped low variance numeric columns:", list(removed_cols))

    return df[filtered_cols]


def encode_features(df, one_hot_threshold=10, do_not_freq_encode=False, exclude_vars=None):
    """
    Encode categorical features using one-hot or frequency encoding.
    
    one_hot_threshold (int): max unique values to use one-hot encoding; otherwise, frequency encoding.
    """
    if exclude_vars is None:
        exclude_vars = []
    exclude_vars = [col for col in exclude_vars if col in df.columns]
    
    df_encoded = df.copy()
    
    categorical_cols = [
        col for col in df_encoded.columns
        if (ptypes.is_string_dtype(df_encoded[col]) or ptypes.is_object_dtype(df_encoded[col]))
        and col not in exclude_vars
    ]
    print(f'Categorical columns to encode: {categorical_cols}')
    
    if len(categorical_cols) <= 0:
        print("No cat variables found. Skipping encoding.")
    
    for col in categorical_cols:
        print(f"Encoding column: {col}")  
        if df_encoded[col].nunique(dropna=True) <= one_hot_threshold:
            print('Using one-hot encoding')
            dummies = pd.get_dummies(df_encoded[col].astype("string"), prefix=col, drop_first=True, dummy_na=False)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
        else:
            if do_not_freq_encode:
                continue
            print('Using frequency encoding')
            freq = df_encoded[col].value_counts(normalize=True)
            df_encoded[col + '_freq'] = df_encoded[col].map(freq)
            df_encoded.drop(columns=col, inplace=True)
    
    return df_encoded

def iqr_outliers(df, iqr_thresh=1.5):
    """
    Detect outliers in numerical columns using the IQR method.
    """
    df_numeric = df.select_dtypes(include=[np.number])
    outliers_dict = {}

    for col in df_numeric.columns:
        series = df_numeric[col].dropna()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_thresh * iqr
        upper = q3 + iqr_thresh * iqr
        outliers = series[(series < lower) | (series > upper)]

        print(f"{col}: {len(outliers)} outliers ({round(len(outliers)/len(series)*100, 2)}%)")

        if not outliers.empty:
            outliers_dict[col] = outliers

    return outliers_dict

def suggest_targets(df, max_unique_numeric=10):
    """
    Suggest suitable target columns based on uniqueness and data type.

    max_unique_numeric (int): number of unique values to consider for classification task.
    """
    df = df.copy()
    candidates = []
    nrows = len(df)

    for col in df.columns:
        series = df[col]
        nunique = series.nunique(dropna=True)
        
        if nunique <= 1:
            continue
        
        tots = round((nunique / nrows) * 100, 4)

        if ptypes.is_bool_dtype(series) or ptypes.is_string_dtype(series):
            candidates.append((col, nunique, tots, 'classification'))
        elif ptypes.is_integer_dtype(series):
            if nunique <= max_unique_numeric:
                candidates.append((col, nunique, tots, 'classification'))
            else:
                candidates.append((col, nunique, tots, 'regression'))
        elif ptypes.is_float_dtype(series):
            if nunique <= max_unique_numeric:
                candidates.append((col, nunique, tots, 'classification'))
            else:
                candidates.append((col, nunique, tots, 'regression'))

    return pd.DataFrame(candidates, columns=['column', 'num_unique_values', 'percent_of_total_rows', 'suggested_task'])
