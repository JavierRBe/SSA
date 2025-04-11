#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 26 Nov 10:40:00 2024

@author: Javier
"""

import pandas as pd
import numpy as np
from scipy.stats import shapiro
import scipy as scipy
import matplotlib.pyplot as plt
import os

def get_metadata_columns(df, prefix='metadata_'):
    """
    Returns all columns in the dataframe that start with a given prefix.

    Parameters:
    - df: pd.DataFrame, the dataframe to inspect.
    - prefix: str, the prefix to search for (default is 'metadata_').

    Returns:
    - metadata_cols: list of column names that start with the given prefix.
    """
    metadata_cols = [col for col in df.columns if col.startswith(prefix)]
    return metadata_cols

def align_data_by_conditions(df, conditions=['prestim', 'bluelight', 'poststim']):
    """
    Aligns data for each well based on conditions and renames columns appropriately.

    Parameters:
    - df: pd.DataFrame, the main dataframe containing all conditions data.
    - conditions: list, names of the conditions in order (default: ['prestim', 'bluelight', 'poststim']).

    Returns:
    - pd.DataFrame: The aligned dataframe with renamed columns for each condition.
    """
    if not conditions:
        # If conditions list is empty, return the input dataframe as is
        return df

    if 'core_well_id' not in df.columns:
        raise ValueError("The dataframe must contain a 'core_well_id' column.")

    # Extract condition from the 'core_well_id'
    condbox = []
    for key in df['core_well_id']:
        currkey = len(condbox)
        for cond in conditions:
            if f"_{cond}" in key:
                condbox.append(cond)
        if currkey == len(condbox):
            condbox.append('nan')

    df['condition'] = condbox

    # Split data into separate dataframes for each condition
    condition_dfs = {cond: df[df['condition'] == cond].drop(columns='condition').reset_index(drop=True)
                     for cond in conditions}

    # Identify metadata and core columns
    metadata_columns = [col for col in df.columns if col.startswith(('metadata_', 'core_'))]
    feature_columns = [col for col in df.columns if col not in metadata_columns + ['core_well_id', 'condition']]

    # Prefer 'bluelight' as the starting condition if it exists; otherwise, use the first valid condition
    starting_condition = 'bluelight' if 'bluelight' in condition_dfs and not condition_dfs['bluelight'].empty else \
        next((cond for cond in conditions if cond in condition_dfs and not condition_dfs[cond].empty), None)

    if not starting_condition:
        raise ValueError("No valid starting condition found in the dataframe.")

    # Start with the metadata columns from the starting condition
    result_df = condition_dfs[starting_condition][metadata_columns + ['core_well_id']].copy()

    # Add feature columns from all conditions with renamed columns
    for condition in conditions:
        if condition in condition_dfs and not condition_dfs[condition].empty:
            renamed_features = {col: f"{col}_{condition}" for col in feature_columns}
            condition_features_df = condition_dfs[condition][feature_columns].rename(columns=renamed_features)
            result_df = pd.concat([result_df, condition_features_df], axis=1)

    # Drop any duplicate 'core_well_id' columns
    result_df = result_df.loc[:, ~result_df.columns.duplicated()]

    return result_df


def drop_nan_core_well_id(df):
    """
    Drops rows where the 'core_well_id' column contains NaN values.

    Parameters:
    - df: pd.DataFrame, the dataframe to process.

    Returns:
    - pd.DataFrame: The dataframe with rows containing NaN in 'core_well_id' removed.
    """
    if 'core_well_id' not in df.columns:
        raise ValueError("The dataframe must contain a 'core_well_id' column.")
    
    return df.dropna(subset=['core_well_id']).reset_index(drop=True)

def drop_rows_with_all_nan_features(df):
    """
    Drops rows where all feature columns (excluding metadata and core columns) contain NaN values.
    
    Parameters:
    - df: pd.DataFrame, the dataframe to process.

    Returns:
    - pd.DataFrame: The dataframe with rows dropped where all feature columns are NaN.
    """
    # Identify metadata and core columns
    metadata_core_columns = [col for col in df.columns if col.startswith(('metadata_', 'core_'))]
    
    # Identify feature columns (non-metadata and non-core columns)
    feature_columns = [col for col in df.columns if col not in metadata_core_columns]
    
    # Drop rows where all feature columns are NaN
    df_cleaned = df.dropna(subset=feature_columns, how='all').reset_index(drop=True)
    
    return df_cleaned

def gather_metadata(df):
    """
    Returns a new dataframe containing only the columns with 'metadata_' or 'core_' prefixes.
    
    Parameters:
    - df: pd.DataFrame, the original dataframe.

    Returns:
    - pd.DataFrame: A new dataframe with only metadata and core columns.
    """
    # Filter columns that start with 'metadata_' or 'core_'
    metadata_core_columns = [col for col in df.columns if col.startswith(('metadata_', 'core_'))]
    
    # Return a new dataframe with only those columns
    return df[metadata_core_columns]


def drop_feat_by_keyword(feat, keywords):
    """
    Remove features (columns) that contain a keyword, only modifying
    feature columns (those without 'core_' or 'metadata_' prefixes).
    
    Parameters:
        feat (pd.DataFrame): The input dataframe with features.
        keywords (str or list): The keyword(s) to filter and drop.

    Returns:
        pd.DataFrame: The filtered dataframe with selected columns dropped.
    """
    import numpy as np

    # Identify feature columns (not prefixed with 'core_' or 'metadata_')
    feature_columns = [col for col in feat.columns if not col.startswith(('core_', 'metadata_'))]

    # Filter only feature columns based on the keywords
    if isinstance(keywords, (list, np.ndarray)):
        for key in keywords:
            feature_columns = [col for col in feature_columns if key not in col]
    elif isinstance(keywords, str):
        feature_columns = [col for col in feature_columns if keywords not in col]

    # Retain non-feature columns and filtered feature columns
    retained_columns = [col for col in feat.columns if col.startswith(('core_', 'metadata_'))] + feature_columns
    return feat[retained_columns]

def _check_for_normality(feat):
    """
    Check if each numeric feature column (excluding 'core_' and 'metadata_' prefixed columns)
    in the DataFrame is normally distributed.

    Parameters:
    - feat (pd.DataFrame): Input DataFrame.

    Returns:
    - result_df (pd.DataFrame): A DataFrame with feature column names and a single row indicating
                                whether each column is normally distributed (True/False).
    """

    # Identify feature columns (not prefixed with 'core_' or 'metadata_')
    feature_columns = [col for col in feat.select_dtypes(include=[np.number]).columns
                       if not col.startswith(('core_', 'metadata_'))]

    _result_dict = {}

    for column in feature_columns:
        # Perform Shapiro-Wilk test
        if len(feat[column].dropna())>2:
            stat, _p_value = shapiro(feat[column].dropna())  # Drop NA values for the test
            # If p-value > 0.05, the data is normally distributed
            _result_dict[column] = _p_value > 0.05
        else:
            _result_dict[column] = 1 > 0.05
    # Convert the result dictionary to a DataFrame
    _norm_cols_rlts = pd.DataFrame(_result_dict, index=['Normality'])

    return _norm_cols_rlts

def check_well_integrity(df,well_type):
    """
    Check the integrity of well data, ensuring only wells with good integrity are kept.
    Filters only the feature columns (not prefixed with 'core_' or 'metadata_') in the feature dataframe.

    Parameters:
    - feat (DataFrame): DataFrame containing feature data.
    - meta (DataFrame): DataFrame containing metadata.

    Returns:
    - feat_filtered (DataFrame): Filtered feature DataFrame containing data only for wells with good integrity.
    - meta_filtered (DataFrame): Filtered metadata DataFrame containing information only for wells with good integrity.
    """
    if well_type=="6WP":

        # Check if feat DataFrame contains 'core_well_id' column
        if 'core_well_id' not in df.columns:
            raise ValueError("'core_well_id' column is missing")
        
        # Check if meta DataFrame contains 'core_well_id', 'is_bad_well', and 'is_good_well' columns
        required_columns = ['core_well_id','metadata_Is_bad_well']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"{missing_columns} columns are required")

        # Filter out wells that are marked as bad or not marked as good
        discarded_wells = df[(df['metadata_Is_bad_well'] != False)]
        num_discarded_wells = len(discarded_wells)
        print(f'{num_discarded_wells} wells were discarded')

        # Update meta DataFrame by removing discarded wells
        df_filtered = df[(df['metadata_Is_bad_well'] == False)]

        # Filter feat DataFrame to include only wells present in the updated meta DataFrame
        # Retain core_ and metadata_ columns as they are
    else:
        # Check if feat DataFrame contains 'core_well_id' column
        if 'core_well_id' not in df.columns:
            raise ValueError("'core_well_id' column is missing")
        
        # Check if meta DataFrame contains 'core_well_id', 'is_bad_well', and 'is_good_well' columns
        required_columns = ['core_well_id', 'metadata_is_good_well','metadata_Is_bad_well']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"{missing_columns} columns are required")

        # Filter out wells that are marked as bad or not marked as good
        discarded_wells = df[(df['metadata_is_good_well'] != 1) & (df['metadata_Is_bad_well'] != False)]
        num_discarded_wells = len(discarded_wells)
        print(f'{num_discarded_wells} wells were discarded')

        # Update meta DataFrame by removing discarded wells
        df_filtered = df[(df['metadata_is_good_well'] == 1) & (df['metadata_Is_bad_well'] == False)]

        # Filter feat DataFrame to include only wells present in the updated meta DataFrame
        # Retain core_ and metadata_ columns as they are

    return df_filtered
 

def discard_columns_by_nan_inf(feat, drop_th):
    """
    Check for missing values (NaN) and infinite values (inf) in numeric feature columns of the DataFrame,
    and discard feature columns that have more than the specified threshold percentage of these values.

    Parameters:
    - feat (pd.DataFrame): The input DataFrame containing the data to be checked.
    - drop_th (float): Threshold percentage for dropping columns.

    Returns:
    - pd.DataFrame: A DataFrame with filtered columns.
    """
    if not isinstance(feat, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    
    # Identify feature columns (not prefixed with 'core_' or 'metadata_') and are numeric
    feature_columns = [col for col in feat.select_dtypes(include=[np.number]).columns if not col.startswith(('core_', 'metadata_'))]

    # Calculate the percentage of missing and infinite values in each feature column
    def percentage_missing_and_inf(column):
        total = len(column)
        num_missing = column.isna().sum()
        num_inf = np.isinf(column).sum()
        return (num_missing + num_inf) / total * 100
    
    # Apply the function to each numeric feature column
    perc_missing_inf = feat[feature_columns].apply(percentage_missing_and_inf)

    # Identify columns to drop (those with more than the threshold)
    warn_th = 1.5 * drop_th
    cols_to_drop = perc_missing_inf[perc_missing_inf > drop_th].index
    cols_to_warn = perc_missing_inf[perc_missing_inf > warn_th].index if len(perc_missing_inf[perc_missing_inf > warn_th]) > 0 else None

    if cols_to_warn is not None:
        print(f'Warning - Columns: {list(cols_to_warn)} have more than {warn_th}% missing or inf values, the integrity of these features should be inspected.')

    # Drop the identified columns
    cleaned_feat = feat.drop(columns=cols_to_drop)

    if len(cols_to_drop) > 0:
        print(f"Columns discarded due to more than {drop_th}% missing or inf values: {list(cols_to_drop)}")
    else:
        print('No features were discarded.')

    return cleaned_feat

def impute_features(feat, impute_method='smart'):
    """
    Replace missing (NaN) and infinite values in the numeric feature columns of a DataFrame.
    Only modifies columns that do not have the 'core_' or 'metadata_' prefix.

    Parameters:
    - feat (pd.DataFrame): The input DataFrame to process.
    - impute_method (str): Imputation method ('smart', 'mean', or 'median').

    Returns:
    - pd.DataFrame: A DataFrame with missing and infinite values in feature columns imputed.
    """
    import numpy as np

    if not isinstance(feat, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # Identify numeric feature columns (not prefixed with 'core_' or 'metadata_')
    feature_columns = [col for col in feat.select_dtypes(include=[np.number]).columns if not col.startswith(('core_', 'metadata_'))]

    if impute_method == 'smart':
        # Identify normality of feature columns
        _norm_cols_rlts = _check_for_normality(feat[feature_columns])

    for col in feature_columns:
        if impute_method == 'smart':
            if _norm_cols_rlts[col].loc['Normality'] == False:
                _imputed_value = feat[col].median()  # Use median for non-normal distributions
            else:
                _imputed_value = feat[col].mean()  # Use mean for normal distributions
        elif impute_method == 'mean':
            _imputed_value = feat[col].mean()  # Use mean for all columns
        elif impute_method == 'median':
            _imputed_value = feat[col].median()  # Use median for all columns
        else:
            raise ValueError("Unknown imputation method. Use 'smart', 'mean', or 'median'.")

        # Replace infinite values with NaN and impute missing values
        feat[col] = feat[col].replace([np.inf, -np.inf], np.nan)  # Replace inf values with NaN
        feat[col] = feat[col].fillna(_imputed_value)  # Impute NaN values

    return feat

def transform_columns_to_labels(df):
    """
    Converts numeric metadata and core columns to string format.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.

    Returns:
    - pd.DataFrame: The DataFrame with 'metadata_' and 'core_' numeric columns converted to strings.
    """
    # Identify metadata and core columns that are numeric
    metadata_core_columns = [col for col in df.select_dtypes(include=['int', 'float']).columns
                             if col.startswith(('metadata_', 'core_'))]

    # Convert these columns to strings
    for column in metadata_core_columns:
        df[column] = df[column].astype(str)

    return df


def categorise_metadata(df):
    """
    Convert 'metadata_' columns to string or category type based on a threshold.
    If the number of unique values is less than 10% of the sample size, the column becomes a category type.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with 'metadata_' columns converted to string or category type.

    Warning:
    - The input DataFrame will be modified!!.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # Filter metadata columns
    metadata_columns = [col for col in df.columns if col.startswith('metadata_') and df[col].dtype == 'object']

    for column in metadata_columns:
        col_len = len(df[column])
        unique_counts = df[column].value_counts()
        if len(unique_counts) < 0.1 * col_len:
            df[column] = df[column].astype('category')
    
    return df


def outlier_inquisition(feat, tolerance=3):
    """
    Replace outliers in numeric feature columns of the DataFrame.
    Only modifies columns that do not have the 'core_' or 'metadata_' prefix.
    
    Outliers are defined as values greater than mean + tolerance*std or less than mean - tolerance*std.
    Outliers are replaced by the median value of the column.

    Parameters:
    - feat (pd.DataFrame): Input DataFrame.
    - tolerance (float): The number of standard deviations to define outliers.

    Returns:
    - pd.DataFrame: A DataFrame with outliers in feature columns replaced by median values.
    """
    import numpy as np

    if not isinstance(feat, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # Identify numeric feature columns (not prefixed with 'core_' or 'metadata_')
    feature_columns = [col for col in feat.select_dtypes(include=[np.number]).columns if not col.startswith(('core_', 'metadata_'))]

    for column in feature_columns:
        mean = feat[column].mean()
        std = feat[column].std()
        median = feat[column].median()

        upper_bound = mean + tolerance * std
        lower_bound = mean - tolerance * std

        # Replace outliers with the median value
        feat[column] = np.where((feat[column] > upper_bound) | (feat[column] < lower_bound), median, feat[column])

    return feat


def craft_well_keys(recap,well_type):
    if well_type=="6WP":
        recap['core_well_id']=recap['metadata_filename']
    else:
        recap['core_well_id']=recap['metadata_filename']+'_'+recap['core_well_name']
    return recap

def drop_wells_by_skeletons(df, threshold,class_column_main,output_directory):
        """
        Drops wells from the dataframe where 'core_n_skeletons' is below the specified threshold.
        After dropping, generates a bar plot showing the number of skeletons per class using the
        class column specified in `class_column_main`.

        Parameters:
        - df (pd.DataFrame): The input dataframe.
        - threshold (int or float): Minimum number of skeletons required to keep a well.
        - output_directory (str): Directory to save the skeleton count plot.
        - class_column_main (str): The column name that identifies the class (from params['class_column_main']).

        Returns:
        - pd.DataFrame: DataFrame with wells below the threshold dropped.
        """

        if 'core_n_skeletons' not in df.columns:
            raise ValueError("'core_n_skeletons' column is missing from the dataframe.")

        if class_column_main not in df.columns:
            raise ValueError(f"'{class_column_main}' column is missing from the dataframe. Check params['class_column_main'].")

        # Drop wells below threshold
        filtered_df = df[df['core_n_skeletons'] >= threshold].reset_index(drop=True)

        # Group by the dynamic class column and calculate total skeleton counts after filtering
        skeletons_by_class = filtered_df.groupby(class_column_main)['core_n_skeletons'].sum()

        # Create the output directory if needed
        preprocessing_dir = os.path.join(output_directory, "Skeleton_count")
        os.makedirs(preprocessing_dir, exist_ok=True)

        # Plot the skeleton counts per class
        plt.figure(figsize=(12, 6))
        skeletons_by_class.sort_values().plot(kind='bar', color='steelblue')
        plt.title('Skeleton Counts Per Class (After Filtering)')
        plt.ylabel('Number of Skeletons')
        plt.xlabel('Class')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(preprocessing_dir, "Skeleton_Counts_Per_Class.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f" Skeleton count plot saved at: {plot_path}")

        return filtered_df

def filter_core_and_metadata_columns(df):
    """
    Filters the dataframe to include only columns with prefixes 'core_' and 'metadata_'.

    Parameters:
    - df (pd.DataFrame): The input dataframe.

    Returns:
    - pd.DataFrame: A dataframe containing only 'core_' and 'metadata_' columns.
    """
    # Select columns that start with 'core_' or 'metadata_'
    filtered_columns = [col for col in df.columns if col.startswith(('core_', 'metadata_'))]
    return df[filtered_columns]

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder DataFrame columns based on the following priority:
    1. Core columns (prefix: 'core_')
    2. Metadata columns:
        - Other metadata columns (no '_value' or '_unit' suffix)
        - *_value columns
        - *_unit columns (following corresponding *_value columns)
    3. Feature columns (columns not in the above categories)
    4. Other columns (e.g., PCA or derived columns)

    Parameters:
    - df (pd.DataFrame): Input DataFrame with mixed column types.

    Returns:
    - pd.DataFrame: DataFrame with reordered columns.
    """

    # Identify column groups
    core_cols = sorted([col for col in df.columns if col.startswith("core_")])
    metadata_cols = [col for col in df.columns if col.startswith("metadata_")]

    # Separate metadata into categories
    metadata_value_cols = [col for col in metadata_cols if col.endswith("_value")]
    metadata_unit_cols = [col for col in metadata_cols if col.endswith("_unit")]
    metadata_other_cols = sorted([col for col in metadata_cols if col not in metadata_value_cols + metadata_unit_cols])

    # Feature columns (excluding identified categories)
    feature_cols = sorted([
        col for col in df.columns 
        if col not in core_cols + metadata_cols
        and not col.startswith(("Day", "Class_Code", "PC"))
    ])

    # Other columns (e.g., PCA or derived columns)
    other_cols = sorted([
        col for col in df.columns
        if col not in core_cols + metadata_cols + feature_cols
    ])

    # Final column order
    final_col_order = core_cols + metadata_other_cols + metadata_value_cols + metadata_unit_cols + feature_cols + other_cols

    # Reorder and return DataFrame
    return df[final_col_order]
