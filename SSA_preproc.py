#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 26 Nov 10:40:00 2024

@author: Javier
"""

"""
This script is designed to preprocess high-throughput screening data, clean and harmonize it, and perform batch correction to account 
for systematic variations. It is user-configurable via a JSON parameter file, allowing flexibility and adaptability to various datasets 
and experimental conditions.

- Preprocess Screening Data:

    Harmonize CSV files into a single dataset with consistent column names.
    Drop unnecessary or poorly structured data.
    Impute missing values and handle outliers.

- Perform Batch Correction:

    Remove systematic variations (batch effects) to ensure fair comparisons across experimental runs.
    Provide multiple batch correction options to suit different types of data and experimental setups.

- Generate an Output File:

    Save the cleaned and batch-corrected data for downstream analysis.

General Settings

    csv_directory (str): Path to the directory containing input CSV files.
    output_file (str): Name of the output file to save the processed data.
    feature_set (str): Specifies a predefined feature set (e.g., tierpsy_16) to include in the analysis.

Data Cleaning

    Drop_unshared_columns (bool): Whether to drop columns not shared across all input files.
    drop_keywords (list): Keywords used to drop specific features (columns) from the dataset.
    nan_inf_threshold (int): Maximum percentage of missing or infinite values allowed in a column before it is dropped.
    impute_method (str): Method to impute missing values (smart, mean, median).
    skeletons_threshold (int): Minimum skeleton count required to retain a well.

Metadata Handling

    metadata (list): List of metadata columns to include in the analysis.
    outlier_tolerance (float): Number of standard deviations to define outliers.

Batch Correction

    Perform_batch_correction (bool): Whether to perform batch correction.
    batch_correction_method (str): Batch correction method to use (see options below).
    class_column_main (str): Column representing the class labels for batch correction.
    main_control_name (str): Name of the control class for batch correction.
    positive_controls (list): Names of positive control groups (optional).
    negative_controls (list): Names of negative control groups (optional).

Options

    Batch Correction Options
        - global_control_mean
        - global_control_median
        - ratio
        - proportional
        - ruv
    Tierpsy Set Options
        - tierpsy_8
        - tierpsy_16
        - tierpsy_256
        - tierpsy_2k
    Impute method Options
        - mean
        - median
        - smart
"""

import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm  # For progress bars
import gc  # For explicit garbage collection

from Helper.SSApp_helper import SSApph
from Engine.SSApp_batchcorr import SSAppbc

def preprocess_data(params_file):
    """
    Preprocess CSV data as per the defined sequence and save the result.

    Parameters:
    - params_file: str, path to the JSON file containing preprocessing parameters.
    """
    # Load parameters from the JSON file
    with open(params_file, 'r') as f:
        params = json.load(f)

    # Get CSV directory path from parameters
    input_folder = params.get("csv_directory", None)
    if not input_folder or not os.path.isdir(input_folder):
        raise ValueError(f"Invalid or missing 'csv_directory' in parameters file: {input_folder}")

    tierpsy_set_file = os.path.join(input_folder, 'tierpsysets.json')
    with open(tierpsy_set_file, 'r') as f:
        tierpsy_set = json.load(f)

    # Incremental file processing
    print("Loading and preprocessing CSV files...")
    csv_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the directory: {input_folder}")

    all_columns = set()
    data_generator = (pd.read_csv(file) for file in csv_files)  # Generator to load files one at a time

    dataframes = []
    for df in tqdm(data_generator, desc="Processing CSV files"):
        all_columns.update(df.columns)  # Collect unique columns across files
        dataframes.append(df)

    print("Reindexing dataframes...")
    all_columns = sorted(all_columns)  # Ensure consistent column order
    reindexed_dfs = (df.reindex(columns=all_columns) for df in dataframes)  # Incremental reindexing
    merged_df = pd.concat(reindexed_dfs, ignore_index=True)  # Merge into a single DataFrame

    # Check for required columns
    print("Checking required columns...")
    required_columns = set(params['key_columns'])
    missing_cols = required_columns - set(merged_df.columns)
    if missing_cols:
        raise ValueError(f"The following required columns are missing: {missing_cols}")

    print("Forging well keys...")
    merged_df = SSApph.craft_well_keys(merged_df)

        # Ensure 'metadata_Time_of_run' exists in the metadata
    if 'metadata_Time_of_run' in merged_df.columns:
        # Extract day from metadata_Time_of_run
        merged_df['metadata_Day_of_run'] = pd.to_datetime(merged_df['metadata_Time_of_run']).dt.date
    else:
        raise KeyError("The column 'metadata_Time_of_run' is missing from the dataset.")

    # Verify the new column is successfully created and contains no missing values
    if merged_df['metadata_Day_of_run'].isnull().any():
        raise ValueError("'metadata_Day_of_run' contains null values after extraction. Please check 'metadata_Time_of_run'.")


    # Select Tierpsy feature set
    print("Selecting features...")
    core_metadata_df = SSApph.filter_core_and_metadata_columns(merged_df)
    chosen_tierpsy_set = params.get("feature_set", [])
    feature_ts_list = tierpsy_set.get(chosen_tierpsy_set, [])
    feat_df = merged_df[feature_ts_list]

    # Combine metadata with features
    print("Combining metadata with features...")
    for idx, metacolumn in enumerate(core_metadata_df.columns):
        if metacolumn not in feat_df.columns:
            feat_df[metacolumn] = core_metadata_df.iloc[:, idx].values

    print("Purifying DataFrame...")
    merged_df = feat_df.drop(columns=[col for col in feat_df.columns if col.startswith('core_well_id.')])

    conditions = params.get("conditions", [])
    # Execute preprocessing steps
    print("Starting preprocessing...")
    steps = [
        ("Dropping NaN core_well_id rows", lambda df: SSApph.drop_nan_core_well_id(df)),
        ("Dropping rows with all NaN features", lambda df: SSApph.drop_rows_with_all_nan_features(df)),
        ("Aligning data by conditions", lambda df: SSApph.align_data_by_conditions(df, conditions)),
        ("Dropping wells by number of skeletons", lambda df: SSApph.drop_wells_by_skeletons(df, params.get("skeletons_threshold", []))),
        ("Dropping features by keyword", lambda df: SSApph.drop_feat_by_keyword(df, params.get("drop_keywords", []))),
        ("Checking well integrity", lambda df: SSApph.check_well_integrity(df)),
        ("Discarding columns with NaN/Inf values", lambda df: SSApph.discard_columns_by_nan_inf(df, params.get("nan_inf_threshold", 10))),
        ("Imputing missing values", lambda df: SSApph.impute_features(df, params.get("impute_method", "smart"))),
        ("Labelling metadata", lambda df: SSApph.transform_columns_to_labels(df)),
        ("Categorizing metadata", lambda df: SSApph.categorise_metadata(df)),
        ("Performing outlier inquisition", lambda df: SSApph.outlier_inquisition(df, params.get("outlier_tolerance", 3))),
    ]

    for step_name, step_func in tqdm(steps, desc="Preprocessing steps"):
        print(f"Executing step: {step_name}")
        merged_df = step_func(merged_df)
        gc.collect()  # Explicit garbage collection after each step

    # Perform Batch Correction
    if params.get("Perform_batch_correction", False):
        print("Performing Batch Correction...")

        # Extract batch correction method and ensure it's valid
        batch_correction_method = params.get("batch_correction_method", "mean")
        valid_methods = ["mean", "median", "ratio", "proportional"]
        if batch_correction_method not in valid_methods:
            raise ValueError(f"Invalid batch correction method: {batch_correction_method}")

        # Define the save path for batch correction plots
        savepath = f"{input_folder}/Preprocessing/BCplots_{batch_correction_method}"
        os.makedirs(savepath, exist_ok=True)

        # Initialize the BatchProcessor
        batch_processor = SSAppbc.BatchProcessor(
            df=merged_df,
            class_column=params["class_column_main"],
            tierset=feature_ts_list,
            main_control_name=params["main_control_name"],
            params=params
        )

        # Perform batch correction
        corrected_df = batch_processor.batch_correct(method=batch_correction_method)

        # Plot before and after correction boxplots
        batch_processor.plot_boxplots(
            before_df=merged_df,
            after_df=corrected_df,
            savepath=savepath,
            batch_size=params.get("batch_size", 50)
        )
    else:
        corrected_df = merged_df.copy()

        # Plot PCAs 
    savepath = f"{input_folder}/Preprocessing/PCA"
    batch_processor.plot_control_only_pca(
        before_df=merged_df,
            after_df=corrected_df,
            savepath=savepath)
    
    batch_processor.plot_full_data_pca(
        before_df=merged_df,
            after_df=corrected_df,
            savepath=savepath)

    corrected_df_sorted=SSApph.reorder_columns(corrected_df)
    # Save the resulting DataFrame
    print("Saving the final DataFrame...")
    output_file = params.get("output_file", "SSARecappp.csv")
    output_path = os.path.join(input_folder, output_file)
    corrected_df_sorted.to_csv(output_path, index=False)
    print(f"Preprocessing complete. File saved at: {output_path}")


if __name__ == "__main__":
    params_file = input("Enter the path to the JSON parameters file: ").strip()
    if not os.path.isfile(params_file):
        print("Error: The specified JSON parameters file does not exist.")
    else:
        preprocess_data(params_file)
