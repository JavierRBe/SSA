#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 9 Dic 14:02:00 2024

@author: Javier
"""
import os
import pandas as pd
import json
from tqdm import tqdm
from SSA import SSA  # Import the updated SSA class
import gc  # For garbage collection
from concurrent.futures import ThreadPoolExecutor, as_completed

def validate_params(params):
    """
    Validate the parameters from the JSON file to ensure consistency and correct format.

    Parameters:
    - params (dict): Dictionary of parameters.

    Prints warnings if inconsistencies or incorrect formats are found.
    """
    required_keys = {
        "csv_directory": str,
        "output_file": str,
        "feature_set": str,
        "clustering_method": str,
        "distance_metric": str,
        "normalize_features": str,
        "n_bootstrap_samples": int,
        "consensus_percentage": int,
        "contamination": float,
        "alpha": float,
        "n_cores": int,
        "conditions": list,
        "avoid_outlier_removal": bool,
        "avoid_normalization_HC": bool,
        "avoid_hierarchical_clustering": bool,
        "avoid_SA": bool,
        "avoid_IFA": bool,
        "avoid_PCAs": bool,
        "avoid_Pcoords": bool,
        "avoid_boxplots": bool
    }

    missing_keys = []
    for key, expected_type in required_keys.items():
        if key not in params:
            missing_keys.append(key)
        elif not isinstance(params[key], expected_type):
            print(f"Warning: Parameter '{key}' is expected to be of type {expected_type.__name__}, but got {type(params[key]).__name__}.")

    if missing_keys:
        print(f"Warning: The following required parameters are missing: {', '.join(missing_keys)}.")


def run_analysis_pipeline(params_file):
    """
    Main function to run the analysis pipeline with parameter validation, progress bars, and modular execution.

    Parameters:
    - params_file (str): Path to the JSON parameters file.
    """
    # Load parameters
    with open(params_file, 'r') as f:
        params = json.load(f)

    # Validate parameters
    print("Validating parameters...")
    validate_params(params)

    # Read data from the CSV file
    input_path = os.path.join(params["csv_directory"], params["output_file"])
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    print("Loading data...")
    data = pd.read_csv(input_path)

    # Initialize the SSA class
    print("Initializing analysis pipeline...")
    ssa = SSA(data, params)

    # Create output directory
    output_dir = os.path.join(params["csv_directory"], "Analysis")
    os.makedirs(output_dir, exist_ok=True)

    # Z-score normalization
    if not params.get("avoid_z_normalization", False):
        print("Performing Z-score normalization...")
        ssa.z_normalize_features()
    else:
        print("Skipping Z-score normalization...")

    # Remove outliers from the control class
    if not params.get("avoid_outlier_removal", False):
        print("Removing outliers from control class...")
        ssa.remove_outliers_control()
    else:
        print("Skipping outlier removal...")

    # Perform hierarchical clustering analysis
    if not params.get("avoid_hierarchical_clustering", False):
        print("Performing hierarchical clustering analysis...")
        ssa.hierarchical_clustering()
        ssa.interactive_hierarchical_clustering()
    else:
        print("Skipping hierarchical clustering...")

    # Statistical analysis
    if not params.get("avoid_SA", False):
        print("Performing statistical analysis...")
        ssa.statistical_analysis()
    else:
        print("Skipping statistical analysis...")

    # Isolation forest analysis
    if not params.get("avoid_IFA", False):
        print("Performing Isolation Forest analysis...")
        # Access the IsolationForestAnalysis class and call its method
        ssa.ifa.run()
    else:
        print("Skipping Isolation Forest analysis...")

    # Hits vs Anomalies analysis
    if not params.get("avoid_SA", False) and not params.get("avoid_IFA", False):
        print("Analyzing hits vs anomalies...")
        hits_vs_anomalies_output = os.path.join(output_dir, "HitsVsAnomalies.csv")
        hits_vs_anomalies_df = ssa.analyze_hits_vs_anomalies()
        hits_vs_anomalies_df.to_csv(hits_vs_anomalies_output, index=False)
    else:
        print("Skipping hits vs anomalies analysis due to missing data (SA or IFA skipped).")

    # Generate plots
    print("Generating plots...")

    # PCA plots
    if not params.get("avoid_PCAs", False):
        try:
            print("Generating PCA plots...")
            ssa.plot_pca()
            #ssa.plot_ellipse_pca()
        except Exception as e:
            print(f"Error during PCA plot generation: {e}")
    else:
        print("Skipping PCA plots...")

    # Parallel coordinates plots
    if not params.get("avoid_Pcoords", False):
        try:
            print("Generating parallel coordinates plots...")
            ssa.plot_parallel_coordinates()
        except Exception as e:
            print(f"Error during parallel coordinates plot generation: {e}")
    else:
        print("Skipping parallel coordinates plots...")

    # Boxplots
    if not params.get("avoid_boxplots", False):
        try:
            print("Generating boxplots...")
            ssa.plot_boxplots()
        except Exception as e:
            print(f"Error during boxplot generation: {e}")
    else:
        print("Skipping boxplots...")
    
    # Waterfall
    if not params.get("avoid_waterfall", False):
        try:
            print("Generating Waterfall plots...")
            #ssa.plot_waterfall()
            ssa.interactive_waterfall_plot()
        except Exception as e:
            print(f"Error during waterfall plots generation: {e}")
    else:
        print("Skipping Waterfall plots...")

    # Feature Ditributions

    if not params.get("avoid_feature_distributions", False):
        try:
            print("Generating Feature Ditribution plots...")
            ssa.plot_feature_distributions()
        except Exception as e:
            print(f"Error during Feature Ditribution plots generation: {e}")
    else:
        print("Skipping Feature Ditributions...")

    print("Analysis pipeline completed. Results saved to:", output_dir)

if __name__ == "__main__":
    # Path to the JSON parameters file
    params_file = input("Enter the path to the JSON parameters file: ").strip()
    
    if not os.path.isfile(params_file):
        print(f"Error: JSON parameters file not found at {params_file}")
    else:
        run_analysis_pipeline(params_file)

