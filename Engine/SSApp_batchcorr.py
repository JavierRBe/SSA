#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue 26 Nov 10:40:00 2024

@author: Javier
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
import os
import gc


class BatchProcessor:
    def __init__(self, df, class_column, tierset, main_control_name, params):
        """
        Initialize the BatchProcessor for batch correction and analysis.

        Parameters:
        - df (pd.DataFrame): The input DataFrame to process.
        - class_column (str): The column name representing class labels.
        - tierset (list): The list of features to include in batch correction.
        - main_control_name (str): The name of the main control class.
        """
        self.df = df
        self.params = params
        self.class_column = class_column
        self.tierset = tierset
        self.main_control_name = main_control_name

        # Extract class-related parameters
        self.main_control = params.get("main_control_name")
        self.positive_controls = params.get("positive_controls", [])
        self.negative_controls = params.get("negative_controls", [])

        # Extract the list of unique classes from the class_column
        self.compound_list = df[class_column].unique().tolist()

        # Ensure main control is included first in the list (if it exists)
        if self.main_control_name in self.compound_list:
            self.compound_list.remove(self.main_control_name)
            self.compound_list.insert(0, self.main_control_name)  # Keep main control first

        # Identify feature columns (exclude metadata and core columns)
        self.feature_columns = [col for col in df.columns if any(base_feature in col for base_feature in tierset)]


    def _get_feature_columns(self):
        """
        Identify feature columns by filtering out metadata and core columns,
        and ensuring they are part of the `tierset`.
        """
        feature_columns = [col for col in self.df.columns if not col.startswith(("metadata_", "core_"))]
        return [col for col in feature_columns if col in self.tierset]

    def batch_correct(self, method="mean"):
        """
        Perform batch correction based on the selected method.

        Parameters:
        - method (str): Correction method ('mean', 'median', 'ratio', 'proportional', or 'ruv').

        Returns:
        - pd.DataFrame: Batch-corrected DataFrame.
        """
        if method == "mean":
            return self._batch_correction_by_global_stat(stat="mean")
        elif method == "median":
            return self._batch_correction_by_global_stat(stat="median")
        elif method == "ratio":
            return self._batch_correction_by_ratio()
        elif method == "proportional":
            return self._proportional_scaling_correction()
        else:
            raise ValueError(f"Unsupported batch correction method: {method}")

    def _batch_correction_by_global_stat(self, stat="mean"):
        """
        Perform batch correction by aligning control data to the global mean or median.

        Parameters:
        - stat (str): Statistic to use for correction ('mean' or 'median').

        Returns:
        - pd.DataFrame: Batch-corrected DataFrame.
        """
        if self.main_control_name not in self.df[self.class_column].unique():
            raise ValueError(f"Control class '{self.main_control_name}' not found in the DataFrame.")

        # Filter control data
        control_df = self.df[self.df[self.class_column] == self.main_control_name]

        # Calculate the global control statistic
        global_stat = control_df[self.feature_columns].agg(stat)

        # Store corrections by date
        day_corrections = {}
        for date in control_df["metadata_Day_of_run"].unique():
            day_data = control_df[control_df["metadata_Day_of_run"] == date][self.feature_columns]
            day_stat = day_data.agg(stat)
            correction_shift = global_stat - day_stat
            day_corrections[date] = correction_shift

        # Apply corrections
        corrected_df = self.df.copy()
        for date, correction_shift in day_corrections.items():
            rows_to_correct = corrected_df["metadata_Day_of_run"] == date
            corrected_df.loc[rows_to_correct, self.feature_columns] += correction_shift

        return corrected_df

    def _batch_correction_by_ratio(self):
        """
        Perform batch correction using ratio-based normalization.

        Returns:
        - pd.DataFrame: Batch-corrected DataFrame.
        """
        if self.main_control_name not in self.df[self.class_column].unique():
            raise ValueError(f"Control class '{self.main_control_name}' not found in the DataFrame.")

        control_df = self.df[self.df[self.class_column] == self.main_control_name]
        day_means = {date: control_df[control_df["metadata_Day_of_run"] == date][self.feature_columns].mean()
                     for date in control_df["metadata_Day_of_run"].unique()}

        corrected_df = self.df.copy()
        for date, day_mean in day_means.items():
            rows_to_correct = corrected_df["metadata_Day_of_run"] == date
            corrected_df.loc[rows_to_correct, self.feature_columns] /= day_mean

        return corrected_df

    def _proportional_scaling_correction(self):
        """
        Perform batch correction using proportional scaling.

        Returns:
        - pd.DataFrame: Batch-corrected DataFrame.
        """
        global_means = self.df[self.df[self.class_column] == self.main_control_name][self.feature_columns].mean()
        corrected_df = self.df.copy()

        for batch in self.df["metadata_Day_of_run"].unique():
            batch_idx = corrected_df["metadata_Day_of_run"] == batch
            control_batch_idx = corrected_df[(self.df["metadata_Day_of_run"] == batch) &
                                             (self.df[self.class_column] == self.main_control_name)]

            if not control_batch_idx.empty:
                batch_means = control_batch_idx[self.feature_columns].mean()
                scaling_factors = global_means / batch_means
                corrected_df.loc[batch_idx, self.feature_columns] *= scaling_factors

        return corrected_df

    def plot_boxplots(self, before_df, after_df, savepath, batch_size=50):
        """
        Generate boxplots before and after correction.

        Parameters:
        - before_df (pd.DataFrame): DataFrame before correction.
        - after_df (pd.DataFrame): DataFrame after correction.
        - savepath (str): Path to save the plots.
        - batch_size (int): Number of features to process per batch.
        """
        os.makedirs(savepath, exist_ok=True)

        before_df = self._preprocess_for_plotting(before_df)
        after_df = self._preprocess_for_plotting(after_df)
        class_color_mapping = self._assign_colors()

        for i in range(0, len(self.feature_columns), batch_size):
            batch_features = self.feature_columns[i:i + batch_size]
            for feature in batch_features:
                self._plot_feature_boxplot(feature, before_df, after_df, savepath)

    def _preprocess_for_plotting(self, df):
        """
        Preprocess data for plotting (filter, sort, and create combined Class_Date column).
        """
        df = df[df[self.class_column].isin(self.compound_list)].copy()
        df["metadata_Day_of_run"] = df["metadata_Day_of_run"].astype(str)
        df["Class_Date"] = df[self.class_column].astype(str) + "_" + df["metadata_Day_of_run"].astype(str)
        df[self.class_column] = pd.Categorical(df[self.class_column], categories=self.compound_list, ordered=True)
        return df.sort_values(by=[self.class_column, "metadata_Day_of_run"])

    def _assign_colors(self):
        """
        Assign unique colors to each class for consistent plotting.
        """
        base_palette = sns.color_palette("husl", len(self.compound_list))
        return {comp: base_palette[i] for i, comp in enumerate(self.compound_list)}

    def _plot_feature_boxplot(self, feature, before_df, after_df, savepath):
        """
        Generates boxplots for a single feature before and after batch correction.

        Parameters:
        - feature (str): The feature to be plotted.
        - before_df (pd.DataFrame): Data before batch correction.
        - after_df (pd.DataFrame): Data after batch correction.
        - savepath (str): Path to save the generated plots.
        """

        # Extract classes from parameters
        main_control = self.main_control
        positive_controls = self.positive_controls
        negative_controls = self.negative_controls

        # Define the subset of classes to be plotted
        selected_classes = [main_control] + negative_controls + positive_controls

        # Filter the data to include only these classes
        before_df = before_df[before_df[self.class_column].isin(selected_classes)]
        after_df = after_df[after_df[self.class_column].isin(selected_classes)]

        # Ensure 'Class_Date' remains categorical
        before_df.loc[:, "Class_Date"] = before_df[self.class_column].astype(str) + "_" + before_df["metadata_Day_of_run"].astype(str)
        after_df.loc[:, "Class_Date"] = after_df[self.class_column].astype(str) + "_" + after_df["metadata_Day_of_run"].astype(str)

        # Assign unique colors per class (one color per class)
        base_palette = sns.color_palette("husl", len(selected_classes))
        class_color_mapping = {cls: base_palette[i] for i, cls in enumerate(selected_classes)}

        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Plot before correction
            sns.boxplot(
                x="Class_Date",
                y=feature,
                hue="Class_Date",  # Prevent FutureWarning
                data=before_df,
                ax=axes[0],
                order=before_df["Class_Date"].unique(),
                palette={cls: class_color_mapping[cls.split("_")[0]] for cls in before_df["Class_Date"].unique()},
                width=0.25,
                linewidth=1.2,
                dodge=False,
                boxprops={"alpha": 0.3}  # Reduce transparency
            )
            axes[0].set_title(f'{feature} - Before Correction')
            axes[0].set_xlabel('')
            axes[0].set_ylabel(feature)
            axes[0].tick_params(axis='x', rotation=90)
            axes[0].legend([], [], frameon=False)  # Hide the legend

            # Plot after correction
            sns.boxplot(
                x="Class_Date",
                y=feature,
                hue="Class_Date",  # Prevent FutureWarning
                data=after_df,
                ax=axes[1],
                order=after_df["Class_Date"].unique(),
                palette={cls: class_color_mapping[cls.split("_")[0]] for cls in after_df["Class_Date"].unique()},
                width=0.25,
                linewidth=1.2,
                dodge=False,
                boxprops={"alpha": 0.3}  # Reduce transparency
            )
            axes[1].set_title(f'{feature} - After Correction')
            axes[1].set_xlabel('')
            axes[1].set_ylabel(feature)
            axes[1].tick_params(axis='x', rotation=90)
            axes[1].legend([], [], frameon=False)  # Hide the legend

            # Save the plot
            plt.tight_layout()
            filename = os.path.join(savepath, f"BatchCorrectionBoxplots_{feature.replace('/', '_')}.png")
            plt.savefig(filename, dpi=300, format="png")
            plt.close("all")
            print(f"Saved: {filename}")

        except Exception as e:
            print(f"Error generating plot for feature '{feature}': {e}")

    def plot_control_only_pca(self, before_df, after_df, savepath):
        """PCA on control samples before & after correction with samples colored by day."""

        os.makedirs(savepath, exist_ok=True)

        def prepare_pca_data(df):
            """
            Prepare data for PCA using main control samples, avoiding DataFrame fragmentation.

            Parameters:
            - df (pd.DataFrame): Input DataFrame with a 'metadata_Day_of_run' column.

            Returns:
            - df (pd.DataFrame): Filtered DataFrame with added 'Day' and 'Day_Code' columns.
            - pca_result (np.ndarray): PCA-transformed data with 3 principal components.
            - day_map (dict): Mapping from day strings to integer codes.
            """

            # Filter only main control samples (with a proper copy to avoid SettingWithCopyWarning)
            df = df[df[self.class_column] == self.main_control_name].copy()

            # Create the 'Day' column from 'metadata_Day_of_run'
            df["Day"] = pd.to_datetime(df["metadata_Day_of_run"]).dt.strftime('%Y-%m-%d')

            # Map days to integer codes
            day_map = {day: i for i, day in enumerate(sorted(df["Day"].unique()))}

            # Add 'Day_Code' column efficiently
            df = pd.concat([df, pd.Series(df["Day"].map(day_map), name="Day_Code")], axis=1)

            # Perform PCA on the feature columns
            pca_result = PCA(n_components=3).fit_transform(df[self.feature_columns])

            return df, pca_result, day_map



        # Prepare data
        control_before, pca_before, before_day_map = prepare_pca_data(before_df)
        control_after, pca_after, after_day_map = prepare_pca_data(after_df)

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Before Correction", "After Correction"),
            specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
            horizontal_spacing=0.15
        )

        # Plot function
        def plot_pca(fig, pca_data, df, day_map, row, col, title):
            fig.add_trace(
                go.Scatter3d(
                    x=pca_data[:, 0],
                    y=pca_data[:, 1],
                    z=pca_data[:, 2],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=df["Day_Code"],  # Integer-coded days
                        colorscale="Viridis",
                        colorbar=dict(
                            title="Sample Day",
                            tickvals=list(day_map.values()),
                            ticktext=list(day_map.keys()),
                            len=0.75
                        )
                    ),
                    text=[f"Day: {day}" for day in df["Day"]],
                    hovertemplate="<b>Day:</b> %{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>"
                ),
                row=row, col=col
            )
            fig.update_scenes(
                xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
                aspectmode='cube', row=row, col=col
            )

        # Add traces
        plot_pca(fig, pca_before, control_before, before_day_map, row=1, col=1, title="Before Correction")
        plot_pca(fig, pca_after, control_after, after_day_map, row=1, col=2, title="After Correction")

        # Update layout
        fig.update_layout(
            title_text="PCA of Control Samples (Before & After Correction)",
            title_x=0.5, width=1000, height=600,
            showlegend=False
        )

        # Save plot
        fig.write_html(os.path.join(savepath, "Control_Only_PCA_Before_After.html"))
        print(f" Saved: {os.path.join(savepath, 'Control_Only_PCA_Before_After.html')}")

    def plot_full_data_pca(self, before_df, after_df, savepath):
        """Full data PCA with toggle button for class-based and day-based coloring."""

        os.makedirs(savepath, exist_ok=True)

        def prepare_pca_data(df):
            """
            Prepare data for PCA with efficient column assignment (no fragmentation).

            Parameters:
            - df (pd.DataFrame): Input DataFrame containing 'metadata_Day_of_run', 'class_column', and feature columns.

            Returns:
            - df (pd.DataFrame): Updated DataFrame with 'Day', 'Day_Code', and 'Class_Code'.
            - pca_result (np.ndarray): PCA result with 3 principal components.
            - day_map (dict): Mapping from day strings to integer codes.
            - class_map (dict): Mapping from class names to integer codes.
            """

            # Identify feature and class columns from df
            feature_columns = [col for col in df.columns if not col.startswith(("metadata_", "core_"))]
            class_column = "class_" if "class_" in df.columns else df.select_dtypes(include='object').columns[0]

            # Create 'Day' column
            day_series = pd.to_datetime(df["metadata_Day_of_run"]).dt.strftime('%Y-%m-%d')

            # Generate mappings
            day_map = {day: i for i, day in enumerate(sorted(day_series.unique()))}
            class_map = {cls: i for i, cls in enumerate(sorted(df[class_column].unique()))}

            # Create new columns in one go to avoid fragmentation
            new_columns = pd.DataFrame({
                "Day": day_series,
                "Day_Code": day_series.map(day_map),
                "Class_Code": df[class_column].map(class_map)
            }, index=df.index)  # Maintain index alignment

            # Concatenate without causing fragmentation
            df = pd.concat([df.reset_index(drop=True), new_columns.reset_index(drop=True)], axis=1)

            # Perform PCA
            pca_result = PCA(n_components=3).fit_transform(df[feature_columns])

            return df, pca_result, day_map, class_map


        # Prepare data
        before_df, pca_before, before_day_map, before_class_map = prepare_pca_data(before_df)
        after_df, pca_after, after_day_map, after_class_map = prepare_pca_data(after_df)

        def get_pca_trace(pca_data, df, code_col, map_dict, label, color_mode):
            """Return a PCA scatter trace with numeric color codes."""
            return go.Scatter3d(
                x=pca_data[:, 0],
                y=pca_data[:, 1],
                z=pca_data[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    color=df[code_col],
                    colorscale="Viridis",
                    colorbar=dict(
                        title="Class" if color_mode == "class" else "Sample Day",
                        tickvals=list(map_dict.values()),
                        ticktext=list(map_dict.keys()),
                        len=0.75
                    )
                ),
                text=[f"Class: {cls}<br>Day: {day}" for cls, day in zip(df[self.class_column], df['Day'])],
                hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>",
                name=label
            )

        # Create figure with toggle
        fig = go.Figure()

        # Add PCA traces
        # Color by class (before/after)
        fig.add_trace(get_pca_trace(pca_before, before_df, "Class_Code", before_class_map, "Before (Class)", "class"))
        fig.add_trace(get_pca_trace(pca_after, after_df, "Class_Code", after_class_map, "After (Class)", "class"))

        # Color by day (before/after)
        fig.add_trace(get_pca_trace(pca_before, before_df, "Day_Code", before_day_map, "Before (Day)", "day"))
        fig.add_trace(get_pca_trace(pca_after, after_df, "Day_Code", after_day_map, "After (Day)", "day"))

        # Toggle buttons
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {"args": [{"visible": [True, True, False, False]}], "label": "Color by Class", "method": "update"},
                    {"args": [{"visible": [False, False, True, True]}], "label": "Color by Day", "method": "update"}
                ],
                "direction": "left", "x": 0.5, "xanchor": "center", "y": 1.2, "showactive": True
            }],
            title_text="PCA of Full Data",
            width=1000, height=600,
            showlegend=False
        )

        # Save plot
        save_path = os.path.join(savepath, "Full_Data_PCA.html")
        fig.write_html(save_path)
        print(f" Saved: {save_path}")





























