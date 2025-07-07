#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 9 Dic 14:02:00 2024

@author: Javier

SSA (Screening Statistical Analysis) Class

This script implements the SSA class, designed for data preprocessing, statistical analysis, 
outlier detection, and visualization. It provides configurable methods to analyze and normalize the data.

Key Components:

**Initialization**:
   - Inputs a pandas DataFrame (`data`) containing metadata, core columns, and feature columns.
   - Reads configuration parameters (`params`) to customize the behavior of the class.
   - Combines class labels (if needed) and organizes the data into metadata, core, and feature columns.

**Data Structure**:
   - Separates feature columns for analysis and creates a dictionary (`data_dict`) where data is grouped by class.
   - Metadata and core columns are identified to ensure flexibility in handling datasets with varying structures.

**Outlier Detection**:
   - Implements robust outlier detection using Isolation Forest, with optional PCA visualization for control class 
     before and after outlier removal.

**Normalization**:
   - Includes methods to normalize features globally or relative to the control class, supporting z-score 
     normalization and other techniques.

**Statistical Analysis**:
   - Performs customizable statistical tests between the control class and other classes.
   - Supports various statistical methods (e.g., t-tests, Mann-Whitney, Pearson correlation) and multiple 
     testing correction (e.g., FDR, Bonferroni).

**Visualization**:
   - Generates hierarchical clustering heatmaps with flexible configurations to adapt to varying feature 
     sets and conditions.
   - Provides additional visualization methods such as PCA, parallel coordinates, and boxplots, with support 
     for class highlighting.

**Key Variables**:
   - `data`: The main DataFrame containing all input data.
   - `params`: Configuration dictionary containing options for clustering, normalization, and statistical tests.
   - `data_dict`: Dictionary grouping data by class.
   - `feature_columns`: Columns in the DataFrame identified as features for analysis.
   - `metadata_columns`: Columns identified as metadata, excluded from feature analysis.
   - `control_class`: The main control class used for statistical comparisons and normalization.

**Feature Sets and conditions**:
   - Supports varying feature sets (e.g., tierpsy_8, tierpsy_16, tierpsy_256, tierpsy_2k) and 1-3 conditions, 
   automatically adjusting configurations such as figure sizes and clustering.

"""

import os
import psutil
import copy
import pandas as pd
from pandas.plotting import parallel_coordinates
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import csv
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import multiprocessing as mp
import queue
from multiprocessing import Pool
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.computation import RQAComputation
from pyrqa.neighbourhood import FixedRadius
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.stats.multitest import multipletests
from scipy.stats import gaussian_kde, chi2
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import plotly.io as pio
import plotly.figure_factory as ff
import itertools
import random
from scipy.stats import wasserstein_distance
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, kruskal, friedmanchisquare, pearsonr, spearmanr, kendalltau, levene, bartlett, ks_2samp
from joblib import Parallel, delayed
import plotly.express as px
from scipy.stats import norm
import gc

class SSA:
    def __init__(self, data, params):
        """
        Initialize the SSA class with memory optimization in mind.
        """
        self.data = data
        self.params = params
        self.output_directory = os.path.join(params["csv_directory"], "Analysis")
        os.makedirs(self.output_directory, exist_ok=True)

        if 'class_column' not in params:
            raise KeyError("Missing required parameter 'class_column' in the params dictionary.")
        if 'main_control_name' not in params:
            raise KeyError("Missing required parameter 'main_control_name' in the params dictionary.")

        self.class_columns = params['class_column']
        self.control_class = params['main_control_name']
        self.combined_class_column = "combined_class"

        if len(self.class_columns) == 1:
            self.data[self.combined_class_column] = self.data[self.class_columns[0]]
        else:
            self.data[self.combined_class_column] = self.data[self.class_columns].astype(str).agg('_'.join, axis=1)

        self.metadata_columns = [col for col in data.columns if col.startswith('metadata_')]
        self.core_columns = [col for col in data.columns if col.startswith('core_')]
        self.feature_columns = [
            col for col in data.columns
            if col not in self.metadata_columns + self.core_columns + [self.combined_class_column]
        ]

        self.apply_well_filter()

        self.features = data[self.feature_columns]
        self.data_dict = {
            cls: df for cls, df in data.groupby(self.combined_class_column)
        }
        self.anomalies = None
        self.combined_data = None
        self.anomaly_scores_df = None
        self.ifa = self.IFA(self)  # Instantiate the IFA subclass
        
    def apply_well_filter(self):
        """
        Applies well-level filtering based on a specified feature and value range.
        Wells outside the specified range are dropped entirely.

        This uses `self.params["well_filter"]` for configuration.
        """
        well_filter = self.params.get("well_filter", {})
        if not well_filter.get("enabled", False):
            print(" Well filter is disabled. Skipping well filtering step.")
            return

        target_classes = well_filter.get("classes", [])
        filter_feature = well_filter.get("feature", None)
        value_range = well_filter.get("value_range", [])

        if not (target_classes and filter_feature and len(value_range) == 2):
            print("⚠️ Well filter is enabled, but configuration is incomplete. Skipping.")
            return

        min_value, max_value = value_range
        print(f" Applying well filter: {filter_feature} in range {min_value} - {max_value} for classes: {', '.join(target_classes)}")

        initial_count = len(self.data)

        # Filter only wells from target classes
        mask_target_classes = self.data[self.combined_class_column].isin(target_classes)

        # Keep only wells where the feature value is within the range
        mask_within_range = self.data[filter_feature].between(min_value, max_value)

        # Combine the two conditions (only applies filter to target classes)
        mask_keep = ~mask_target_classes | (mask_target_classes & mask_within_range)

        # Apply the mask to self.data
        self.data = self.data[mask_keep].reset_index(drop=True)

        final_count = len(self.data)
        print(f" Well filter applied. {initial_count - final_count} wells removed.")

    def remove_outliers_control(self):
        """
        Remove outliers from the control class using Isolation Forest and plot PCA of controls before
        and after outlier removal, with synchronized axis limits and a legend for clarity.

        Improvements:
        - Increased contamination parameter.
        - Use robust scaling for better outlier detection.
        - Visualize outliers explicitly in PCA plots.
        - Numeric-only fix to handle feature columns.
        """
        control_df = self.data_dict[self.control_class]

        # Select numeric feature-only data
        control_features = control_df[self.feature_columns].select_dtypes(include=[np.number]).fillna(0)

        # Robust scaling (optional for outlier detection)
        scaled_features = (control_features - control_features.median()) / (control_features.quantile(0.75) - control_features.quantile(0.25))

        # Perform PCA before outlier removal
        pca = PCA(n_components=2)
        pca_before = pca.fit_transform(scaled_features)
        explained_variance_before = pca.explained_variance_ratio_ * 100

        # Train Isolation Forest with adjusted parameters
        model = IsolationForest(
            n_estimators=500,
            contamination=self.params.get("outlier_contamination", 0.15),  
            random_state=42
        )
        is_outlier = model.fit_predict(scaled_features) == -1

        # Visualize outliers
        control_df['is_outlier'] = is_outlier  # Add outlier flag for visualization
        cleaned_control_df = control_df.loc[~is_outlier].copy()

        # Perform PCA after outlier removal
        scaled_cleaned_features = (cleaned_control_df[self.feature_columns]
                                .select_dtypes(include=[np.number])
                                .fillna(0) - control_features.median()) / (
                control_features.quantile(0.75) - control_features.quantile(0.25)
        )
        pca_after = pca.fit_transform(scaled_cleaned_features)
        explained_variance_after = pca.explained_variance_ratio_ * 100

        # Determine axis limits to ensure plots are comparable
        combined_pca = np.vstack([pca_before, pca_after])
        x_min, x_max = combined_pca[:, 0].min(), combined_pca[:, 0].max()
        y_min, y_max = combined_pca[:, 1].min(), combined_pca[:, 1].max()

        # Plot PCA before and after outlier removal
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
        plt.suptitle("PCA of Control Class Before and After Outlier Removal", fontsize=16)

        # PCA Before
        axes[0].scatter(
            pca_before[:, 0], pca_before[:, 1],
            c=is_outlier, cmap='coolwarm', alpha=0.7  # Reversed colormap for "warm-to-cool"
        )
        axes[0].set_xlim(x_min, x_max)
        axes[0].set_ylim(y_min, y_max)
        axes[0].set_title(f"Before Outlier Removal\n(PC1: {explained_variance_before[0]:.2f}%, PC2: {explained_variance_before[1]:.2f}%)")
        axes[0].set_xlabel("Principal Component 1")
        axes[0].set_ylabel("Principal Component 2")
        axes[0].legend(['Inliers', 'Outliers'], loc='upper right')

        # PCA After
        axes[1].scatter(
            pca_after[:, 0], pca_after[:, 1],
            alpha=0.7, c='steelblue'
        )
        axes[1].set_xlim(x_min, x_max)
        axes[1].set_ylim(y_min, y_max)
        axes[1].set_title(f"After Outlier Removal\n(PC1: {explained_variance_after[0]:.2f}%, PC2: {explained_variance_after[1]:.2f}%)")
        axes[1].set_xlabel("Principal Component 1")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        #plt.show()

        # Update the control data in the dictionary
        self.data_dict[self.control_class] = cleaned_control_df.drop(columns='is_outlier')

        del control_features, scaled_features, scaled_cleaned_features
        gc.collect()

    def normalize_features_by_control(self):
        """
        Returns a new dataset where all feature values are normalized relative to the control class.
        This does NOT modify self.data_dict; instead, it creates a temporary dataset for hierarchical clustering.

        Normalization: 
        - Uses already Z-normalized data to maintain correct scaling.
        - Subtracts the control mean from each feature (so control mean should be 0).
        """

        # Create a deep copy of self.data_dict to avoid modifying original data
        normalized_data_dict = {cls: df.copy() for cls, df in self.data_dict.items()}

        # Extract Z-normalized features from control
        control_df = normalized_data_dict[self.control_class][self.feature_columns]
        control_features = control_df.select_dtypes(include=[np.number])

        # Compute means from Z-normalized data
        control_mean = control_features.mean()

        # Apply normalization to each class separately
        for cls, df in normalized_data_dict.items():

            # Select numeric feature columns
            numeric_features = df[self.feature_columns].select_dtypes(include=[np.number])

            # Subtract the control mean directly from the numeric features
            df[self.feature_columns] = numeric_features - control_mean


        gc.collect()

        return normalized_data_dict  # Return a separate instance, leaving self.data_dict untouched


    def hierarchical_clustering(self):
        """
        Perform hierarchical clustering with:
        - A large pool of contrasting colors for class distinction.
        - Adaptive color assignment to handle thousands of classes.
        - Configurable row color mapping based on user-specified metadata column(s).
        - Row colors provided as a DataFrame if multiple columns are specified.
        - Ensures row_colors index matches grouped_df index for proper mapping.
        """
        # Configuration options
        clustering_method = self.params.get("clustering_method", "ward")
        distance_metric = self.params.get("distance_metric", "euclidean")
        font_scale = self.params.get("font_scale", 1.0)
        vmin = self.params.get("vmin", -2)
        vmax = self.params.get("vmax", 2)
        normalize_features = self.params.get("normalize_features", "None")
        row_color_columns = self.params.get("color_mapping_columns", ["metadata_treatment_1"])  

        if not self.params.get("avoid_normalization_HC", False):
            print("Normalizing features by control class...")
            normalized_data_dict = self.normalize_features_by_control()
        else:
            print("Skipping normalization for hierarchical clustering...")
            normalized_data_dict = self.data_dict

        # ---------------------------
        # Data Preparation
        # ---------------------------
        dataframes = []
        for class_name, df in normalized_data_dict.items():
            feature_df = df[self.feature_columns].select_dtypes(include=[float, int]).copy()
            feature_df['class'] = class_name
            dataframes.append(feature_df)

        combined_df = pd.concat(dataframes, ignore_index=True).dropna(how="all", axis=1).dropna(how="all", axis=0)

        feature_columns_only = [col for col in combined_df.columns if col != 'class']
        grouped_df = combined_df.groupby('class')[feature_columns_only].mean()
        
        # Linkage for clustering
        linkage_matrix = linkage(grouped_df, method=clustering_method, metric=distance_metric, optimal_ordering=True)

        # Automatically shapes the clustermap fig size
        n_classes = grouped_df.shape[0]
        n_features = grouped_df.shape[1]

        # Dynamically scale figure size
        height = max(6, min(0.3 * n_classes, 50))   # Cap max height
        width  = max(8, min(0.25 * n_features, 40)) # Cap max width

        # Dynamically increase width if labels are long
        max_label_len = max(len(label) for label in grouped_df.columns)
        extra_width = 0.06 * max_label_len  # ~0.06 inches per char

        width += extra_width

        figsize = (width, height)


        # ---------------------------
        # Row Color Mapping (Class-Level Mapping)
        # ---------------------------
        if not isinstance(row_color_columns, list):
            row_color_columns = [row_color_columns]

        metadata_columns = [col for col in row_color_columns if col in self.data.columns]

        if not metadata_columns:
            print("Warning: Specified row_color_columns not found in metadata. Proceeding without row colors.")
            row_colors = None
        else:
            # Group metadata by 'class' to ensure one value per class
            class_metadata = self.data.groupby(self.class_columns)[metadata_columns].first()

            # Ensure the index matches grouped_df
            class_metadata = class_metadata.reindex(grouped_df.index)

            # Check for existing *_color columns
            existing_color_columns = [col for col in class_metadata.columns if col.endswith('_color')]

            if existing_color_columns:
                # Use existing color columns
                row_colors = class_metadata[existing_color_columns].copy()
            else:
                # Generate new color columns
                #print("No existing color columns found. Generating new color mappings.")

                max_classes = 12000
                base_palette = (
                    sns.color_palette("tab20", 20)
                    + sns.color_palette("Set3", 12)
                    + sns.color_palette("Paired", 12)
                )
                large_palette = list(base_palette) * (max_classes // len(base_palette) + 1)
                random.shuffle(large_palette)
                hex_palette = [to_hex(color) for color in large_palette]

                # Generate colors for each metadata column
                for col in metadata_columns:
                    unique_values = class_metadata[col].unique()
                    color_map = {val: hex_palette[i % len(hex_palette)] for i, val in enumerate(unique_values)}
                    class_metadata[f"{col}_color"] = class_metadata[col].map(color_map)
                    self.data[f"{col}_color"] = self.data[col].map(color_map)  # Save mapping back to self.data

                row_colors = class_metadata[[f"{col}_color" for col in metadata_columns]]

            # Replace NaNs with white and ensure final alignment
            row_colors = row_colors.fillna("#FFFFFF")
            row_colors.index = grouped_df.index

        # ---------------------------
        # Heatmap and Clustermap
        # ---------------------------
        font_scale = min(1.2, max(0.5, 20 / n_classes))
        sns.set(font_scale=font_scale)

        g = sns.clustermap(
            grouped_df,
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
            figsize=figsize,
            yticklabels=True,
            xticklabels=True,
            row_linkage=linkage_matrix,
            row_colors=row_colors,
            dendrogram_ratio=(0.1, 0.05),
        )

        # Title and Aesthetics
        g.ax_heatmap.set_title(f"Hierarchical Clustering Heatmap - Method: {clustering_method}", fontsize=16, pad=50)

        # Adjust colorbar position
        label_width_estimate = min(0.1, max(0.03, 0.0065 * max_label_len))

        heatmap_bbox = g.ax_heatmap.get_position()
        g.cax.set_position([
            heatmap_bbox.x1 + label_width_estimate,
            heatmap_bbox.y0,
            0.015,
            heatmap_bbox.height * 0.5
        ])


        # Format labels
        # Update existing tick formatting
        for label in g.ax_heatmap.get_yticklabels():
            label.set_size(min(10, 200 / n_classes))
            label.set_rotation(0)

        for label in g.ax_heatmap.get_xticklabels():
            label.set_size(min(9, 160 / n_features))
            label.set_rotation(90)


        # Save plot and attributes
        output_dir = os.path.join(self.output_directory, "Hierarchical_Clustering")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "Hierarchical_Clustering_Heatmap.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Save attributes
        self.grouped_df = grouped_df
        self.data_matrix = grouped_df.values
        self.feature_names = grouped_df.columns.tolist()
        self.class_labels = grouped_df.index.tolist()
        self.linkage_matrix = linkage_matrix
        self.plot_config = {
            "figsize": figsize,
            "font_scale": font_scale,
            "vmin": vmin,
            "vmax": vmax,
        }

        # Save the ordered features
        self.feature_order = g.data2d.columns.tolist()

        del combined_df, grouped_df, linkage_matrix
        gc.collect()

    def interactive_hierarchical_clustering(self):
        """
        Generate an interactive hierarchical clustering heatmap with:
        - Y-axis dendrogram with colored branches.
        - Heatmap aligned with dendrogram leaf order.
        - Class labels inside the heatmap cells on the diagonal.
        - Clickable rows to copy core_well_id.
        - Hover information with class names, Z-scores, and additional metadata.
        """

        # ---------------------------
        # Extract necessary data
        # ---------------------------
        linkage_matrix = self.linkage_matrix
        grouped_df = self.grouped_df
        core_well_ids = grouped_df.index.tolist()

        hover_reach = self.params.get("hover_reach",10)
        hover_columns = self.params.get("HC_hover_columns", [])

        vmin = self.params.get("vmin", -2)
        vmax = self.params.get("vmax", 2)

        # Build metadata DataFrame from self.data_dict
        # ---------------------------
        metadata_list = []

        for class_name, df in self.data_dict.items():
            if not df.empty:
                # Select hover columns that exist in df
                selected_cols = [col for col in hover_columns if col in df.columns]
                meta_info = {}

                for col in selected_cols:
                    column_data = df[col].dropna()

                    if column_data.empty:
                        meta_info[col] = "N/A"
                    elif pd.api.types.is_numeric_dtype(column_data):
                        # For numeric data: show mean value (rounded to 3 decimals)
                        meta_info[col] = f"{column_data.mean():.3f}"
                    else:
                        # For non-numeric data: show top hover_reach most frequent values
                        top_values = column_data.value_counts().index[:hover_reach].tolist()
                        meta_info[col] = ", ".join(map(str, top_values))

                meta_info['class'] = class_name
                metadata_list.append(meta_info)

        # Create a DataFrame and convert to dictionary for quick access
        metadata_df = pd.DataFrame(metadata_list).set_index('class')
        metadata_dict = metadata_df.to_dict(orient='index')

        # ---------------------------
        #  Generate dendrogram data
        # ---------------------------
        dendro = dendrogram(
            linkage_matrix,
            orientation='left',
            no_plot=True,
            color_threshold=0.7 * max(linkage_matrix[:, 2])
        )

        leaf_order = dendro['leaves']
        ordered_df = grouped_df.iloc[leaf_order]
        ordered_core_well_ids = [core_well_ids[i] for i in leaf_order]
        dendro_colors = [mcolors.to_hex(color) for color in dendro['color_list']]

        # ---------------------------
        # Create dendrogram traces
        # ---------------------------
        dendro_traces = [
        go.Scatter(
            x=[-val for val in dcoord],  #  Invert X-coordinates for horizontal flip
            y=icoord,
            mode='lines',
            line=dict(color=color, width=2),
            hoverinfo='none',
            showlegend=False
        )
        for icoord, dcoord, color in zip(dendro['icoord'], dendro['dcoord'], dendro_colors)]


        # ---------------------------
        # Prepare heatmap hover text
        # ---------------------------
        text_annotations = np.full(ordered_df.shape, "", dtype=object)
        
        hover_text = []
        for i, class_label in enumerate(ordered_core_well_ids):
            meta_info = metadata_dict.get(class_label, {})
            meta_str = "<br>".join([f"<b>{col}:</b> {meta_info.get(col, 'N/A')}" for col in hover_columns])

            row_hover_data = []
            for j, feature in enumerate(ordered_df.columns):
                z_score = ordered_df.iloc[i, j]
                hover_info = (
                    f"<b>Class:</b> {class_label}<br>"
                    f"<b>Feature:</b> {feature}<br>"
                    f"<b>Z-score:</b> {z_score:.2f}<br>{meta_str}"
                )
                row_hover_data.append(hover_info)
            hover_text.append(row_hover_data)



        # ---------------------------
        # Create heatmap trace
        # ---------------------------
        heatmap = go.Heatmap(
            z=ordered_df.values,
            x=ordered_df.columns,
            y=ordered_core_well_ids,
            colorscale='RdBu_r',
            zmin=vmin, zmax=vmax,
            colorbar=dict(title='Z-score'),
            text=text_annotations,
            texttemplate="%{text}",
            hoverinfo="text",
            hovertext=hover_text,
            textfont={"color": "#004258", "size": 10},
        )

        # ---------------------------
        # 📏 Dynamic margin adjustment
        # ---------------------------
        max_label_length = max(len(str(label)) for label in ordered_core_well_ids)
        margin_left = max(100, max_label_length * 10)


        # ---------------------------
        # Create subplots layout
        # ---------------------------
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.2, 0.8],
            horizontal_spacing=0.005,
            shared_yaxes=True,
            specs=[[{'type': 'xy'}, {'type': 'heatmap'}]]
        )

        # Add dendrogram and heatmap
        for trace in dendro_traces:
            fig.add_trace(trace, row=1, col=1)
        fig.add_trace(heatmap, row=1, col=2)

        # ---------------------------
        # Layout configuration
        # ---------------------------
        n_classes = ordered_df.shape[0]
        n_features = ordered_df.shape[1]

        fig_width = max(1000, int(n_features * 25))
        fig_height = max(600, int(n_classes * 18))

        fig.update_layout(
            title_text="Interactive Hierarchical Clustermap",
            width=fig_width,
            height=fig_height,
            title_xanchor='center',
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=margin_left, r=50, t=50, b=100),
            hovermode='closest'
        )

        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(
            tickmode='array',
            tickvals=list(range(len(ordered_core_well_ids))),
            ticktext=ordered_core_well_ids,
            row=1, col=2
        )

        # ---------------------------
        # Add class labels inside the heatmap using go.Scatter
        # ---------------------------
        scatter_text_trace = go.Scatter(
            x=[ordered_df.columns[0]] * len(ordered_core_well_ids),  # Align labels to the first column
            y=ordered_core_well_ids,
            mode='text',
            text=ordered_core_well_ids,
            textposition='middle right',  # Align text inside heatmap cells
            textfont=dict(color="#0291C1", size=12, family="Palatino Linotype"),
            hoverinfo='skip',  # Prevents interfering with heatmap hover
            showlegend=False   # Hides from legend
        )

        fig.add_trace(scatter_text_trace, row=1, col=2)

        # ---------------------------
        # JavaScript for copy functionality
        # ---------------------------
        clipboard_js = """
        document.addEventListener('DOMContentLoaded', function() {
            var plot = document.getElementsByClassName('plotly-graph-div')[0];
            if (plot) {
                plot.on('plotly_click', function(data) {
                    if (data && data.points && data.points.length > 0) {
                        var coreWellID = data.points[0].y;
                        navigator.clipboard.writeText(coreWellID).then(function() {
                            alert('Copied to clipboard: ' + coreWellID);
                        }).catch(function(err) {
                            console.error('Clipboard copy failed: ', err);
                        });
                    }
                });
            }
        });
        """

        # ---------------------------
        # Save HTML with embedded JS
        # ---------------------------
        output_dir = os.path.join(self.output_directory, "Hierarchical_Clustering")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "Interactive_Hierarchical_Clustering_Heatmap.html")

        html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')
        html_with_js = html_content.replace('</body>', f'<script>{clipboard_js}</script></body>')

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_with_js)


    def z_normalize_features(self):
        """
        Perform Z-score normalization on numeric feature columns globally across all data.
        - Uses efficient NumPy vectorized operations instead of DataFrame `.apply()`
        - Batch plots features to reduce overhead
        """
        avoidplots = self.params.get("avoid_normalization_plots", False)
        print(" Starting Z-score normalization...")

        # Ensure only numeric feature columns are considered
        numeric_features = self.features.select_dtypes(include=[np.number])

        # Compute global mean & std **(NumPy Optimized)**
        global_means = numeric_features.mean().values
        global_stds = numeric_features.std().replace(0, 1).values  # Replace 0 std with 1

        # Extract feature names
        feature_names = numeric_features.columns.tolist()
        feature_idx_map = {col: i for i, col in enumerate(feature_names)}

        # Create output directory
        output_dir = os.path.join(self.output_directory, "Z_Normalization")
        os.makedirs(output_dir, exist_ok=True)

        # Track before normalization values efficiently (only required columns)
        before_normalization = {
            cls: df[feature_names].copy(deep=False) for cls, df in self.data_dict.items()
        }

        # **FAST NORMALIZATION - NumPy Vectorized**
        for cls, df in self.data_dict.items():
            df.loc[:, feature_names] = (df[feature_names] - global_means) / global_stds

        print(" Normalization completed. Generating plots...")

        if avoidplots == False:
            # **Batch Processing of Plots**
            batch_size = 10  # Number of features per figure (adjustable)
            num_batches = int(np.ceil(len(feature_names) / batch_size))

            for batch in range(num_batches):
                batch_features = feature_names[batch * batch_size : (batch + 1) * batch_size]
                if not batch_features:
                    break

                plt.figure(figsize=(16, 8))

                for i, feature in enumerate(batch_features, 1):
                    before_df = pd.concat(
                        [before_normalization[cls][[feature]].assign(Class=cls) for cls in self.data_dict.keys()]
                    )
                    after_df = pd.concat(
                        [self.data_dict[cls][[feature]].assign(Class=cls) for cls in self.data_dict.keys()]
                    )

                    if before_df[feature].isna().all() or after_df[feature].isna().all():
                        print(f"⚠️ Skipping {feature} due to all NaN values.")
                        continue

                    plt.subplot(2, len(batch_features), i)
                    sns.boxplot(x="Class", y=feature, data=before_df)
                    plt.xticks([], [])
                    plt.title(f"Before: {feature}")

                    plt.subplot(2, len(batch_features), i + len(batch_features))
                    sns.boxplot(x="Class", y=feature, data=after_df)
                    plt.xticks([], [])
                    plt.title(f"After: {feature}")

                plt.tight_layout()
                plot_path = os.path.join(output_dir, f"Z_Normalization_Batch_{batch+1}.png")
                plt.savefig(plot_path, dpi=300)
                plt.close()

            print(f" Saved boxplots in {output_dir}")
        else:
            print(f" Skipping Normalization plots...")

        #  **Save Normalized Data Efficiently**
        combined_df = pd.concat([df.assign(Class=cls) for cls, df in self.data_dict.items()], ignore_index=True)
        combined_df.to_csv(os.path.join(output_dir, "Z_Normalized_Data.csv"), index=False)

        #  **Memory Cleanup**
        del numeric_features, global_means, global_stds, before_normalization, combined_df
        gc.collect()

        print(" Z-score normalization completed successfully!")

    def statistical_analysis(self):
        """
        Perform statistical tests comparing each class to the control class based on user-specified parameters.
        Includes support for **blocked permutation tests** to account for day-to-day variance, with batch processing and multi-core execution.

        Returns:
        - pvalues_df (pd.DataFrame): DataFrame of p-values for each feature across classes.
        """
        # Ensure the 'Analysis' folder exists
        os.makedirs(self.output_directory, exist_ok=True)

        # Path for saving the p-values DataFrame
        output_file = os.path.join(self.output_directory, "Statistical_Analysis_pvalues.csv")

        control_features = self.data_dict[self.control_class][self.feature_columns]
        pvalues = {}

        # Read test type, multiple testing correction, and permutation settings
        stat_test_type = self.params.get("stat_test_type", "t-test")
        mt_correction = self.params.get("mt_correction", "False Discovery Rate (FDR) Correction")
        use_permutation = self.params.get("permutation_test", False)
        num_permutations = self.params.get("num_permutations", 500)
        num_cores = self.params.get("n_cores", -1)  # Use all available cores by default
        batch_size = self.params.get("batch_size", 10)  # Number of features per batch

        # Read day information for blocked permutations
        if "metadata_Day_of_run" not in self.data:
            raise ValueError("The parameter 'metadata_Day_of_run' must be specified for blocked permutation tests.")
        day_column = "metadata_Day_of_run"

       # Blocked permutation test function
        def blocked_permutation_test(stat_func, x, y, day_labels, num_perms):
            observed_stat = stat_func(x, y)
            unique_days = np.intersect1d(np.unique(day_labels), np.unique(day_labels))

            def single_permutation():
                permuted_stats = []
                for day in unique_days:
                    day_indices_x = np.where(day_labels[:len(x)] == day)[0]
                    day_indices_y = np.where(day_labels[len(x):] == day)[0]

                    if len(day_indices_x) == 0 or len(day_indices_y) == 0:
                        continue  # Skip days without data in both groups

                    combined = np.concatenate([x[day_indices_x], y[day_indices_y]])
                    np.random.shuffle(combined)

                    perm_x = combined[:len(day_indices_x)]
                    perm_y = combined[len(day_indices_x):]
                    permuted_stats.append(stat_func(perm_x, perm_y))

                return np.sum(permuted_stats) if permuted_stats else 0

            permuted_statistics = Parallel(n_jobs=num_cores)(
                delayed(single_permutation)() for _ in range(num_perms)
            )

            p_value = (np.sum(np.abs(permuted_statistics) >= np.abs(observed_stat)) + 1) / (num_perms + 1)
            return p_value

        # Test function mapper
        def get_test_function(test_name):
            return {
                "t-test": (
                    lambda x, y: ttest_ind(x, y, nan_policy='omit')[1],
                    lambda x, y, days: blocked_permutation_test(lambda a, b: np.abs(np.mean(a) - np.mean(b)), x, y, days, num_permutations)
                ),
                "Mann-Whitney": (
                    lambda x, y: mannwhitneyu(x, y, alternative='two-sided')[1],
                    lambda x, y, days: blocked_permutation_test(lambda a, b: np.abs(np.median(a) - np.median(b)), x, y, days, num_permutations)
                ),
                "Kruskal-Wallis": (
                    lambda x, y: kruskal(x, y)[1],
                    lambda x, y, days: blocked_permutation_test(lambda a, b: np.abs(np.median(a) - np.median(b)), x, y, days, num_permutations)
                ),
                "Wilcoxon": (
                    lambda x, y: wilcoxon(x, y)[1],
                    lambda x, y, days: blocked_permutation_test(lambda a, b: np.abs(np.median(a) - np.median(b)), x, y, days, num_permutations)
                ),
            }.get(test_name, (None, None))

        test_func, perm_test_func = get_test_function(stat_test_type)
        if test_func is None:
            raise ValueError(f"Unsupported statistical test: '{stat_test_type}'")

        feature_batches = [self.feature_columns[i:i + batch_size] for i in range(0, len(self.feature_columns), batch_size)]

        for cls, df in tqdm(self.data_dict.items(), desc="Statistical Tests"):
            if cls == self.control_class:
                continue

            class_features = df[self.feature_columns]
            day_labels = df[day_column].values if use_permutation else None
            pvalues[cls] = []

            for batch in feature_batches:
                if use_permutation:
                    batch_results = Parallel(n_jobs=num_cores)(
                        delayed(perm_test_func)(
                            control_features[feature].values,
                            class_features[feature].values,
                            np.concatenate([
                                self.data_dict[self.control_class][day_column].values,
                                df[day_column].values
                            ])
                        ) for feature in batch
                    )
                else:
                    batch_results = Parallel(n_jobs=num_cores)(
                        delayed(test_func)(control_features[feature].values, class_features[feature].values)
                        for feature in batch
                    )

                pvalues[cls].extend(batch_results)

        pvalues_df = pd.DataFrame(pvalues, index=self.feature_columns).T

        # Multiple testing correction
        correction_method = 'fdr_bh' if mt_correction.lower() == "fdr" else mt_correction.lower()
        for feature in pvalues_df.columns:
            valid_pvals = pvalues_df[feature].fillna(1.0)
            _, corrected, _, _ = multipletests(valid_pvals, alpha=0.05, method=correction_method)
            pvalues_df.loc[:, feature] = corrected

        pvalues_df.to_csv(output_file)
        print(f" Statistical analysis results saved at: {output_file}")

        self.pvalues_df = pvalues_df.T
        return pvalues_df


########################
    
    class IFA:
        """
        Isolation Forest Analysis (IFA) - Per-sample anomaly detection against randomized control iterations.
        """

        def __init__(self, ssa_instance):
            """
            Initialize IFA with the SSA instance.

            Parameters:
            - ssa_instance (SSA): Reference to the main SSA class.

            Supported Kernels:
            "linear": Classic PCA
            "rbf": Gaussian Kernel
            "poly": Polinomial Kernel
            "sigmoid": Hyperbolic Tangent Kernel
            "cosine": Cosine Kernel
            "anova": ANOVA Kernel
            "laplace": Laplace Kernel
            "chaos": Chaos Theory powered Analysis
            "hybrid": Chaos + PCA

            """
            self.ssa = ssa_instance
            self.output_directory = os.path.join(self.ssa.output_directory, "Isolation_Forest")
            os.makedirs(self.output_directory, exist_ok=True)
            self.params = self.ssa.params
            self.n_iterations = self.params.get("n_iterations", 100)
            self.contamination = self.params.get("contamination", 0.2)
            self.consensus_percentage = self.params.get("consensus_percentage", 80)
            self.bootstrap_set_size = self.params.get("bootstrap_set_size", 200)
            self.num_workers = self.params.get("n_cores", mp.cpu_count())
            self.censure = self.params.get("censure_level", 0.10)

             # Initialize data structures
            self.sample_anomaly_percentages = {}  # Ensure it always exists
            self.variance_metrics = {}  # Ensure variance tracking is initialized

        def run(self):
            """
            Execute Isolation Forest analysis with variance-based voting from full feature space.
            Now includes self-comparison of control class (control vs control) as a baseline.
            """
            print("Starting Isolation Forest Analysis...")

            #self.log_memory_usage("Before Loading Data")
            
            # Load class labels only (avoid full dataset copying)
            full_combined_data_labels = pd.concat([
                pd.DataFrame({"class_": [cls] * len(df)}) for cls, df in self.ssa.data_dict.items()
            ], ignore_index=True)

            #self.log_memory_usage("After Loading Data (Label Preparation)")

            # Prepare PCA-Reduced Data (avoid copying full data)
            combined_data, _ = self._prepare_data()
            pca_combined_data = self._apply_pca(combined_data)
            
            #self.log_memory_usage("After PCA Transformation")

            self.compute_control_iteration_means(pca_combined_data)

            target_classes = list(pca_combined_data["class_"].unique())
            if self.ssa.control_class not in target_classes:
                target_classes.append(self.ssa.control_class)

            # Use multiprocessing.Manager to create a shared queue
            with mp.Manager() as manager:
                result_queue = manager.Queue()

                # Start a separate process to write results to CSV
                writer_process = mp.Process(target=self._write_results_to_csv, args=(result_queue,))
                writer_process.start()

                #self.log_memory_usage("Before Multiprocessing Pool")

                # Parallel Processing using a process pool
                with mp.Pool(processes=self.num_workers) as pool:
                    results = [
                        pool.apply_async(
                            self._process_target_class, 
                            args=(index + 1, len(target_classes), target_class, 
                                full_combined_data_labels, pca_combined_data, result_queue)
                        ) 
                        for index, target_class in enumerate(tqdm(target_classes, desc="Processing Classes"))
                    ]
                    
                    pool.close()
                    pool.join()  # Ensures all processes are completed

                #self.log_memory_usage("After Multiprocessing Pool (Processing Done)")

                # Signal writer process to stop
                result_queue.put(None)
                writer_process.join()

            #self.log_memory_usage("Before Collecting Results")

            # Collect Results
            final_scores = {}
            class_anomaly_votes = {}
            anomaly_scores_list = []

            for result in results:
                output = result.get()
                if output:
                    target_class, final_vote_percentages, votes, scores = output
                    class_anomaly_votes[target_class] = votes
                    anomaly_scores_list.extend(scores)

            self.ssa.anomaly_scores_df = pd.DataFrame(anomaly_scores_list)

            # Compute Final Class-Level Anomaly Scores
            for cls, votes in class_anomaly_votes.items():
                if len(votes) > 2:
                    sorted_votes = sorted(votes)
                    lower_cutoff = max(1, int(len(sorted_votes) * self.censure))
                    upper_cutoff = max(1, int(len(sorted_votes) * (1 - self.censure)))
                    trimmed_votes = sorted_votes[lower_cutoff:upper_cutoff]
                else:
                    trimmed_votes = votes

                final_scores[cls] = np.mean(trimmed_votes) if trimmed_votes else 0

            self.ssa.anomalies = [cls for cls, score in final_scores.items() if score >= self.consensus_percentage]

            # Initialize empty dictionary before collecting results
            for result in results:
                output = result.get()
                if output:
                    target_class, final_vote_percentages, votes, scores = output
                    class_anomaly_votes[target_class] = votes
                    anomaly_scores_list.extend(scores)

                    self.sample_anomaly_percentages[target_class] = final_vote_percentages

            # Convert results to DataFrame
            self.ssa.anomaly_scores_df = pd.DataFrame(anomaly_scores_list)

            # Calculate p-values

            null_distribution = self.control_iteration_means
            control_std = np.std(null_distribution)
            control_mean = np.mean(null_distribution)

            p_value_list = []

            # Tunable parameters for logistic smoothing
            k = 1.0    # Slope (lower = smoother)
            z0 = 2.0   # Center (threshold z-score, e.g. ~significance)

            for cls, percentages in self.sample_anomaly_percentages.items():
                if cls == self.ssa.control_class:
                    continue

                if len(percentages) == 0:
                    continue

                target_mean = np.mean(percentages)
                observed_diff = abs(target_mean - control_mean)

                if control_std > 0:
                    z_score = observed_diff / control_std
                else:
                    z_score = 0

                # Logistic smooth p-value approximation
                logistic_p = 1 / (1 + np.exp(k * (z_score - z0)))

                p_value_list.append({
                    "class_": cls,
                    "mean_anomaly_vote_percentage": target_mean,
                    "p_value": logistic_p,
                    "z_score": z_score
                })

            # Convert to DataFrame
            pvalue_df = pd.DataFrame(p_value_list)

            # Apply FDR correction
            if not pvalue_df.empty:
                pvals = pvalue_df["p_value"]
                fdr_results = multipletests(pvals, alpha=0.05, method="fdr_bh")
                pvalue_df["fdr_corrected_p"] = fdr_results[1]
            else:
                print(" No p-values to correct.")

            # Store in SSA for later access
            self.ssa.pvalue_report_df = pvalue_df

            # Save to CSV
            output_path = os.path.join(self.output_directory, "IFA_PValue_Report.csv")
            pvalue_df.to_csv(output_path, index=False)
            print(f" P-value report saved to: {output_path}")



            # Plot results
            if len(self.ssa.data_dict) < 100:
                self._plot_anomaly_votes()
                self._plot_anomaly_votes_interactive()
            else:
                self._plot_anomaly_votes_interactive()

            print("Parallel Isolation Forest analysis completed.")


        def _process_target_class(self, index, total_classes, target_class, full_combined_data_labels, pca_combined_data, result_queue):
            """
            Process a single target class against control using Isolation Forest in parallel.
            Ensures that bootstrapping happens inside iterations for variability.
            """
            try:
                print(f"Processing class {index}/{total_classes}: {target_class}")

                # Compute control baseline once per class
                self._compute_control_baseline()

                # Extract only necessary control/target samples (avoid full dataset copies)
                control_samples = self.ssa.data_dict[self.ssa.control_class][self.ssa.feature_columns]
                target_samples = self.ssa.data_dict.get(target_class, pd.DataFrame())[self.ssa.feature_columns]

                if control_samples.empty or target_samples.empty:
                    print(f"⚠️ Skipping {target_class}: No valid control or target samples.")
                    return None

                class_anomaly_votes = []
                anomaly_scores_list = []
                final_vote_percentages = []

                for iteration in range(self.n_iterations):
                    self.log_memory_usage(f"Iteration {iteration} Start - {target_class}")

                    # Bootstrap fresh data each iteration to ensure variability
                    full_control_bootstrap, full_target_bootstrap = self._bootstrap_full_data(control_samples, target_samples)
                    control_bootstrap, target_bootstrap = self._bootstrap_sample(pca_combined_data, target_class)

                    self.log_memory_usage(f"After Bootstrapping Iteration {iteration} - {target_class}")

                    # Compute Variance Metrics
                    full_combined_features = pd.concat([full_control_bootstrap, full_target_bootstrap])
                    top3_variance, variance_vote = self._compute_variance_metrics(full_combined_features)

                    self.log_memory_usage(f"After Variance Computation Iteration {iteration} - {target_class}")

                    # Train Isolation Forest on bootstrapped control samples
                    model = self._fit_isolation_forest(control_bootstrap)

                    # Predict anomaly percentages from PCA-Reduced Data
                    anomaly_predictions = model.predict(target_bootstrap.drop(columns=["class_"]))
                    anomaly_percentage = (anomaly_predictions == -1).mean() * 100

                    # Compute shape vote (distributional difference)
                    mean_wd, shape_vote = self._compute_shape_metrics(full_control_bootstrap, full_target_bootstrap)


                    # Compute PCA Vote
                    pca_vote = 1 if anomaly_percentage >= self.consensus_percentage else 0

                    # Weighted Final Vote (90% PCA, 10% Variance)
                    variance_vote=variance_vote*1.2
                    shape_vote=shape_vote*2
                    final_vote_percentage = 0.7 * anomaly_percentage + 0.1 * variance_vote * 100 + 0.2 * shape_vote * 100
                    final_vote = 1 if final_vote_percentage >= self.consensus_percentage else 0

                    # Store results
                    final_vote_percentages.append(final_vote_percentage)
                    class_anomaly_votes.append(final_vote)

                    # Send data to result queue
                    result_queue.put((target_class, iteration, anomaly_percentage, pca_vote, top3_variance, variance_vote,shape_vote, final_vote))

                    self.log_memory_usage(f"After Model Prediction Iteration {iteration} - {target_class}")
                    anomaly_scores_list.append({
                        "class_": target_class,
                        "iteration": iteration + 1,
                        "anomaly_vote_percentage": final_vote_percentage
                    })

                    # Explicit memory cleanup
                    del full_control_bootstrap, full_target_bootstrap, control_bootstrap, target_bootstrap, model
                    gc.collect()

                    self.log_memory_usage(f"After Memory Cleanup Iteration {iteration} - {target_class}")

                self.log_memory_usage(f"End Processing {target_class}")
                return target_class, final_vote_percentages, class_anomaly_votes, anomaly_scores_list

            except Exception as e:
                print(f"Error processing {target_class}: {e}")
                return None

        def compute_control_iteration_means(self, pca_combined_data):
            """
            Computes the control vs. control iteration means to form the empirical null distribution.
            These means will be used for p-value calculations.
            
            Stores:
                self.control_iteration_means
            """
            print("Computing control-vs-control iteration means for null distribution...")

            control_samples = self.ssa.data_dict[self.ssa.control_class][self.ssa.feature_columns]
            if control_samples.empty:
                raise ValueError("No control samples found!")

            control_iteration_means = []

            for iteration in range(self.n_iterations):
                # Bootstrapped sample from PCA-reduced control data
                control_bootstrap, control_bootstrap_2 = self._bootstrap_sample(
                    pca_combined_data, self.ssa.control_class
                )

                # Fit Isolation Forest on one bootstrap
                model = self._fit_isolation_forest(control_bootstrap)

                # Predict on second bootstrap
                predictions = model.predict(control_bootstrap_2.drop(columns=["class_"]))
                anomaly_percentage = (predictions == -1).mean() * 100

                control_iteration_means.append(anomaly_percentage)

                if iteration % max(1, self.n_iterations // 10) == 0:
                    print(f"   Iteration {iteration+1}/{self.n_iterations} done.")

            self.control_iteration_means = control_iteration_means
            print(f" Completed control-vs-control iteration means. Stored {len(self.control_iteration_means)} values.")


        def _write_results_to_csv(self, result_queue):
            """
            Writes results from the multiprocessing queue to a CSV file.
            This prevents memory buildup by streaming data instead of keeping everything in RAM.
            """
            ballot_report_path = os.path.join(self.output_directory, "IFA_Ballot_Report.csv")

            with open(ballot_report_path, mode='w', newline='') as ballot_file:
                ballot_writer = csv.writer(ballot_file)
                ballot_writer.writerow([
                    "Target_Class", "Iteration", "PCA_Anomaly_Percentage", 
                    "PCA_Vote", "Target_Top3_Variance", "Variance_Vote","Shape_Vote" ,"Final_Vote"
                ])

                while True:
                    try:
                        data = result_queue.get()
                        if data is None:
                            break  # Stop when None is received
                        ballot_writer.writerow(data)
                    except queue.Empty:
                        continue  # Avoid blocking on empty queue

        def _prepare_data(self):
            combined_data = pd.concat([
                df[self.ssa.feature_columns].assign(class_=cls)
                for cls, df in self.ssa.data_dict.items()
            ], ignore_index=True)
            combined_data["anomaly_vote_percentage"] = 0  # Initialize column to avoid Seaborn error
            control_data = combined_data[combined_data["class_"] == self.ssa.control_class].drop(columns=["class_"])
            return combined_data, control_data
        
        def _prepare_full_data(self):
            """
            Prepare the full feature dataset from all classes for variance analysis.
            
            This method gathers the original (pre-PCA) feature data from `self.ssa.data_dict`,
            adds class labels, and returns the combined dataset. The resulting dataframe 
            retains all original features for variance-based analysis during bootstrapping.

            Returns:
            - full_combined_data (DataFrame): Combined dataset of all classes with original features.
            """
            full_combined_data = pd.concat([
                df[self.ssa.feature_columns].assign(class_=cls)
                for cls, df in self.ssa.data_dict.items()
            ], ignore_index=True)

            return full_combined_data

        
        def _bootstrap_full_data(self, control_samples, target_samples):
            """
            Bootstrap samples from full feature data with balanced sample sizes for variance analysis.
            
            Parameters:
            - control_samples (DataFrame): The control class feature data.
            - target_samples (DataFrame): The target class feature data.

            Returns:
            - full_control_bootstrap (DataFrame): Balanced control samples.
            - full_target_bootstrap (DataFrame): Balanced target samples.
            """
            # Find the minimum sample size
            min_samples = min(len(control_samples), len(target_samples))

            if min_samples == 0:
                print("⚠️ Warning: One of the classes has no samples! Returning empty dataframes.")
                return pd.DataFrame(columns=control_samples.columns), pd.DataFrame(columns=target_samples.columns)

            # Bootstrap to balance the sample sizes
            full_control_bootstrap = control_samples.sample(
                n=min_samples, replace=len(control_samples) < min_samples
            )
            full_target_bootstrap = target_samples.sample(
                n=min_samples, replace=len(target_samples) < min_samples
            )

            return full_control_bootstrap, full_target_bootstrap

        def _compute_control_baseline(self):
            """
            Compute baseline variance metrics from the control class only.
            Stores values for reference during variance vote calculation.
            """
            # Load control samples directly instead of using `full_combined_data`
            control_samples = pd.concat([
                df[self.ssa.feature_columns]
                for cls, df in self.ssa.data_dict.items()
                if cls == self.ssa.control_class
            ], ignore_index=True)

            if control_samples.empty:
                raise ValueError("No control samples found! Ensure control class exists in data.")

            pca = PCA()
            pca.fit(control_samples)

            self.control_top3_variance = sum(pca.explained_variance_ratio_[:3])
            self.control_components_for_70 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.50) + 1



        def _compute_rqa_metrics(self, X):
            """
            Compute RQA metrics for each sample using Recurrence Quantification Analysis (RQA).
            """
            
            rqa_metrics = []
            for i in range(len(X)):
                sample_series = X.iloc[i].values
                time_series = TimeSeries(sample_series, embedding_dimension=2, time_delay=1)
                settings = Settings(
                    time_series,
                    neighbourhood=FixedRadius(0.1)
                )
                computation = RQAComputation.create(settings)
                result = computation.run()

                rqa_metrics.append([
                    result.recurrence_rate,
                    result.determinism,
                    result.laminarity,
                    result.longest_diagonal_line,
                    result.longest_vertical_line,
                    result.trapping_time
                ])

            rqa_df = pd.DataFrame(rqa_metrics, columns=[
                "RQA_RR", "RQA_DET", "RQA_LAM", "RQA_LDL", "RQA_LVL", "RQA_TT"
            ])
            return rqa_df.values

        def _apply_pca(self, combined_data):
            """
            Apply dimensionality reduction with optional RQA metrics.
            Supports standard PCA, kernel PCA, chaos-only, and hybrid (PCA+RQA).
            """
            feature_columns = [col for col in combined_data.columns if col not in ["class_", "anomaly_vote_percentage"]]
            X = combined_data[feature_columns]

            # Determine kernel type directly from the params
            kernel_type = self.params.get("kernel", "linear")

            # Compute PCA components unless chaos is selected
            if kernel_type not in ["chaos", "hybrid"]:
                gamma = 1 / X.shape[1]

                # Custom kernels
                def anova_kernel(X, Y=None):
                    Y = X if Y is None else Y
                    return np.sum(np.exp(-gamma * (X[:, None] - Y) ** 2), axis=2)

                def laplace_kernel(X, Y=None):
                    Y = X if Y is None else Y
                    return np.exp(-np.sum(np.abs(X[:, None] - Y), axis=2) / gamma)

                kernel = {
                    "anova": anova_kernel,
                    "laplace": laplace_kernel
                }.get(kernel_type, kernel_type)

                # Apply Kernel PCA or standard PCA
                if kernel_type == "linear":
                    pca = PCA(n_components=3)
                    transformed_data = pca.fit_transform(X)
                else:
                    kpca = KernelPCA(n_components=3, kernel=kernel)
                    transformed_data = kpca.fit_transform(X)
            else:
                transformed_data = np.empty((X.shape[0], 0))

            # Compute RQA metrics if chaos or hybrid
            if kernel_type in ["chaos", "hybrid"]:
                rqa_metrics = self._compute_rqa_metrics(X)
                transformed_data = np.hstack([transformed_data, rqa_metrics]) if transformed_data.size else rqa_metrics

            # Prepare PCA DataFrame
            pca_df = pd.DataFrame(transformed_data, columns=[f"PC{i+1}" for i in range(transformed_data.shape[1])])
            pca_df["class_"] = combined_data["class_"].values
            pca_df["anomaly_vote_percentage"] = 0
            return pca_df

        def _bootstrap_sample(self, data, target_class):
            control_samples = data[data["class_"] == self.ssa.control_class]
            target_samples = data[data["class_"] == target_class]

            target_count = int(self.bootstrap_set_size * self.contamination)
            control_count = self.bootstrap_set_size - target_count

            # Bootstrap target samples with replacement if needed
            target_bootstrap = target_samples.sample(n=target_count, replace=True)
            control_bootstrap = control_samples.sample(n=control_count, replace=True)

            return control_bootstrap, target_bootstrap

        def _fit_isolation_forest(self, bootstrapped_data):
            model = IsolationForest(n_estimators=300, contamination=self.contamination, random_state=42, max_samples=0.7)
            model.fit(bootstrapped_data.drop(columns=["class_"]))
            return model

        def _plot_iteration_pca(
            self, 
            control_bootstrap, 
            target_bootstrap, 
            target_class, 
            iteration, 
            anomaly_predictions, 
            anomaly_percentage
        ):
            """
            Generate PCA scatter plot representing per-iteration anomaly detection results.
            Properly handles length differences due to target bootstrapping.
            """

            # Combine Control and Target Data
            combined_data = pd.concat([control_bootstrap, target_bootstrap])
            total_samples = len(combined_data)

            # Perform PCA
            pca = PCA(n_components=2)
            pca_transformed = pca.fit_transform(combined_data.drop(columns=["class_"], errors="ignore"))

            # Colors and Markers
            colors = []
            markers = []

            control_count = len(control_bootstrap)
            target_count = len(target_bootstrap)
            prediction_count = len(anomaly_predictions)

            # Controls: Dark Blue Squares
            colors.extend(["#0F303B"] * control_count)
            markers.extend(["s"] * control_count)

            # Targets: Use Anomaly Predictions
            # Align with `anomaly_predictions` and cap at target_count
            for i in range(target_count):
                if i < prediction_count:
                    if anomaly_predictions[i] == -1:
                        colors.append("#A3957F")  # Anomalous target (light brown)
                    else:
                        colors.append("#226176")  # Normal target (teal)
                    markers.append("o")  # Circle marker
                else:
                    # Fallback if predictions are short
                    colors.append("#888888")
                    markers.append("x")

            # Length Safety Fix
            min_length = min(len(pca_transformed), len(colors), len(markers))
            pca_transformed = pca_transformed[:min_length]
            colors = colors[:min_length]
            markers = markers[:min_length]

            # Plot PCA
            plt.figure(figsize=(10, 7))
            for i, (x, y) in enumerate(pca_transformed):
                plt.scatter(x, y, c=colors[i], marker=markers[i], alpha=0.7)

            plt.title(f"PCA Iteration {iteration} for {target_class} - Vote: {anomaly_percentage:.2f}%")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.grid(True)
            plt.tight_layout()

            # Save Plot
            class_dir = os.path.join(self.output_directory, "PCA_Plots", target_class)
            os.makedirs(class_dir, exist_ok=True)
            plot_path = os.path.join(class_dir, f"PCA_Iteration_{iteration}.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()
    

        def _plot_anomaly_votes(self):
            """
            Boxplot of anomaly vote % with p-value-aware coloring:
            - Gray: control class
            - Gold: significant (p < 0.05)
            - Steelblue gradient: not significant but scaled by smooth_score
            """

            plt.figure(figsize=(22, 10))

            # Prepare data for plotting
            plot_data = []
            for cls, percentages in self.sample_anomaly_percentages.items():
                if not isinstance(percentages, list) or len(percentages) == 0:
                    continue
                lower = int(len(percentages) * self.censure / 2)
                upper = int(len(percentages) * (1 - self.censure / 2))
                trimmed = sorted(percentages)[lower:upper]
                for val in trimmed:
                    plot_data.append({"class_": cls, "anomaly_vote_percentage": val})
            
            plot_df = pd.DataFrame(plot_data)
            class_means = plot_df.groupby("class_")["anomaly_vote_percentage"].mean()

            # Ensure p-value dataframe exists
            if not hasattr(self.ssa, "pvalue_report_df"):
                raise AttributeError("Run the p-value computation step first and store it as self.ssa.pvalue_report_df.")

            # Merge p-values and smooth scores
            pval_df = self.ssa.pvalue_report_df[["class_", "p_value", "z_score"]].copy()
            pval_df["is_significant"] = pval_df["p_value"] < 0.05

            merged_df = class_means.reset_index().merge(pval_df, on="class_", how="left")
            merged_df["is_significant"] = merged_df["is_significant"].fillna(False)
            merged_df["z_score"] = pd.to_numeric(merged_df["z_score"], errors='coerce').fillna(0.0)

            # Build color mapping
            norm = plt.Normalize(merged_df["z_score"].min(), merged_df["z_score"].max())
            cmap = plt.cm.viridis

            color_mapping = {}
            label_mapping = {}

            for _, row in merged_df.iterrows():
                cls = row["class_"]
                z = row["z_score"]
                sig = row["is_significant"]

                if cls == self.ssa.control_class:
                    color_mapping[cls] = "gray"
                elif sig:
                    # Significant → gradient by z
                    color_mapping[cls] = to_hex(cmap(norm(z)))
                else:
                    color_mapping[cls] = "steelblue"

                label_mapping[cls] = f"{cls}*" if sig else cls

            # Apply mappings to plot dataframe
            sorted_classes = merged_df.sort_values("anomaly_vote_percentage")["class_"].tolist()
            renamed_plot_df = plot_df.copy()
            renamed_plot_df["label"] = renamed_plot_df["class_"].map(label_mapping)

            # Build final color palette for seaborn with correct labels
            final_palette = {label_mapping[k]: v for k, v in color_mapping.items()}

            # Use hue to fix seaborn warning
            sns.boxplot(
                data=renamed_plot_df,
                x="label",
                y="anomaly_vote_percentage",
                hue="label",
                palette=final_palette,
                order=[label_mapping[c] for c in sorted_classes],
                legend=False,
                width=0.25
            )

            plt.axhline(y=self.consensus_percentage, color="red", linestyle="--", label="Consensus Threshold")
            plt.ylim(-10, 110)
            plt.xlabel("Class")
            plt.ylabel("Anomaly Vote Percentage")
            plt.title("Anomaly Votes (Significance & Smooth Score Coloring)")
            plt.xticks(rotation=90)
            plt.tight_layout()

            save_path = os.path.join(self.output_directory, "IFA_Anomaly_Votes_Significance_Smooth.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f" Saved significance-aware smoothed boxplot: {save_path}")



        def _compute_shape_metrics(self, control_samples, target_samples):
            """
            Compute a shape-based anomaly score using Wasserstein distances.
            Compares the shape of each feature's distribution between control and target.
            
            Returns:
            - mean_wasserstein (float): Averaged Wasserstein distance across features.
            - shape_vote (float): Normalized vote score in [0,1] based on relative shift.
            """
            feature_dists = []

            for feature in control_samples.columns:
                control_values = control_samples[feature].dropna()
                target_values = target_samples[feature].dropna()

                if len(control_values) > 5 and len(target_values) > 5:
                    wd = wasserstein_distance(control_values, target_values)
                    feature_dists.append(wd)

            if not feature_dists:
                return 0.0, 0.0

            mean_wasserstein = np.mean(feature_dists)

            # Normalize: divide by max(control range) + small epsilon
            control_range = np.maximum(
                control_samples.max() - control_samples.min(), 1e-8
            ).mean()

            shape_vote = np.clip(mean_wasserstein / (control_range * 1.5), 0, 1)
            return mean_wasserstein, shape_vote

        def _plot_anomaly_votes_interactive(self):
            """
            Generate an interactive scatter plot for anomaly vote percentages per class.
            - Supports 5,000+ classes.
            - Each class is represented as a dot (mean anomaly vote %).
            - Vertical error bars show SEM (Standard Error of the Mean).
            - Classes are ordered from highest to lowest anomaly percentage.
            - Saves CSV file with sorted class anomaly percentages.
            """

            print("Generating interactive anomaly vote plot...")

            # Convert self.sample_anomaly_percentages to a DataFrame
            plot_data = []
            for cls, percentages in self.sample_anomaly_percentages.items():
                lower = int(len(percentages) * self.censure / 2)
                upper = int(len(percentages) * (1 - self.censure / 2))
                trimmed_percentages = sorted(percentages)[lower:upper]

                if len(trimmed_percentages) > 1:
                    mean_value = np.mean(trimmed_percentages)
                    sem_value = np.std(trimmed_percentages) / np.sqrt(len(trimmed_percentages))  # Standard Error of Mean
                else:
                    mean_value = trimmed_percentages[0] if trimmed_percentages else 0
                    sem_value = 0

                plot_data.append({
                    "class_": cls,
                    "mean_anomaly_vote_percentage": mean_value,
                    "sem": sem_value
                })

            # Convert to DataFrame
            plot_df = pd.DataFrame(plot_data)

            # Sort by anomaly vote percentage (High → Low)
            plot_df = plot_df.sort_values(by="mean_anomaly_vote_percentage", ascending=False).reset_index(drop=True)

            # Save CSV File
            csv_path = os.path.join(self.output_directory, "IFA_Anomaly_Vote_Sorted.csv")
            plot_df.to_csv(csv_path, index=False)
            print(f"Saved sorted anomaly votes: {csv_path}")

            plot_df = plot_df.sort_values(by="mean_anomaly_vote_percentage", ascending=True).reset_index(drop=True)

            # Prepare Interactive Plot
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=plot_df["class_"],
                y=plot_df["mean_anomaly_vote_percentage"],
                mode='markers',
                marker=dict(size=6, color='steelblue', opacity=0.8),
                error_y=dict(type='data', array=plot_df["sem"], color='gray', thickness=1),
                hoverinfo="x+y",
                name="Anomaly Vote %"
            ))

            # Add Consensus Threshold Line
            fig.add_trace(go.Scatter(
                x=plot_df["class_"],
                y=[self.consensus_percentage] * len(plot_df),
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name="Consensus Threshold"
            ))

            # Layout Customization
            fig.update_layout(
                title="Interactive Class-Level Anomaly Vote Percentages",
                xaxis=dict(title="Class", tickangle=90, showticklabels=False),
                yaxis=dict(title="Anomaly Vote Percentage", range=[-10, 110]),
                template="plotly_white",
                margin=dict(l=50, r=20, t=50, b=100),
                hovermode="closest"
            )

            # Save Interactive HTML Plot
            plot_path = os.path.join(self.output_directory, "IFA_Anomaly_Votes_Interactive.html")
            fig.write_html(plot_path)
            print(f"Saved interactive anomaly vote plot: {plot_path}")

        def _compute_variance_metrics(self, combined_features):
            """
            Compute variance-based metrics using log-scaled deviation from control variance.
            """
            combined_features = combined_features.drop(columns=["class_"], errors="ignore")

            # Perform PCA on combined samples
            pca = PCA()
            pca.fit(combined_features)

            # Calculate top 3 variance explained for the target
            target_top3_variance = sum(pca.explained_variance_ratio_[:3])

            # Calculate variance vote using log-scaled deviation
            variance_factor = np.log1p(abs(target_top3_variance - self.control_top3_variance))
            variance_vote = np.clip(
                variance_factor / np.log1p(self.control_top3_variance * 2),
                0, 1
            )

            return target_top3_variance, variance_vote
        
        def _save_ballot_report(self, target_class, iteration, anomaly_percentage, pca_vote, target_top3_variance, variance_vote, final_vote):
            """
            Save vote details into the ballot report CSV with the new variance-only approach.
            """
            if not hasattr(self, "ballot_report_writer"):
                ballot_report_path = os.path.join(self.output_directory, "IFA_Ballot_Report.csv")
                self.ballot_report_file = open(ballot_report_path, mode='w', newline='')
                self.ballot_report_writer = csv.writer(self.ballot_report_file)
                self.ballot_report_writer.writerow([
                    "Target_Class", "Iteration", "PCA_Anomaly_Percentage", 
                    "PCA_Vote", "Target_Top3_Variance", 
                    "Variance_Vote", "Final_Vote"
                ])

            self.ballot_report_writer.writerow([
                target_class, iteration, anomaly_percentage, 
                pca_vote, round(target_top3_variance, 4), 
                round(variance_vote, 4), final_vote
            ])
        
        def log_memory_usage(self,label):
            """ Logs the current RAM usage with a label. """
            process = psutil.Process(os.getpid())
            mem_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            #print(f"[RAM] {label}: {mem_usage:.2f} MB")

########################    
    
    def analyze_hits_vs_anomalies(self):
        """
        Compare significant hits (from p-values) vs. anomalies (from IFA final votes).

        Steps:
        - Hits: Classes with significant features (p-value < alpha).
        - Anomalies: Classes with mean final votes ≥ consensus_percentage.
        - Result: CSV highlighting matches, hits-only, and anomalies-only.

        Returns:
        - pd.DataFrame: Combined comparison of hits vs anomalies.
        """
        output_filename = "HitsVsAnomalies.csv"
        anomaly_threshold = self.params.get("consensus_percentage", 85)
        significance_level = self.params.get("alpha", 0.05)

        # Safety Checks
        if not hasattr(self, 'pvalues_df'):
            raise AttributeError("Run statistical analysis (p-values) before running this method.")
        if not hasattr(self, 'anomaly_scores_df'):
            raise AttributeError("Run IFA before running this method.")

        # Identify Hits (Significant Features)
        hits = []
        significant_features_count = []
        for class_label in self.pvalues_df.columns:
            significant_features = (self.pvalues_df[class_label] < significance_level).sum()
            if significant_features > 0:
                hits.append(class_label)
                significant_features_count.append(significant_features)

        # Identify Anomalies from Final Votes
        mean_final_votes = self.anomaly_scores_df.groupby('class_')['anomaly_vote_percentage'].mean()
        anomalies = mean_final_votes[mean_final_votes >= anomaly_threshold].index.tolist()

        # Build DataFrames
        hits_df = pd.DataFrame({
            'class': hits,
            'mean_anomaly_vote': [mean_final_votes.get(cls, 0) for cls in hits],
            'significant_features_number': significant_features_count,
            'status': ['Match' if cls in anomalies else 'Only Hit' for cls in hits]
        })

        anomalies_df = pd.DataFrame({
            'class': anomalies,
            'mean_anomaly_vote': [mean_final_votes[cls] for cls in anomalies],
            'significant_features_number': [
                (self.pvalues_df[cls] < significance_level).sum() if cls in self.pvalues_df.columns else float('nan')
                for cls in anomalies
            ],
            'status': ['Match' if cls in hits else 'Only Anomaly' for cls in anomalies]
        })

        # Combine DataFrames
        hits_and_anomalies = hits_df[hits_df['class'].isin(anomalies)]
        only_anomalies = anomalies_df[~anomalies_df['class'].isin(hits)]
        only_hits = hits_df[~hits_df['class'].isin(anomalies)]

        final_df = pd.concat([
            hits_and_anomalies.sort_values('mean_anomaly_vote', ascending=False),  # Matches
            only_anomalies.sort_values('mean_anomaly_vote', ascending=False),     # Only Anomalies
            only_hits.sort_values('significant_features_number', ascending=False)  # Only Hits
        ]).reset_index(drop=True)

        # Save & Return
        output_path = os.path.join(self.output_directory, output_filename)
        final_df.to_csv(output_path, index=False)

        print(f"Hits vs Anomalies analysis saved to: {output_path}")
        return final_df

    def plot_pca(self):
        """
        Generates interactive 3D PCA plots with toggle for Class vs Class+Date.
        - Preserves clipboard copying.
        - Shows clean class legend initially.
        - When toggled, shows detailed Class+Date legend.
        """
        print("Generating interactive PCA plots...")

        shapes = ['circle', 'circle-open', 'cross', 'diamond','diamond-open', 'square', 'square-open', 'x']


        base_colors = (
            px.colors.qualitative.Set1 +
            px.colors.qualitative.Pastel +
            px.colors.qualitative.Dark24 +
            px.colors.qualitative.Light24
        )


        pca_output_dir = os.path.join(self.output_directory, "PCA")
        os.makedirs(pca_output_dir, exist_ok=True)

        # Prepare combined data from all classes
        data = pd.concat([
            df[self.feature_columns].assign(class_=cls, metadata_Day_of_run=df["metadata_Day_of_run"])
            for cls, df in self.data_dict.items()
        ], ignore_index=True)

        features = data[self.feature_columns].fillna(0)
        classes = data["class_"]
        run_dates = data["metadata_Day_of_run"].astype(str)
        core_well_ids = self.data["core_well_id"]

        # Compute PCA
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(features)
        explained_variance = pca.explained_variance_ratio_ * 100

        # Prepare DataFrame for plotting
        plot_df = pd.DataFrame(pca_result, columns=["PC1", "PC2", "PC3"])
        plot_df["Class"] = classes
        plot_df["Run Date"] = run_dates
        plot_df["CoreWellID"] = core_well_ids
        plot_df["Class+Date"] = plot_df["Class"] + "_" + plot_df["Run Date"]

        # Unique values
        unique_classes = sorted(classes.unique())
        unique_class_dates = sorted(plot_df["Class+Date"].unique())

        # Build (color, shape) combinations, exhausting all colors for each shape before moving to next shape
        color_shape_combinations = []
        for shape in shapes:
            for color in base_colors:
                color_shape_combinations.append((color, shape))

        # Assign color and shape to each class
        class_styles = {}
        for i, cls in enumerate(unique_classes):
            if cls == self.control_class:
                class_styles[cls] = {'color': 'black', 'shape': 'circle'}  # Control always black circle
            else:
                color, shape = color_shape_combinations[i % len(color_shape_combinations)]
                class_styles[cls] = {'color': color, 'shape': shape}

        class_date_colors = {cd: base_colors[i % len(base_colors)] for i, cd in enumerate(unique_class_dates)}

        # Create initial plot with class-based traces
        fig = go.Figure()

        traces_class = []
        traces_class_date = []

        for cls in unique_classes:
            df_class = plot_df[plot_df["Class"] == cls]
            traces_class.append(go.Scatter3d(
                x=df_class["PC1"], y=df_class["PC2"], z=df_class["PC3"],
                mode='markers',
                marker=dict(size=8 if cls == self.control_class else 7, 
                            color='black' if cls == self.control_class else class_styles[cls]['color'],
                            symbol=class_styles[cls]['shape']),
                name=cls,
                customdata=df_class[["CoreWellID"]],
                hovertemplate="<b>CoreWellID:</b> %{customdata[0]}<extra></extra>",
                visible=True
            ))

        for class_date in unique_class_dates:
            df_class_date = plot_df[plot_df["Class+Date"] == class_date]
            traces_class_date.append(go.Scatter3d(
                x=df_class_date["PC1"], y=df_class_date["PC2"], z=df_class_date["PC3"],
                mode='markers',
                marker=dict(size=5, color=class_date_colors[class_date]),
                name=class_date,
                customdata=df_class_date[["CoreWellID"]],
                hovertemplate="<b>CoreWellID:</b> %{customdata[0]}<extra></extra>",
                visible=False
            ))

        for trace in traces_class + traces_class_date:
            fig.add_trace(trace)

        # Add toggle button
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=1.2,
                    yanchor="top",
                    pad={"r": 10, "t": 10},
                    bgcolor="rgba(255,255,255,0.95)",
                    buttons=[
                        dict(
                            label="Color by Class",
                            method="update",
                            args=[{"visible": [True] * len(traces_class) + [False] * len(traces_class_date)}]
                        ),
                        dict(
                            label="Color by Class+Date",
                            method="update",
                            args=[{"visible": [False] * len(traces_class) + [True] * len(traces_class_date)}]
                        ),
                        dict(
                            label="Show All",
                            method="update",
                            args=[{"visible": [True] * (len(traces_class) + len(traces_class_date))}]
                        ),
                        dict(
                            label="Hide All",
                            method="update",
                            args=[{"visible": ["legendonly"] * (len(traces_class) + len(traces_class_date))}]
                        )
                    ]
                )
            ],
            margin=dict(t=80)  # add top margin to avoid clipping
        )


        # 3D plot axis & layout
        fig.update_layout(
            scene=dict(
                xaxis_title=f"PC1 ({explained_variance[0]:.2f}% variance)",
                yaxis_title=f"PC2 ({explained_variance[1]:.2f}% variance)",
                zaxis_title=f"PC3 ({explained_variance[2]:.2f}% variance)"
            )
        )

        # Clipboard copying functionality
        clipboard_js = """
        document.addEventListener('DOMContentLoaded', function() {
            var plot = document.getElementsByClassName('plotly-graph-div')[0];
            if (plot) {
                plot.on('plotly_click', function(data) {
                    if (data && data.points && data.points.length > 0) {
                        var coreWellID = data.points[0].customdata[0];
                        navigator.clipboard.writeText(coreWellID).then(function() {
                            var message = document.createElement('div');
                            message.innerText = 'Copied to clipboard: ' + coreWellID;
                            message.style.position = 'fixed';
                            message.style.top = '10px';
                            message.style.left = '50%';
                            message.style.transform = 'translateX(-50%)';
                            message.style.backgroundColor = 'black';
                            message.style.color = 'white';
                            message.style.padding = '10px';
                            message.style.borderRadius = '5px';
                            document.body.appendChild(message);
                            setTimeout(() => { message.remove(); }, 2000);
                        }).catch(function(err) {
                            console.error('Clipboard copy failed: ', err);
                        });
                    }
                });
            }
        });
        """

        # Save plot with clipboard support
        output_path = os.path.join(pca_output_dir, "PCA_Control_vs_All.html")
        html_content = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
        html_with_js = html_content.replace('</body>', f'<script>{clipboard_js}</script></body>')

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_with_js)

        print(f"Saved interactive PCA plot: {output_path}")

        # Load custom groups from params
        additional_class_lists = self.params.get("pca_custom_groups", [])

        for idx, class_list in enumerate(additional_class_lists):
            try:
                # Filter for only the desired classes
                filtered_df = plot_df[plot_df["Class"].isin(class_list)]

                if filtered_df.empty:
                    print(f"⚠️ Custom set {idx+1} has no matching data and will be skipped.")
                    continue

                unique_filtered_classes = sorted(filtered_df["Class"].unique())
                unique_filtered_class_dates = sorted(filtered_df["Class+Date"].unique())

                # Rebuild traces for this custom set
                fig = go.Figure()

                traces_class = []
                traces_class_date = []

                for cls in unique_filtered_classes:
                    df_class = filtered_df[filtered_df["Class"] == cls]
                    traces_class.append(go.Scatter3d(
                        x=df_class["PC1"], y=df_class["PC2"], z=df_class["PC3"],
                        mode='markers',
                        marker=dict(
                                    size=14 if cls == self.control_class else 8,
                                    color=class_styles.get(cls, {"color": "gray"})['color'],
                                    symbol=class_styles.get(cls, {"shape": "circle"})['shape']             
                                    ),
                        name=cls,
                        customdata=df_class[["CoreWellID"]],
                        hovertemplate="<b>CoreWellID:</b> %{customdata[0]}<extra></extra>",
                        visible=True
                    ))

                for class_date in unique_filtered_class_dates:
                    df_class_date = filtered_df[filtered_df["Class+Date"] == class_date]
                    traces_class_date.append(go.Scatter3d(
                        x=df_class_date["PC1"], y=df_class_date["PC2"], z=df_class_date["PC3"],
                        mode='markers',
                        marker=dict(size=5, color=class_date_colors.get(class_date, "gray")),
                        name=class_date,
                        customdata=df_class_date[["CoreWellID"]],
                        hovertemplate="<b>CoreWellID:</b> %{customdata[0]}<extra></extra>",
                        visible=False
                    ))

                for trace in traces_class + traces_class_date:
                    fig.add_trace(trace)

                # Add toggle button
                fig.update_layout(
                    updatemenus=[
                        dict(
                            type="buttons",
                            direction="right",
                            showactive=True,
                            x=0.01,        # distance from left
                            xanchor="left",
                            y=1.15,        # slightly above the plot
                            yanchor="top",
                            buttons=[
                                dict(
                                    label="Color by Class",
                                    method="update",
                                    args=[{"visible": [True] * len(traces_class) + [False] * len(traces_class_date)}]
                                ),
                                dict(
                                    label="Color by Class+Date",
                                    method="update",
                                    args=[{"visible": [False] * len(traces_class) + [True] * len(traces_class_date)}]
                                ),
                                dict(
                                    label="Show All",
                                    method="update",
                                    args=[{"visible": [True] * (len(traces_class) + len(traces_class_date))}]
                                ),
                                dict(
                                    label="Hide All",
                                    method="update",
                                    args=[{"visible": [False] * (len(traces_class) + len(traces_class_date))}]
                                )
                            ]
                        )
                    ]
                )

                # 3D plot axis
                fig.update_layout(
                    scene=dict(
                        xaxis_title=f"PC1 ({explained_variance[0]:.2f}% variance)",
                        yaxis_title=f"PC2 ({explained_variance[1]:.2f}% variance)",
                        zaxis_title=f"PC3 ({explained_variance[2]:.2f}% variance)"
                    )
                )

                # Clipboard support for custom set
                clipboard_js = """
                document.addEventListener('DOMContentLoaded', function() {
                    var plot = document.getElementsByClassName('plotly-graph-div')[0];
                    if (event.key === "h" || event.key === "H") {
                        Plotly.restyle(document.getElementsByClassName('plotly-graph-div')[0], "visible", [false]);
                        console.log("All classes hidden (Hotkey: H)");
                    }
                    if (plot) {
                        plot.on('plotly_click', function(data) {
                            if (data && data.points && data.points.length > 0) {
                                var coreWellID = data.points[0].customdata[0];
                                navigator.clipboard.writeText(coreWellID).then(function() {
                                    var message = document.createElement('div');
                                    message.innerText = 'Copied to clipboard: ' + coreWellID;
                                    message.style.position = 'fixed';
                                    message.style.top = '10px';
                                    message.style.left = '50%';
                                    message.style.transform = 'translateX(-50%)';
                                    message.style.backgroundColor = 'black';
                                    message.style.color = 'white';
                                    message.style.padding = '10px';
                                    message.style.borderRadius = '5px';
                                    document.body.appendChild(message);
                                    setTimeout(() => { message.remove(); }, 2000);
                                }).catch(function(err) {
                                    console.error('Clipboard copy failed: ', err);
                                });
                            }
                        });
                    }
                });
                """

                # Save custom set plot with embedded JS
                output_path = os.path.join(pca_output_dir, f"PCA_CustomSet_{idx+1}.html")
                html_content = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
                html_with_js = html_content.replace('</body>', f'<script>{clipboard_js}</script></body>')

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(html_with_js)

                print(f"Saved interactive PCA plot for Custom Set {idx+1}: {output_path}")

            except Exception as e:
                print(f"Error during PCA plot generation for Custom Set {idx+1}: {e}")

    def plot_ellipse_pca(self):

        output_dir = os.path.join(self.output_directory, "PCA")
        os.makedirs(output_dir, exist_ok=True)

        data = pd.concat([
            df[self.feature_columns].assign(class_=cls, core_well_id=df['core_well_id'])
            for cls, df in self.data_dict.items()
        ], ignore_index=True)

        features = data[self.feature_columns].fillna(0)
        classes = data["class_"]
        core_well_ids = data["core_well_id"]

        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(features)
        data["PC1"], data["PC2"], data["PC3"] = pca_result[:, 0], pca_result[:, 1], pca_result[:, 2]
        explained_variance = pca.explained_variance_ratio_ * 100

        confidence_levels = [0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95]

        class_colors = {
            cls: 'black' if cls == self.control_class else px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]
            for i, cls in enumerate(sorted(data['class_'].unique()))
        }

        def calculate_ellipse_and_density(df, pcx, pcy):
            points = np.column_stack([df[pcx], df[pcy]])
            mean = np.mean(points, axis=0)
            cov = np.cov(points, rowvar=False)

            kde = gaussian_kde(points.T)
            grid_x, grid_y = np.meshgrid(
                np.linspace(min(points[:, 0]), max(points[:, 0]), 100),
                np.linspace(min(points[:, 1]), max(points[:, 1]), 100)
            )
            grid_positions = np.vstack([grid_x.ravel(), grid_y.ravel()])
            densities = kde(grid_positions).reshape(grid_x.shape)

            ellipses = {}
            for level in confidence_levels:
                chi2_val = chi2.ppf(level, 2)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                width, height = 2 * np.sqrt(eigenvalues * chi2_val)
                ellipse = np.column_stack([np.cos(np.linspace(0, 2*np.pi, 100)), np.sin(np.linspace(0, 2*np.pi, 100))])
                ellipse = ellipse @ np.diag([width / 2, height / 2]) @ eigenvectors.T + mean
                ellipses[level] = ellipse

            return mean, ellipses, grid_x, grid_y, densities

        def extract_contour_polygons(grid_x, grid_y, densities, levels):
            fig, ax = plt.subplots()
            contour = ax.contourf(grid_x, grid_y, densities, levels=levels, cmap="Blues", alpha=0.7)
            plt.close(fig)

            polygons = []
            for collection in contour.collections:
                for path in collection.get_paths():
                    vertices = path.vertices
                    if len(vertices) > 3:  # Avoid degenerate paths
                        polygons.append(vertices)

            return polygons

        def create_pca_plot(pcx, pcy, filename):
            fig = go.Figure()
            density_traces, ellipse_traces = [], []
            legend_shown = set()

            for class_name, class_df in data.groupby("class_"):
                mean, ellipses, grid_x, grid_y, densities = calculate_ellipse_and_density(class_df, pcx, pcy)
                class_color = class_colors[class_name]

                fig.add_trace(go.Scatter(
                    x=[mean[0]], y=[mean[1]], mode='markers',
                    marker=dict(size=8, color=class_color, line=dict(width=1, color='black')),
                    name=class_name,
                    legendgroup=class_name,
                    showlegend=class_name not in legend_shown,
                    customdata=[class_df.iloc[0]["core_well_id"]],
                    hovertemplate=f"<b>Class:</b> {class_name}<br>{pcx}: {{x:.2f}}<br>{pcy}: {{y:.2f}}<br><extra></extra>"
                ))
                legend_shown.add(class_name)

                for level, ellipse in ellipses.items():
                    ellipse_traces.append(go.Scatter(
                        x=ellipse[:, 0], y=ellipse[:, 1], mode='lines',
                        line=dict(color=class_color, width=1),
                        legendgroup=class_name,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    fig.add_trace(ellipse_traces[-1])

                levels = np.linspace(densities.min(), densities.max(), 6)[1:]  # 5 levels
                polygons = extract_contour_polygons(grid_x, grid_y, densities, levels)

                max_opacity = 0.6
                min_opacity = 0.15
                opacities = np.linspace(min_opacity, max_opacity, len(polygons))  # Darker inner, lighter outer

                for idx, polygon in enumerate(polygons):
                    density_traces.append(go.Scatter(
                        x=polygon[:, 0],
                        y=polygon[:, 1],
                        fill='toself',
                        fillcolor=class_color,
                        opacity=opacities[idx],
                        line=dict(width=0),
                        legendgroup=class_name,
                        showlegend=False,
                        visible=False
                    ))
                    fig.add_trace(density_traces[-1])

            fig.update_layout(
                updatemenus=[dict(
                    type='buttons',
                    buttons=[
                        dict(label="Show Ellipses Only", method="update", args=[{"visible": [
                            t not in density_traces for t in fig.data
                        ]}]),
                        dict(label="Show Density Only", method="update", args=[{"visible": [
                            t in density_traces or (t not in density_traces and t not in ellipse_traces) for t in fig.data
                        ]}])
                    ],
                    direction='left',
                    showactive=True,
                    x=0.5, xanchor="center",
                    y=1.15, yanchor="top"
                )]
            )

            fig.update_layout(
                xaxis_title=f"{pcx} ({explained_variance[{'PC1':0, 'PC2':1, 'PC3':2}[pcx]]:.2f}% variance)",
                yaxis_title=f"{pcy} ({explained_variance[{'PC1':0, 'PC2':1, 'PC3':2}[pcy]]:.2f}% variance)",
                title_text=f"PCA: {pcx} vs {pcy} with Multi-Level Ellipses and Layered KDE Densities",
                legend=dict(title="Class", itemsizing="constant"),
                width=1000, height=800,
                plot_bgcolor='white', paper_bgcolor='white',
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False)
            )

            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w") as f:
                f.write(pio.to_html(fig))

            print(f" Saved: {output_path}")

        create_pca_plot("PC1", "PC2", "Density_PCA_PC1_vs_PC2.html")
        create_pca_plot("PC1", "PC3", "Density_PCA_PC1_vs_PC3.html")
        create_pca_plot("PC2", "PC3", "Density_PCA_PC2_vs_PC3.html")

        print(" All PCA plots saved successfully.")

    def plot_parallel_coordinates(self):
        """
        Generate parallel coordinates plots for the control class vs each target class.

        **Key Features:**
        - **Target and Control are now both represented using median + IQR.**
        - **No individual target lines—improves readability.**
        - **All features included (no p-value or extreme value filtering).**
        - **Condition-Specific Plots:** Features are grouped based on conditions.
        - **Organized Outputs:** Plots are saved in condition-specific folders.
        """

        # Verify required attributes
        if not hasattr(self, 'feature_order'):
            raise AttributeError("Hierarchical clustering must be performed before running this method.")

        # Configuration parameters
        conditions = self.params.get("conditions", ["bluelight"])  # Default condition list
        axis_range = self.params.get("PC_axis_range", [-3, 3])

        # Output directory setup
        parallel_coordinates_dir = os.path.join(self.output_directory, "Parallel_Coordinates")
        os.makedirs(parallel_coordinates_dir, exist_ok=True)

        # Use all features, including extreme values and non-significant ones
        all_features = self.feature_order.copy()

        # Define colors
        control_color = "steelblue"
        target_color = "darkkhaki"

        # Plot by condition
        for condition in conditions:
            # Select features relevant to the condition
            condition_features = [f for f in all_features if f.endswith(f"_{condition}")]

            if not condition_features:
                print(f"⚠️ No features found for condition: {condition}")
                continue

            # Prepare control statistics
            control_data = self.data_dict[self.control_class][condition_features]
            control_median = control_data.median()
            control_q1 = control_data.quantile(0.25)
            control_q3 = control_data.quantile(0.75)

            # Create condition-specific output folder
            condition_dir = os.path.join(parallel_coordinates_dir, f"{condition}")
            os.makedirs(condition_dir, exist_ok=True)

            # Plot control vs each target class
            for target_class, target_df in tqdm(self.data_dict.items(), desc=f"Plotting ({condition})"):
                if target_class == self.control_class:
                    continue

                target_data = target_df[condition_features]
                if target_data.empty or control_data.empty:
                    continue

                # Compute target class statistics (median + IQR)
                target_median = target_data.median()
                target_q1 = target_data.quantile(0.25)
                target_q3 = target_data.quantile(0.75)

                # Initialize the plot
                plt.figure(figsize=(20, 10))
                x = np.arange(len(condition_features))

                # Plot control median and IQR
                plt.fill_between(x, control_q1, control_q3, color=control_color, alpha=0.3, label=f"{self.control_class} IQR")
                plt.plot(x, control_median, color=control_color, linewidth=2, label=f"{self.control_class} Median")

                # Plot target median and IQR
                if len(target_data) <= 50:
                    for i in range(len(target_data)):
                        plt.plot(x, target_data.iloc[i], color="#956705", alpha=0.3, linewidth=1)
                    plt.plot(x, target_data.median(), color=target_color, linewidth=2, label=f"{target_class} Median")
                else:
                    target_median = target_data.median()
                    target_q1 = target_data.quantile(0.25)
                    target_q3 = target_data.quantile(0.75)
                    plt.fill_between(x, target_q1, target_q3, color=target_color, alpha=0.3, label=f"{target_class} IQR")
                    plt.plot(x, target_median, color=target_color, linewidth=2, label=f"{target_class} Median")

                # Dynamically set Y-axis limits based on the plotted data
                ymin = min(control_q1.min(), target_q1.min(), control_median.min(), target_median.min())
                ymax = max(control_q3.max(), target_q3.max(), control_median.max(), target_median.max())

                padding = 0.2 * (ymax - ymin) if ymax > ymin else 1

                # Plot customization
                plt.xticks(ticks=x, labels=condition_features, rotation=90, fontsize=8)
                plt.title(f"Parallel Coordinates: {target_class} vs {self.control_class} | {condition}")
                plt.xlabel("Features")
                plt.ylabel("Normalized Values")

                plt.ylim(ymin - padding, ymax + padding)

                plt.legend()
                plt.tight_layout()

                # Save the plot
                sanitized_target_class = target_class.replace(" ", "_").replace("/", "_")
                output_path = os.path.join(
                    condition_dir, f"Parallel_Coordinates_{self.control_class}_vs_{sanitized_target_class}.pdf"
                )
                plt.savefig(output_path, format='pdf')
                plt.close()

                print(f" Saved plot for {self.control_class} vs {target_class} at {output_path}")
                gc.collect()

        print(" All parallel coordinates plots generated.")

    def plot_boxplots(self, batch_size=50):
        """
        Generate boxplots for raw features by class, overlaid with scatter points colored by date.
        Classes are ordered as: main control → negative controls → other target classes → positive controls.

        Parameters:
        - batch_size (int): Number of features to process in each batch to manage memory usage.
        """

        # Ensure the output directory exists
        savepath = os.path.join(self.output_directory, "Box_Plots")
        os.makedirs(savepath, exist_ok=True)

        # Extract parameters from `self`
        main_control = self.params.get("main_control_name")
        positive_controls = self.params.get("positive_controls", [])
        negative_controls = self.params.get("negative_controls", [])
        feature_set = set(self.params.get("feature_set", []))  # Ensure it's a set for quick lookups

        class_colors = {
            self.control_class: "steelblue",
            **{cls: "#E982F6" for cls in self.params.get("highlight_classes", [])},
            **{cls: "#B30505" for cls in self.params.get("positive_controls", [])},
            **{cls: "#97D285" for cls in self.params.get("negative_controls", [])},
        }

        # Filter for all classes
        data=self.data
        filtered_data = {
            cls: df for cls, df in data.groupby(self.combined_class_column)
        }

        # Define class ordering: control → negative → others → positive
        class_order = (
            [main_control] +
            negative_controls +
            [cls for cls in filtered_data if cls not in [main_control] + negative_controls + positive_controls] +
            positive_controls
        )

        # Extract feature columns (exclude metadata and core columns)
        all_columns = list(next(iter(filtered_data.values())).columns)

        # Ensure feature is considered if **any** part of it matches a feature in feature_set
        feature_columns = [
            col for col in all_columns if not col.startswith(("metadata_", "core_","combined_class")) and 
            any(feature in col for feature in feature_set)
        ]

        if not feature_columns:
            print("No valid features found in feature set for boxplot generation.")
            return

        # Collect all unique dates for the legend
        unique_days = pd.concat([df["metadata_Day_of_run"] for df in filtered_data.values()]).unique()
        date_palette = dict(zip(unique_days, sns.color_palette("husl", len(unique_days))))

        # Process features in batches
        for batch_start in range(0, len(feature_columns), batch_size):
            batch_features = feature_columns[batch_start:batch_start + batch_size]
            print(f"Processing batch {batch_start // batch_size + 1}/{-(-len(feature_columns) // batch_size)}")

            for feature in batch_features:
                try:
                    # Combine data for the current feature
                    combined_data = []
                    for cls, df in filtered_data.items():
                        combined_data.append(pd.DataFrame({
                            "Class": cls,
                            "Value": df[feature],
                            "Date": df["metadata_Day_of_run"]
                        }))
                    combined_df = pd.concat(combined_data, ignore_index=True).dropna(subset=["Value"])

                    # Sort by class order
                    combined_df["Class"] = pd.Categorical(combined_df["Class"], categories=class_order, ordered=True)
                    combined_df.sort_values(by=["Class"], inplace=True)

                    # Generate the plot
                    plt.figure(figsize=(16, 8))

                    # Create boxplots
                    sns.boxplot(
                        data=combined_df,
                        x="Class",
                        y="Value",
                        palette={cls: class_colors.get(cls, "#C3C3C3") for cls in class_order},
                        width=0.3,  # Reduce box width
                        linewidth=1.5,
                        boxprops={"alpha": 0.3},  # Set transparency
                        legend=False,
                        hue="Class"
                    )

                    # Add scatter points
                    sns.stripplot(
                        data=combined_df,
                        x="Class",
                        y="Value",
                        hue="Date",
                        palette=date_palette,
                        jitter=True,
                        size=4,
                        alpha=0.7,  # Set transparency for scatter points
                        dodge=False  # Ensure points align with boxes
                    )

                    # Customize aesthetics
                    plt.title(f"Boxplot for {feature}", fontsize=16)
                    plt.xlabel("")  # Remove x-axis label
                    plt.ylabel(feature, fontsize=14)
                    plt.xticks(rotation=45)

                    # Add a segmented horizontal line for the control mean
                    if self.control_class in combined_df["Class"].unique():
                        control_mean = combined_df.loc[combined_df["Class"] == self.control_class, "Value"].mean()
                        plt.axhline(y=control_mean, color="#710A0A", linestyle="--", linewidth=1.5)  # Thin segmented line

                    # Show legend for dates
                    handles, labels = plt.gca().get_legend_handles_labels()
                    unique_labels = dict(zip(labels, handles))
                    plt.legend(unique_labels.values(), unique_labels.keys(), title="Date of Run", loc="upper right", fontsize=10, frameon=False)

                    # Save the plot
                    plot_path = os.path.join(savepath, f"Boxplot_{feature.replace('/', '_')}.png")
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=300, format="png")
                    plt.close()
                    print(f"Saved boxplot for feature '{feature}' at {plot_path}")

                except Exception as e:
                    print(f"Error generating plot for feature '{feature}': {e}")

            # Explicit garbage collection
            gc.collect()

    def interactive_waterfall_plot(self):
        """
        Generate per-feature interactive waterfall plots showing % change from control mean for each sample.
        - Uses vertical lines instead of bars.
        - Bins are assigned sequentially based on sorted DataFrame order.
        - Preserves global sorting of bins across all classes.
        - Saves one CSV per feature.
        - Saves interactive Plotly HTML per feature.
        - Stored in 'Analysis/Waterfall_Plots'.
        """
        output_dir = os.path.join(self.output_directory, "Waterfall_Plots")
        os.makedirs(output_dir, exist_ok=True)

        control_means = self.data_dict[self.control_class][self.feature_columns].mean()

        class_color_map = {
            cls: px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
            for i, cls in enumerate(self.data_dict.keys())
        }

        for feature in self.feature_columns:
            sample_data = []
            
            for cls, df in self.data_dict.items():
                if cls == self.control_class or df.empty:
                    continue
                for idx, value in df[feature].dropna().items():
                    percent_change = ((value - control_means[feature]) / control_means[feature]) * 100
                    sample_data.append({
                        'class': cls,
                        'percent_change': percent_change,
                        'sample_id': f'{cls}_{idx}'
                    })
            
            if not sample_data:
                continue
            
            # Convert to DataFrame and sort as in CSV output
            sample_df = pd.DataFrame(sample_data)
            sample_df = sample_df.sort_values(by='percent_change', ascending=False).reset_index(drop=True)
            
            # Assign bins sequentially based on order in sorted DataFrame
            sample_df['bin'] = sample_df.index

            # Save CSV file
            csv_path = os.path.join(output_dir, f"Waterfall_map_{feature}.csv")
            sample_df.to_csv(csv_path, index=False)
            print(f"Saved ordered sample data to: {csv_path}")

            fig = go.Figure()

            # **Single loop to plot all samples in sequential bin order**
            for _, row in sample_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['bin'], row['bin']],  # Vertical line at each bin position
                    y=[0, row['percent_change']],
                    mode='lines',
                    line=dict(color=class_color_map[row['class']], width=2),
                    name=row['class'],  # Class name for legend
                    hoverinfo='x+y+name'
                ))

            fig.update_layout(
                title=f"Interactive Waterfall Plot - {feature}",
                xaxis=dict(title="Binned Samples (Sorted)", showgrid=False, type="category"),
                yaxis=dict(title="% Change from Control Mean", showgrid=True,
                        range=[sample_df['percent_change'].min() - 5, sample_df['percent_change'].max() + 5]),
                template="plotly_white",
                height=600,
                margin=dict(l=50, r=20, t=50, b=100),
                hoverlabel=dict(font=dict(color='black'))
            )

            fig.write_html(os.path.join(output_dir, f"IWaterfall_plot_{feature}.html"))
            print(f"Saved Interactive Waterfall plot: {feature}")

        print("All Interactive Waterfall plots generated.")

    def plot_feature_distributions(self):
        """
        Generates feature distribution plots:
        - One plot per feature using raw data.
        - Black dots for class medians.
        - Silver-gray vertical lines for IQR.
        - Classes sorted by median value (ascending).
        - Horizontal dashed line for control median (#780B0B).
        - Saves sorting order in CSV.
        - Output: .svg files.
        """
        print("Generating feature distribution plots...")

        # Output directory
        output_dir = os.path.join(self.output_directory, "Feature_Distributions")
        os.makedirs(output_dir, exist_ok=True)

        for feature in self.feature_columns:
            feature_data = []

            for cls, df in self.data_dict.items():
                if df.empty or feature not in df:
                    continue
                
                values = df[feature].dropna()
                if values.empty:
                    continue
                
                median = np.median(values)
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)

                feature_data.append({
                    'class': cls,
                    'median': median,
                    'q1': q1,
                    'q3': q3
                })

            # Skip if no valid data
            if not feature_data:
                continue

            # Convert to DataFrame & Sort by Median
            feature_df = pd.DataFrame(feature_data)
            feature_df = feature_df.sort_values(by='median', ascending=True).reset_index(drop=True)

            # Save sorting order
            csv_path = os.path.join(output_dir, f"Feature_Order_{feature}.csv")
            feature_df.to_csv(csv_path, index=False)
            print(f"Saved sorting order: {csv_path}")

            # Extract control median
            control_median = feature_df[feature_df["class"] == self.control_class]["median"].values
            control_median = control_median[0] if len(control_median) > 0 else None

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, row in feature_df.iterrows():
                ax.plot([i, i], [row["q1"], row["q3"]], color="silver", linewidth=2)  # IQR Line
                ax.scatter(i, row["median"], color="black", zorder=3)  # Median Dot

            # Control median line
            if control_median is not None:
                ax.axhline(y=control_median, color="#780B0B", linestyle="dashed", linewidth=1.5, label="Control Median")

            # Labels & Formatting
            ax.set_xticks(range(len(feature_df)))
            ax.set_xticklabels(feature_df["class"], rotation=90, fontsize=8)
            ax.set_ylabel(f"{feature} Value")
            ax.set_title(f"Feature Distribution: {feature}")
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            
            # Save Plot
            plot_path = os.path.join(output_dir, f"Feature_Distribution_{feature}.svg")
            plt.tight_layout()
            plt.savefig(plot_path, format="svg", dpi=300)
            plt.close()

            print(f"Saved plot: {plot_path}")

        print("All feature distribution plots generated.")
    

    def generate_kde_plots(self):
        """
        Generates KDE plots for each feature comparing the control class vs every target class.
        - No clustering or PCA involved.
        - One plot per feature per target class.
        - Handles missing values and empty target classes gracefully.
        """

        print("Generating control vs. target KDE density plots...")

        control_class = self.control_class
        output_dir = os.path.join(self.output_directory, "KDE_Plots")
        os.makedirs(output_dir, exist_ok=True)

        for target_class, df_target in self.data_dict.items():
            if target_class == control_class:
                continue

            target_dir = os.path.join(output_dir, f"{control_class}_vs_{target_class.replace(' ', '_')}")
            os.makedirs(target_dir, exist_ok=True)

            control_data = self.data_dict[control_class]
            if control_data.empty or df_target.empty:
                print(f"Skipping {target_class} due to empty data.")
                continue

            for feature in self.feature_columns:
                control_vals = control_data[feature].dropna()
                target_vals = df_target[feature].dropna()

                if len(control_vals) < 2 or len(target_vals) < 2:
                    continue  # Skip poorly populated data

                try:
                    # Compute KDEs
                    x_min = min(control_vals.min(), target_vals.min())
                    x_max = max(control_vals.max(), target_vals.max())
                    x_vals = np.linspace(x_min, x_max, 300)

                    kde_control = gaussian_kde(control_vals)
                    kde_target = gaussian_kde(target_vals)

                    y_control = kde_control(x_vals)
                    y_target = kde_target(x_vals)

                    # Plot
                    plt.figure(figsize=(8, 5))
                    plt.plot(x_vals, y_control, label=control_class, color="steelblue")
                    plt.plot(x_vals, y_target, label=target_class, color="darkorange")
                    plt.fill_between(x_vals, y_control, alpha=0.3, color="steelblue")
                    plt.fill_between(x_vals, y_target, alpha=0.3, color="darkorange")
                    plt.title(f"KDE: {feature}")
                    plt.xlabel(feature)
                    plt.ylabel("Density")
                    plt.legend()
                    plt.tight_layout()

                    # Save
                    safe_feature = feature.replace("/", "_").replace(" ", "_")
                    plot_path = os.path.join(target_dir, f"KDE_{safe_feature}.png")
                    plt.savefig(plot_path, dpi=150)
                    plt.close()

                except Exception as e:
                    print(f"Error plotting feature {feature} for {target_class}: {e}")

            gc.collect()

        print(" KDE plots comparing control to target classes generated.")

    def interactive_3d_pca_with_class_kde(self):
        """
        Generates a 3D PCA plot with:
        - KDE isosurfaces per class (≥30 samples)
        - Each surface colored by class
        - No scatter points
        """
        print("Generating 3D PCA with KDE overlays by class (no scatter)...")

        output_dir = os.path.join(self.output_directory, "PCA")
        os.makedirs(output_dir, exist_ok=True)

        # Combine data
        combined_df = pd.concat([
            df[self.feature_columns].assign(class_name=cls)
            for cls, df in self.data_dict.items()
            if not df.empty
        ], ignore_index=True)

        # PCA
        features = combined_df[self.feature_columns].fillna(0)
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(features)
        explained = pca.explained_variance_ratio_ * 100

        plot_df = pd.DataFrame(pca_result, columns=["PC1", "PC2", "PC3"])
        plot_df["Class"] = combined_df["class_name"]

        unique_classes = sorted(plot_df["Class"].unique())
        color_palette = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
        class_colors = {cls: color_palette[i % len(color_palette)] for i, cls in enumerate(unique_classes)}

        fig = go.Figure()

        for cls in unique_classes:
            class_df = plot_df[plot_df["Class"] == cls]

            if len(class_df) < 30:
                continue

            try:
                xyz = class_df[["PC1", "PC2", "PC3"]].T.values
                kde = gaussian_kde(xyz, bw_method="scott")
                xmin, ymin, zmin = xyz.min(axis=1)
                xmax, ymax, zmax = xyz.max(axis=1)

                xg, yg, zg = np.mgrid[
                    xmin:xmax:30j,
                    ymin:ymax:30j,
                    zmin:zmax:30j
                ]
                positions = np.vstack([xg.ravel(), yg.ravel(), zg.ravel()])
                density = kde(positions).reshape(xg.shape)

                color = class_colors[cls]
                colorscale = [[0, color], [1, color]]

                fig.add_trace(go.Isosurface(
                    x=xg.flatten(),
                    y=yg.flatten(),
                    z=zg.flatten(),
                    value=density.flatten(),
                    isomin=np.percentile(density, 70),
                    isomax=density.max(),
                    surface_count=1,
                    opacity=0.3,
                    caps=dict(x_show=False, y_show=False, z_show=False),
                    showscale=False,
                    colorscale=colorscale,
                    name=cls
                ))

                # Add invisible marker for legend
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=8, color=color),
                    name=cls,
                    showlegend=True,
                    hoverinfo='skip'
                ))

            except Exception as e:
                print(f" Skipped KDE for {cls}: {e}")

        fig.update_layout(
            title="3D PCA with KDE Isosurfaces by Class",
            scene=dict(
                xaxis_title=f"PC1 ({explained[0]:.2f}%)",
                yaxis_title=f"PC2 ({explained[1]:.2f}%)",
                zaxis_title=f"PC3 ({explained[2]:.2f}%)",
            ),
            margin=dict(l=10, r=10, t=40, b=10),
            width=1000,
            height=800,
            legend=dict(itemsizing="constant")
        )

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(label="Show KDEs", method="update",
                            args=[{"visible": [True] * len(fig.data)}]),
                        dict(label="Hide All", method="update",
                            args=[{"visible": [False] * len(fig.data)}]),
                    ],
                    showactive=True,
                    x=0.0,
                    y=1.15,
                    xanchor='left',
                    yanchor='top'
                )
            ]
        )


        output_path = os.path.join(output_dir, "PCA_3D_KDE.html")
        fig.write_html(output_path, include_plotlyjs="cdn")
        print(f" Saved 3D KDE-only PCA plot: {output_path}")

        gc.collect()



        