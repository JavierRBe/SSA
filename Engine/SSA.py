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

**Data Organization**:
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
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import csv
import multiprocessing as mp
from multiprocessing import Pool
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from matplotlib.lines import Line2D
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
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, kruskal, friedmanchisquare, pearsonr, spearmanr, kendalltau, levene, bartlett, ks_2samp
from joblib import Parallel, delayed
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
        self.features = data[self.feature_columns]
        self.data_dict = {
            cls: df for cls, df in data.groupby(self.combined_class_column)
        }
        self.anomalies = None
        self.combined_data = None
        self.anomaly_scores_df = None
        self.ifa = self.IFA(self)  # Instantiate the IFA subclass
        

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
        Normalize all feature columns in the dataset by subtracting the control class mean.
        Only numeric feature columns are used for normalization.
        """
        # Ensure only numeric columns are used
        control_features = self.data_dict[self.control_class][self.feature_columns].select_dtypes(include=[np.number])
        control_mean = control_features.mean()

        for cls, df in self.data_dict.items():
            # Select numeric feature columns for the current class
            numeric_features = df[self.feature_columns].select_dtypes(include=[np.number])

            # Subtract the control mean from the numeric features
            normalized_features = numeric_features - control_mean

            # Update the original DataFrame with the normalized features
            df.update(normalized_features)
        
        del control_features
        gc.collect()

    def hierarchical_clustering(self):
        """
        Perform hierarchical clustering on the dataset using the specified configuration options.
        Ensures proper layout for title, legend, and dendrograms, and highlights specified class labels.
        """
        # NTS: you can pass a PandasDF as row_colors if you want multiple colour mapping
        # Read configuration options
        clustering_method = self.params.get("clustering_method", "ward")
        distance_metric = self.params.get("distance_metric", "euclidean")
        font_scale = self.params.get("font_scale", 1.0)
        vmin = self.params.get("vmin", -2)
        vmax = self.params.get("vmax", 2)
        normalize_features = self.params.get("normalize_features", "None")

        row_mapping = self.params.get("row_mapping", "None")
        row_mapping_cmap = self.params.get("row_mapping_cmap", "None")

        # Classes to highlight
        control_class = self.params.get("main_control_name", [])
        positive_class = self.params.get("positive_controls", [])
        negative_class = self.params.get("negative_controls", [])
        highlight_classes = self.params.get("highlight_classes", [])
        figsize=(12, 8)

        # Prepare the data
        dataframes = []
        labels = []

        for class_name, df in self.data_dict.items():
            # Select only numeric feature columns
            feature_df = df[self.feature_columns].select_dtypes(include=[np.number]).copy()
            feature_df['class'] = class_name
            labels.extend([class_name] * feature_df.shape[0])
            dataframes.append(feature_df)

        combined_df = pd.concat(dataframes, ignore_index=True)

        # Drop rows or columns with all NaN values
        combined_df = combined_df.dropna(how="all", axis=1).dropna(how="all", axis=0)

        # Extract feature columns and group by class
        feature_columns_only = [col for col in combined_df.columns if col != 'class']
        grouped_df = combined_df.groupby('class')[feature_columns_only].mean()

        # Optional normalization
        if normalize_features == "zscore":
            grouped_df = (grouped_df - grouped_df.mean()) / grouped_df.std()
        elif normalize_features == "minmax":
            grouped_df = (grouped_df - grouped_df.min()) / (grouped_df.max() - grouped_df.min())

        # Calculate linkage with optimal ordering
        linkage_matrix = linkage(
            grouped_df, method=clustering_method, metric=distance_metric, optimal_ordering=True
        )

        # Adjust dendrogram parameters dynamically
        avoid_dendrograms = self.params.get("avoid_dendrograms", False)
        row_linkage_param = linkage_matrix if not avoid_dendrograms else None
        col_dendrogram_ratio = 0.05 if not avoid_dendrograms else 0
        row_dendrogram_ratio = 0.1 if not avoid_dendrograms else 0

        if row_mapping:  
            row_colors = grouped_df.index.map(row_mapping_cmap)# Future upgrade: This could be a DF instead for multiple row mapping
        else:
            row_colors = None

        # Create the heatmap
        sns.set(font_scale=font_scale)
        g = sns.clustermap(
            grouped_df,
            vmin=vmin,
            vmax=vmax,
            figsize=figsize,
            yticklabels=True,
            xticklabels=True,
            dendrogram_ratio=(row_dendrogram_ratio, col_dendrogram_ratio),
            row_linkage=row_linkage_param,
            **({"row_colors": row_colors} if row_colors is not None else {}) 
        )

        # Adjust the title
        g.ax_heatmap.set_title(f'Hierarchical Clustering Heatmap - Method: {clustering_method}', fontsize=16, pad=50)

        # Dynamically adjust the colorbar legend position and dimensions
        longest_xlabel = max(
            g.ax_heatmap.get_xticklabels(),
            key=lambda lbl: lbl.get_window_extent(renderer=g.ax_heatmap.figure.canvas.get_renderer()).width
        )

        legend_width = longest_xlabel.get_window_extent(renderer=g.ax_heatmap.figure.canvas.get_renderer()).width / g.ax_heatmap.figure.dpi / g.ax_heatmap.figure.get_figwidth()

        heatmap_bbox = g.ax_heatmap.get_position()
        legend_x = heatmap_bbox.x1 + 0.02  # Slightly to the right of the y-axis labels
        legend_y = heatmap_bbox.y0 - 0.125  # Below the x-axis labels but not overlapping
        legend_height = 0.02  # Fixed height
        legend_width += 0.02  # Add a slight margin for clarity

        g.cax.set_position([legend_x, legend_y, 0.5*legend_width,5*legend_height ])

        # Adjust y-axis label font size and position
        for label in g.ax_heatmap.get_yticklabels():
            label.set_size(10)
            label.set_rotation(0)

        for label in g.ax_heatmap.get_xticklabels():
            label.set_size(8)
            label.set_rotation(90)

        # Highlight specific classes
        clustered_order = g.dendrogram_row.reordered_ind
        clustered_labels = [grouped_df.index[i] for i in clustered_order]

        for i, label in enumerate(clustered_labels):
            if label in highlight_classes:
                g.ax_heatmap.get_yticklabels()[i].set_color("#E982F6")
            elif label in negative_class:
                g.ax_heatmap.get_yticklabels()[i].set_color("#97D285")
            elif label in positive_class:
                g.ax_heatmap.get_yticklabels()[i].set_color("#B30505")
            elif label in control_class:
                g.ax_heatmap.get_yticklabels()[i].set_color("steelblue")

        # Save the ordered features
        self.feature_order = g.data2d.columns.tolist()

        # Save the plot
        output_dir = os.path.join(self.output_directory, "Hierarchical_Clustering")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "Hierarchical_Clustering_Heatmap.png")

        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Save necessary attributes for the interactive version
        self.grouped_df = grouped_df
        self.data_matrix = grouped_df.values
        self.feature_names = grouped_df.columns.tolist()
        self.class_labels = grouped_df.index.tolist()
        self.linkage_matrix = linkage_matrix
        self.plot_config = {
            "figsize": figsize,
            "font_scale": font_scale,
            "vmin": vmin,
            "vmax": vmax
        }
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
            colorscale='inferno',
            zmin=-2, zmax=2,
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
        fig.update_layout(
            title_text="Interactive Hierarchical Clustermap",
            width=1200,
            height=800,
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
        """
        # Ensure only numeric feature columns are considered
        numeric_features = self.features.select_dtypes(include=[np.number])
        
        # Calculate global mean and standard deviation for numeric features
        global_means = numeric_features.mean()
        global_stds = numeric_features.std()

        # Apply Z-score normalization to each class's feature DataFrame
        for cls, df in self.data_dict.items():
            df[self.feature_columns] = df[self.feature_columns].apply(
                lambda x: (x - global_means[x.name]) / global_stds[x.name] if x.name in numeric_features.columns else x
            )
        del numeric_features, global_means, global_stds
        gc.collect()

    def statistical_analysis(self):
        """
        Perform statistical tests comparing each class to the control class based on user-specified parameters.
        Includes support for **permutation tests** with batch processing and multi-core execution.
        
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
        stat_test_type = self.params.get("stat_test_type", "T-test")
        mt_correction = self.params.get("mt_correction", "False Discovery Rate (FDR) Correction")
        use_permutation = self.params.get("permutation_test", False)
        num_permutations = self.params.get("num_permutations", 500)
        num_cores = self.params.get("n_cores", -1)  # Use all available cores by default
        batch_size = self.params.get("batch_size", 10)  # Number of features per batch

        # Define permutation test function
        def permutation_test(func, x, y, num_perms):
            """
            Performs a permutation test using shuffled group labels.
            Runs in parallel for efficiency.

            Parameters:
            - func: Statistical test function.
            - x, y: Data samples.
            - num_perms: Number of permutations.

            Returns:
            - p-value from the permutation test.
            """
            observed_stat = func(x, y)
            combined = np.concatenate([x, y])

            def single_permutation():
                np.random.shuffle(combined)
                x_perm = combined[:len(x)]
                y_perm = combined[len(x):]
                return func(x_perm, y_perm) >= observed_stat

            # Run permutations in parallel
            count = sum(Parallel(n_jobs=num_cores)(delayed(single_permutation)() for _ in range(num_perms)))
            return count / num_perms

        # Define function mapping
        def get_test_function(test_name):
            """
            Returns the appropriate test function (regular or permutation-based).
            """
            test_map = {
                "t-test": (lambda x, y: ttest_ind(x, y, nan_policy='omit')[1], lambda x, y: permutation_test(lambda a, b: np.abs(np.mean(a) - np.mean(b)), x, y, num_permutations)),
                "Independent t-test": (lambda x, y: ttest_ind(x, y, nan_policy='omit')[1], lambda x, y: permutation_test(lambda a, b: np.abs(np.mean(a) - np.mean(b)), x, y, num_permutations)),
                "Paired t-test": (lambda x, y: ttest_rel(x, y, nan_policy='omit')[1], lambda x, y: permutation_test(lambda a, b: np.abs(np.mean(a - b)), x, y, num_permutations)),
                "Welch t-test": (lambda x, y: ttest_ind(x, y, equal_var=False, nan_policy='omit')[1], lambda x, y: permutation_test(lambda a, b: np.abs(np.mean(a) - np.mean(b)), x, y, num_permutations)),
                "Mann-Whitney": (lambda x, y: mannwhitneyu(x, y, alternative='two-sided')[1], lambda x, y: permutation_test(lambda a, b: np.abs(np.median(a) - np.median(b)), x, y, num_permutations)),
                "Wilcoxon": (lambda x, y: wilcoxon(x, y)[1], lambda x, y: permutation_test(lambda a, b: np.abs(np.median(a) - np.median(b)), x, y, num_permutations)),
                "Kruskal-Wallis": (lambda x, y: kruskal(x, y)[1], lambda x, y: permutation_test(lambda a, b: np.abs(np.median(a) - np.median(b)), x, y, num_permutations)),
                "Friedman": (lambda x, y: friedmanchisquare(x, y)[1], lambda x, y: permutation_test(lambda a, b: np.abs(np.median(a) - np.median(b)), x, y, num_permutations)),
                "Pearson": (lambda x, y: pearsonr(x, y)[1], lambda x, y: permutation_test(lambda a, b: np.abs(pearsonr(a, b)[0]), x, y, num_permutations)),
                "Spearman": (lambda x, y: spearmanr(x, y)[1], lambda x, y: permutation_test(lambda a, b: np.abs(spearmanr(a, b)[0]), x, y, num_permutations)),
                "Kendall": (lambda x, y: kendalltau(x, y)[1], lambda x, y: permutation_test(lambda a, b: np.abs(kendalltau(a, b)[0]), x, y, num_permutations)),
                "Levene": (lambda x, y: levene(x, y)[1], lambda x, y: permutation_test(lambda a, b: np.abs(np.var(a) - np.var(b)), x, y, num_permutations)),
                "Bartlett": (lambda x, y: bartlett(x, y)[1], lambda x, y: permutation_test(lambda a, b: np.abs(np.var(a) - np.var(b)), x, y, num_permutations)),
                "Kolmogorov-Smirnov": (lambda x, y: ks_2samp(x, y)[1], lambda x, y: permutation_test(lambda a, b: np.abs(ks_2samp(a, b)[0]), x, y, num_permutations)),
            }

            return test_map.get(test_name, None)

        test_func, perm_test_func = get_test_function(stat_test_type)
        if test_func is None:
            raise ValueError(f"Statistical test '{stat_test_type}' is not supported.")

        # Batch processing of statistical tests
        feature_batches = [self.feature_columns[i:i+batch_size] for i in range(0, len(self.feature_columns), batch_size)]

        for cls, df in tqdm(self.data_dict.items(), desc="Statistical Tests"):
            if cls == self.control_class:
                continue

            class_features = df[self.feature_columns]
            pvalues[cls] = []

            for batch in feature_batches:
                pvalues[cls].extend(
                    Parallel(n_jobs=num_cores)(
                        delayed(perm_test_func if use_permutation else test_func)(control_features[f], class_features[f]) for f in batch
                    )
                )

        pvalues_df = pd.DataFrame(pvalues, index=self.feature_columns).T

        # Apply multiple testing correction
        correction_method = 'fdr_bh' if mt_correction == "fdr" else mt_correction
        for feature in pvalues_df.columns:
            valid_pvals = pvalues_df[feature].dropna()
            _, corrected_pvals, _, _ = multipletests(valid_pvals, alpha=0.05, method=correction_method)
            pvalues_df.loc[valid_pvals.index, feature] = corrected_pvals

        # Save the results
        pvalues_df.to_csv(output_file)
        print(f"Statistical analysis results saved to {output_file}")

        # Store the results in self for later use
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
            self.n_iterations = self.params.get("n_iterations", 20)
            self.contamination = self.params.get("contamination", 0.15)
            self.consensus_percentage = self.params.get("consensus_percentage", 85)
            self.bootstrap_set_size = self.params.get("bootstrap_set_size", 200)
            self.num_workers = self.params.get("n_cores", mp.cpu_count())
            self.censure = self.params.get("censure_level", 0.30)

             # Initialize data structures
            self.sample_anomaly_percentages = {}  # Ensure it always exists
            self.variance_metrics = {}  # Ensure variance tracking is initialized

        def run(self):
            """
            Execute Isolation Forest analysis with variance-based voting from full feature space.
            Now includes self-comparison of control class (control vs control) as a baseline.
            """
            print("Starting per-sample Isolation Forest anomaly detection with control self-comparison...")

            # Prepare Full Feature Data for Variance Analysis
            self.full_combined_data = self._prepare_full_data()

            # Prepare PCA-Reduced Data for Isolation Forest
            combined_data, control_data = self._prepare_data()
            pca_combined_data = self._apply_pca(combined_data)

            # Include control class in target list for self-comparison
            target_classes = list(pca_combined_data["class_"].unique())
            if self.ssa.control_class not in target_classes:
                target_classes.append(self.ssa.control_class)

            class_anomaly_votes = {cls: [] for cls in target_classes}

            # Initialize anomaly percentages storage (INCLUDING CONTROL CLASS)
            self.sample_anomaly_percentages = {
                cls: [] for cls in pca_combined_data["class_"].unique()}


            # Compute Control Baseline
            self._compute_control_baseline()

            for target_class in tqdm(target_classes, desc="Performing Isolation Forest Analysis"):
                for iteration in range(self.n_iterations):
                    
                    # Bootstrap from Full Data for Variance Analysis
                    full_control_bootstrap, full_target_bootstrap = self._bootstrap_full_data(target_class)

                    # Compute Variance Metrics (from Full Feature Data)
                    full_combined_features = pd.concat([full_control_bootstrap, full_target_bootstrap])
                    top3_variance, variance_vote = self._compute_variance_metrics(full_combined_features)

                    # Bootstrap from PCA Data for IFA
                    control_bootstrap, target_bootstrap = self._bootstrap_sample(pca_combined_data, target_class)
                    model = self._fit_isolation_forest(control_bootstrap)

                    # Predict anomaly percentages from PCA-Reduced Data
                    anomaly_predictions = model.predict(target_bootstrap.drop(columns=["class_"]))
                    anomaly_percentage = (anomaly_predictions == -1).mean() * 100

                    # Compute PCA Vote
                    pca_vote = 1 if anomaly_percentage >= self.consensus_percentage else 0

                    # Weighted Final Vote (35% Variance, 65% PCA)
                    final_vote_percentage = 0.65 * anomaly_percentage + 0.35 * variance_vote * 100
                    final_vote = 1 if final_vote_percentage >= self.consensus_percentage else 0

                    # Store Anomaly Percentage for Plotting
                    self.sample_anomaly_percentages[target_class].append(final_vote_percentage)

                    
                    class_anomaly_votes[target_class].append(final_vote)

                    # Save Iteration Metrics for Control Self-Comparison
                    self._save_ballot_report(
                        target_class, iteration, anomaly_percentage, 
                        pca_vote, top3_variance, 
                        variance_vote, final_vote
                    )

                    # Store Metrics
                    if target_class not in self.variance_metrics:
                        self.variance_metrics[target_class] = {
                            "top3_variance": []
                        }
                    self.variance_metrics[target_class]["top3_variance"].append(top3_variance)

                    self._plot_iteration_pca(
                        full_control_bootstrap,  # Use full feature space
                        full_target_bootstrap,
                        target_class,
                        iteration,
                        anomaly_predictions,
                        anomaly_percentage
                    )

                    # Create DataFrame from collected iteration votes
                    anomaly_scores_list = []
                    for target_class, votes in self.sample_anomaly_percentages.items():
                        for iteration, vote in enumerate(votes, start=1):
                            anomaly_scores_list.append({
                                "class_": target_class,
                                "iteration": iteration,
                                "anomaly_vote_percentage": vote
                            })

            
            # Save the full iteration-level vote data
            self.ssa.anomaly_scores_df = pd.DataFrame(anomaly_scores_list)


            # Compute final class anomaly scores using the lowest 30% of model outcomes
            # Compute final class anomaly scores using double-sided censure
            final_scores = {}
            for cls, votes in class_anomaly_votes.items():
                if len(votes) > 2:
                    sorted_votes = sorted(votes)
                    lower_cutoff = max(1, int(len(sorted_votes) * self.censure))
                    upper_cutoff = max(1, int(len(sorted_votes) * (1 - self.censure)))
                    
                    # Trim from both ends
                    trimmed_votes = sorted_votes[lower_cutoff:upper_cutoff]
                else:
                    # Fallback if too few votes
                    trimmed_votes = votes

                # Compute mean of trimmed votes
                final_scores[cls] = np.mean(trimmed_votes) if trimmed_votes else 0


            # Determine anomalies based on filtered scores
            anomalies = [cls for cls, score in final_scores.items() if score >= self.consensus_percentage]

            self.ssa.anomalies = anomalies

            # Plot Anomaly Votes after completion
            self._plot_anomaly_votes()

            print("Variance-enhanced Isolation Forest analysis with self-comparison completed.")

            return anomalies, pca_combined_data


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

        
        def _bootstrap_full_data(self, target_class):
            """
            Bootstrap samples from full feature data with balanced sample sizes for variance analysis.
            
            Balances control and target samples by using the same number of samples 
            (bootstrap if necessary).
            
            Returns:
            - full_control_bootstrap (DataFrame): Balanced control samples.
            - full_target_bootstrap (DataFrame): Balanced target samples.
            """
            full_control_samples = self.full_combined_data[
                self.full_combined_data["class_"] == self.ssa.control_class
            ]
            full_target_samples = self.full_combined_data[
                self.full_combined_data["class_"] == target_class
            ]

            # Find the minimum sample size
            min_samples = min(len(full_control_samples), len(full_target_samples))

            # Bootstrap to balance the sample sizes
            full_control_bootstrap = full_control_samples.sample(
                n=min_samples, replace=len(full_control_samples) < min_samples
            )
            full_target_bootstrap = full_target_samples.sample(
                n=min_samples, replace=len(full_target_samples) < min_samples
            )

            return full_control_bootstrap, full_target_bootstrap


        def _compute_control_baseline(self):
            """
            Compute baseline variance metrics from the control class only.
            Stores values for reference during variance vote calculation.
            """
            control_samples = self.full_combined_data[self.full_combined_data["class_"] == self.ssa.control_class].drop(columns=["class_"], errors="ignore")

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
            model = IsolationForest(n_estimators=500, contamination=self.contamination, random_state=42, max_samples=0.9)
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
            Generate boxplots for the distribution of anomaly percentages per class using data from self.sample_anomaly_percentages.
            - Classes are ordered by their mean anomaly score.
            - Colors are assigned based on whether the mean exceeds the consensus threshold.
            - Boxplot width is reduced to 20% of the default width.
            """
            plt.figure(figsize=(20, 10))

            # Convert the anomaly percentages dictionary into a DataFrame with trimmed votes
            plot_data = []
            for cls, percentages in self.sample_anomaly_percentages.items():
                # Trim to the lowest 30% of values
                trimmed_percentages = sorted(percentages)[:max(1, int(len(percentages) * self.censure))]
                for value in trimmed_percentages:
                    plot_data.append({"class_": cls, "anomaly_vote_percentage": value})


            plot_df = pd.DataFrame(plot_data)

            # Generate the color mapping based on class mean scores
            class_means = plot_df.groupby("class_")["anomaly_vote_percentage"].mean()
            sorted_classes = class_means.sort_values().index.tolist()

            color_mapping = {}
            for cls in sorted_classes:
                mean_score = class_means[cls]
                if cls == self.ssa.control_class:
                    color_mapping[cls] = "gray"
                elif mean_score >= self.consensus_percentage:
                    color_mapping[cls] = "darkkhaki"
                else:
                    color_mapping[cls] = "steelblue"

            # Plot the boxplot using the anomaly percentage data
            sns.boxplot(
                x="class_", 
                y="anomaly_vote_percentage", 
                data=plot_df, 
                order=sorted_classes, 
                palette=color_mapping,  # Dictionary-based color assignment
                width=0.2,  # 20% of original width
                hue="class_",
                legend=False
            )

            # Add a horizontal line for the consensus threshold
            plt.axhline(y=self.consensus_percentage, color="red", linestyle="--", label="Consensus Threshold")
            plt.ylim(-10, 110)
            plt.title("Class-Level Anomaly Vote Percentages (Model Outcomes)")
            plt.xlabel("Class")
            plt.ylabel("Anomaly Vote Percentage")
            plt.xticks(rotation=90)
            plt.legend()
            plt.tight_layout()

            # Save the plot
            plot_path = os.path.join(self.output_directory, "IFA_Anomaly_Votes.png")
            plt.savefig(plot_path, format="png", dpi=300)
            plt.close()

            print(f"Saved IFA anomaly vote boxplot: {plot_path}")

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

        # 🧬 Combine DataFrames
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
        Generates interactive 3D PCA plots:
        - Control vs All Classes
        - Custom class list PCA plots from parameters

        Features:
        - Distinguishes up to 12,000 classes (colors, markers, edge styles, transparencies, fill patterns)
        - Randomized class-style assignment
        - Interactive 3D plots with hover labels
        - Control points 3x larger for visibility
        - Variance explained shown on axes
        - Legends only if classes ≤30
        - Plots saved to 'Analysis/PCA'
        """
        print("Generating interactive PCA plots...")

        # Output directory
        pca_output_dir = os.path.join(self.output_directory, "PCA")
        os.makedirs(pca_output_dir, exist_ok=True)

        # Extract features and classes
        data = pd.concat([
            df[self.feature_columns].assign(class_=cls)
            for cls, df in self.data_dict.items()
        ], ignore_index=True)
        features = data[self.feature_columns].fillna(0)
        classes = data["class_"]
        core_well_ids = self.data["core_well_id"]
        
        # PCA computation
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(features)
        explained_variance = pca.explained_variance_ratio_ * 100

        # Style definitions
        colors = px.colors.qualitative.Alphabet  # 26 distinct colors
        markers = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star']
        edge_styles = ['solid', 'dot', 'dash', 'longdash']
        transparencies = [0.4, 0.55, 0.7, 0.85, 1.0]
        fill_patterns = ['/', '\\', '|', '-', '+']

        # Generate all possible combinations
        style_combinations = list(itertools.product(colors, markers, edge_styles, transparencies, fill_patterns))
        random.shuffle(style_combinations)  # Randomize assignment

        unique_classes = sorted(classes.unique())
        if len(unique_classes) > len(style_combinations):
            raise ValueError(f"Too many classes ({len(unique_classes)}). Max supported: {len(style_combinations)}.")

        # Map classes to random styles
        class_style_map = {cls: style_combinations[i] for i, cls in enumerate(unique_classes)}

        nightmode=self.params.get("PCA_nightmode", False)

        def create_pca_plot(pca_data, class_labels, title, filename, nightmode,core_well_id):
            plot_df = pd.DataFrame(pca_data, columns=["PC1", "PC2", "PC3"])
            plot_df["Class"] = class_labels
            plot_df["CoreWellID"] = core_well_id

            # Assign styles to each class
            plot_df["Color"] = plot_df["Class"].map(lambda cls: class_style_map[cls][0])
            plot_df["Symbol"] = plot_df["Class"].map(lambda cls: class_style_map[cls][1])
            plot_df["Alpha"] = plot_df["Class"].map(lambda cls: class_style_map[cls][3])
            plot_df["Size"] = plot_df["Class"].map(lambda cls: 15 if cls == self.control_class else 5)

            # Create the interactive 3D scatter plot with core_well_id in custom_data
            fig = px.scatter_3d(
                plot_df,
                x="PC1", y="PC2", z="PC3",
                color="Class",
                symbol="Class",
                size="Size",
                opacity=0.8,
                hover_data={"CoreWellID": True, "Class": True},
                custom_data=["CoreWellID"],
                title=title
            )

            # Layout adjustments for night mode and white background
            axis_settings = dict(
                backgroundcolor="black" if nightmode else "white",
                gridcolor="gray" if nightmode else "lightgray",
                zerolinecolor="white" if nightmode else "black",
                color="white" if nightmode else "black",
            )

            fig.update_layout(
                scene=dict(
                    xaxis=dict(title=f"PC1 ({explained_variance[0]:.2f}% variance)", **axis_settings),
                    yaxis=dict(title=f"PC2 ({explained_variance[1]:.2f}% variance)", **axis_settings),
                    zaxis=dict(title=f"PC3 ({explained_variance[2]:.2f}% variance)", **axis_settings),
                    bgcolor="black" if nightmode else "white"
                ),
                legend=dict(
                    visible=len(plot_df["Class"].unique()) <= 30,
                    font=dict(color="white" if nightmode else "black")
                ),
                paper_bgcolor="black" if nightmode else "white",
                plot_bgcolor="black" if nightmode else "white",
            )

            # JavaScript to copy core_well_id 
            clipboard_js = """
            document.addEventListener('DOMContentLoaded', function() {
                var plot = document.getElementsByClassName('plotly-graph-div')[0];
                if (plot) {
                    plot.on('plotly_click', function(data) {
                        if (data && data.points && data.points.length > 0) {
                            var coreWellID = data.points[0].customdata[0];  // Access core_well_id
                            navigator.clipboard.writeText(coreWellID).then(function() {
                                alert('Copied to clipboard: ' + coreWellID);  // Confirmation alert
                            }).catch(function(err) {
                                console.error('Clipboard copy failed: ', err);
                            });
                        }
                    });
                }
            });
            """

            # Add the JS code into the HTML file
            html_content = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
            html_with_js = html_content.replace('</body>', f'<script>{clipboard_js}</script></body>')

            # Save the HTML file with the JavaScript for clipboard copying
            output_path = os.path.join(pca_output_dir, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_with_js)

            print(f"Saved interactive PCA plot: {output_path}")

        # Plot: Control vs All Classes
        create_pca_plot(
            pca_data=pca_result,
            class_labels=classes,
            title="PCA: Control vs All Classes",
            filename="PCA_Control_vs_All.html",
            nightmode=nightmode,
            core_well_id=core_well_ids
        )

        # Ensure consistent indexing before PCA
        data = data.reset_index(drop=True)
        pca_result = pca.fit_transform(data[self.feature_columns])

        # Custom class list plots (from params)
        additional_class_lists = self.params.get("pca_custom_groups", [])

        for idx, class_list in enumerate(additional_class_lists):
            try:
                # Ensure both data and self.data have the same indices
                data = data.reset_index(drop=True)
                self.data = self.data.reset_index(drop=True)

                # Filter based on class list
                filtered_indices = data["class_"].isin(class_list)

                # Extract PCA and labels
                pca_filtered = pca_result[filtered_indices.values]
                class_labels_filtered = data.loc[filtered_indices, "class_"].values

                # Extract core_well_id from self.data using aligned indices
                core_well_ids_filtered = self.data.loc[data.index[filtered_indices], "core_well_id"].values

                # Validate lengths
                if not (len(pca_filtered) == len(class_labels_filtered) == len(core_well_ids_filtered)):
                    raise ValueError(
                        f"Length mismatch: PCA data ({len(pca_filtered)}), Classes ({len(class_labels_filtered)}), CoreWellIDs ({len(core_well_ids_filtered)})"
                    )

                # Generate PCA plot
                create_pca_plot(
                    pca_data=pca_filtered,
                    class_labels=class_labels_filtered,
                    title=f"PCA_Control_vs_CustomSet_{idx+1}",
                    filename=f"PCA_Control_vs_CustomSet_{idx+1}.html",
                    nightmode=nightmode,
                    core_well_id=core_well_ids_filtered
                )

            except Exception as e:
                print(f"Error during PCA plot generation: {e}")

        print("PCA plotting completed.")

    def plot_parallel_coordinates(self):
        """
        Generate parallel coordinates plots for the control class vs each target class.
        Features are ordered based on the hierarchical clustering result.
        Saves the plots in a Parallel_Coordinates folder inside the Analysis directory.
        """
        # Ensure p-values and feature order are available
        if not hasattr(self, 'pvalues_df'):
            raise AttributeError("Statistical analysis (p-values) must be performed before running this method.")
        if not hasattr(self, 'feature_order'):
            raise AttributeError("Hierarchical clustering must be performed before running this method.")

        # Create the output directory
        parallel_coordinates_dir = os.path.join(self.output_directory, "Parallel_Coordinates")
        os.makedirs(parallel_coordinates_dir, exist_ok=True)

        # Determine the significance threshold
        significance_threshold = self.params.get("significance_threshold", 0.05)

        # Identify significant features
        pvalues_df = self.pvalues_df.T
        significant_features = pvalues_df.columns[
            (pvalues_df < significance_threshold).any(axis=0)
        ].tolist()

        if not significant_features:
            print("No significant features found to plot.")
            return

        # Filter significant features based on hierarchical clustering order
        ordered_features = [f for f in self.feature_order if f in significant_features]

        if not ordered_features:
            print("No significant features found in the hierarchical clustering order.")
            return

        # Retrieve control class data
        control_data = self.data_dict[self.control_class][self.feature_columns].select_dtypes(include=[np.number])

        class_counter=1
        # Plot control vs each target class
        for target_class, target_data in self.data_dict.items():
            if target_class == self.control_class:
                continue

            # Select numeric features only for the target class
            target_data = target_data[self.feature_columns].select_dtypes(include=[np.number])

            if control_data.empty or target_data.empty:
                continue

            # Combine control and target data
            combined_data = pd.concat([
                control_data[ordered_features].assign(Class=self.control_class),
                target_data[ordered_features].assign(Class=target_class)
            ])

            # Plot the parallel coordinates
            plt.figure(figsize=(15, 8))
            pd.plotting.parallel_coordinates(
                combined_data, class_column='Class', color=["steelblue", "darkkhaki"], alpha=0.7
            )
            plt.title(f"Parallel Coordinates: {self.control_class} vs {target_class}")
            plt.xlabel("Features")
            plt.ylabel("Feature Values")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            # Save the plot
            output_path = os.path.join(parallel_coordinates_dir, f"Parallel_Coordinates_{self.control_class}_vs_{target_class}.png")
            plt.savefig(output_path, dpi=300)
            plt.close("all")

            print(f"{class_counter}:Saved parallel coordinates plot for {self.control_class} vs {target_class} at {output_path}")
            class_counter+=1

            del target_data, combined_data
            gc.collect()

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
        filtered_data = self.data_dict

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
