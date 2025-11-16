import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from itertools import product
from typing import List, Dict, Optional
from experiment import ExperimentFromCSVConverter
import numpy as np
import seaborn as sns


# Define a Parula-like colormap with peach/red tones
def create_parula_colormap():
    """
    Create a colormap similar to MATLAB's parula but with peach and red tones before yellow.
    """
    colors = [
        (0.2081, 0.1663, 0.5292),  # Deep blue
        (0.2116, 0.3896, 0.6780),  # Blue
        (0.1544, 0.5842, 0.7578),  # Cyan
        (0.1667, 0.7569, 0.7176),  # Light cyan
        (0.5259, 0.8706, 0.5843),  # Green
        (0.9020, 0.6706, 0.3490),  # Peach/orange
        (0.9490, 0.4000, 0.3020),  # Coral/red
        (0.9831, 0.7062, 0.3081),  # Golden orange
    ]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('parula_warm', colors, N=n_bins)
    return cmap


class ExperimentBatch:
    """
    A collection of experiment configurations for comparison and analysis.
    """
    
    def __init__(
        self,
        csv_file: str,
        models: List[str],
        prompt_conditions_list: List[Dict],
        experiment_ids: List[int],
        experiment_names: Optional[Dict[int, str]] = None
    ):
        """
        Initialize an ExperimentBatch.
        
        Args:
            csv_file: Path to the CSV file with experiment results
            models: List of model names to compare
            prompt_conditions_list: List of prompt condition dictionaries
            experiment_ids: List of experiment IDs to include
            experiment_names: Optional dict mapping experiment_id to custom name
        """
        self.csv_file = csv_file
        self.models = models
        self.prompt_conditions_list = prompt_conditions_list
        self.experiment_ids = experiment_ids
        self.experiment_names = experiment_names or {}
        
        # Generate all experiment configurations
        self.experiment_configs = [
            {
                "experiment_id": experiment_id,
                "model": model,
                "prompt_conditions": prompt_conditions
            }
            for model, prompt_conditions, experiment_id in product(
                models, prompt_conditions_list, experiment_ids
            )
        ]
        
        # Load and filter the DataFrame
        df = pd.read_csv(csv_file)
        
        # Filter by experiment IDs
        df = df[df['experiment_id'].isin(experiment_ids)]
        
        # Filter by models
        df = df[df['model'].isin(models)]
        
        # Filter by prompt conditions
        condition_filters = []
        for prompt_conditions in prompt_conditions_list:
            condition_mask = pd.Series([True] * len(df), index=df.index)
            for key, value in prompt_conditions.items():
                condition_mask &= (df[key] == value)
            condition_filters.append(condition_mask)
        
        # Combine all condition filters with OR
        if condition_filters:
            combined_filter = condition_filters[0]
            for mask in condition_filters[1:]:
                combined_filter |= mask
            df = df[combined_filter]
        
        # Store the filtered DataFrame
        self.df = df
        
        if len(self.df) == 0:
            print("WARNING: No data matches the specified filters!")
            print(f"  Experiment IDs: {experiment_ids}")
            print(f"  Models: {models}")
            print(f"  Prompt conditions: {prompt_conditions_list}")
    
    def _get_experiment_name(self, experiment_id: int) -> str:
        """Get the custom name for an experiment, or default to 'Exp {id}'."""
        return self.experiment_names.get(experiment_id, f"Exp {experiment_id}")
    
    def to_summary_dataframe(self) -> pd.DataFrame:
        """
        Create a summary DataFrame with one row per experiment configuration.
        
        Returns:
            DataFrame with columns: model, experiment_id, prompt_conditions,
            mean_ratio, median_ratio, std_ratio, within_tight_pct, within_acceptable_pct, etc.
        """
        summary_data = []
        
        for config in self.experiment_configs:
            experiment = ExperimentFromCSVConverter.from_csv(
                self.csv_file,
                experiment_id=config["experiment_id"],
                filters={"model": config["model"]},
                prompt_conditions=[config["prompt_conditions"]]
            )
            df = experiment.to_dataframe()
            ratios = df["ratio_observed_expected"]
            
            # Calculate summary statistics
            summary = {
                "model": config["model"],
                "experiment_id": config["experiment_id"],
                "experiment_name": self._get_experiment_name(config["experiment_id"]),
                "include_nielson": config["prompt_conditions"].get("include_nielson", False),
                "include_examples": config["prompt_conditions"].get("include_examples", False),
                "n_conversions": len(ratios),
                "mean_ratio": ratios.mean(),
                "median_ratio": ratios.median(),
                "std_ratio": ratios.std(),
                "min_ratio": ratios.min(),
                "max_ratio": ratios.max(),
                "within_tight_pct": ((ratios >= 0.95) & (ratios <= 1.05)).sum() / len(ratios) * 100,
                "within_acceptable_pct": ((ratios >= 0.7) & (ratios <= 1.3)).sum() / len(ratios) * 100,
                "within_broad_pct": ((ratios >= 0.3) & (ratios <= 3)).sum() / len(ratios) * 100,
            }
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def plot_overall_cdf(self, figsize=(12, 8), ax=None, colormap='parula'):
        """
        Create a CDF plot comparing overall performance across all experiment configurations.
        
        Args:
            figsize: Figure size
            ax: Optional axes to plot on
            colormap: Colormap to use (default: 'parula', alternatives: 'viridis', 'plasma', 'turbo')
            
        Returns:
            fig, ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        # Get dynamic colors - sample evenly across the colormap
        n_experiments = len(self.experiment_configs)
        if colormap == 'parula':
            cmap = create_parula_colormap()
        else:
            cmap = plt.cm.get_cmap(colormap)
        colors = [cmap(i / (n_experiments - 1)) for i in range(n_experiments)]
        
        for i, config in enumerate(self.experiment_configs):
            experiment = ExperimentFromCSVConverter.from_csv(
                self.csv_file,
                experiment_id=config["experiment_id"],
                filters={"model": config["model"]},
                prompt_conditions=[config["prompt_conditions"]]
            )
            df = experiment.to_dataframe()
            all_ratios = df["ratio_observed_expected"]
            
            # Create label
            nielson = "w/ Nielson" if config['prompt_conditions'].get('include_nielson') else "w/o Nielson"
            examples = "w/ Examples" if config['prompt_conditions'].get('include_examples') else "w/o Examples"
            exp_name = self._get_experiment_name(config["experiment_id"])
            label = f"{config['model']} ({exp_name}): {nielson}, {examples}"
            
            color = colors[i]
            
            sorted_ratios = np.sort(all_ratios)
            cdf = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
            ax.plot(sorted_ratios, cdf, label=label, color=color, linewidth=2.5)
        
        ax.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Ideal Ratio (1.0)")
        ax.axvline(0.7, color="orange", linestyle="--", linewidth=1.5, label="Lower Bound (0.7)")
        ax.axvline(1.3, color="orange", linestyle="--", linewidth=1.5, label="Upper Bound (1.3)")
        ax.set_xlabel("Ratio Observed/Expected", fontsize=12)
        ax.set_ylabel("Cumulative Probability", fontsize=12)
        ax.set_title("Overall CDF Comparison Across Experiments", fontsize=14)
        ax.set_xscale("log")
        x_ticks = [0.1, 0.3, 0.7, 1.0, 1.3, 2, 5, 10, 30]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(tick) for tick in x_ticks])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
        
        return fig, ax
    
    def plot_overall_histogram(self, figsize=(12, 8), colormap='Spectral', sort_by_best=False):
        """
        Generate an overall stacked histogram comparing all experiment configurations,
        sorted by the proportion in the 0.95-1.05 range.
        
        Args:
            figsize (tuple): Figure size
            colormap (str): Colormap to use (not used - custom colors applied)
            sort_by_best (bool): Whether to sort by best performance (0.95-1.05 range)
            
        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects
        """
        import numpy as np
        
        # Clean model names - remove provider prefix
        self.df['model_clean'] = self.df['model'].apply(
            lambda x: x.split('/')[-1] if '/' in x else x
        )
        
        # Get prompt condition columns
        prompt_cols = list(self.prompt_conditions_list[0].keys()) if self.prompt_conditions_list else []
        
        # Find constant columns (same value across all rows)
        constant_conditions = {}
        for col in prompt_cols:
            unique_vals = self.df[col].unique()
            if len(unique_vals) == 1:
                constant_conditions[col] = unique_vals[0]
        
        # Variable columns for grouping
        variable_cols = [col for col in prompt_cols if col not in constant_conditions]
        groupby_cols = variable_cols + ["experiment_id", "model_clean"]
        
        # Calculate proportions for each configuration
        config_proportions = []
        for name, group in self.df.groupby(groupby_cols):
            proportions = {'config': name}
            total = len(group)
            ratio = group['ratio_observed_expected']
            
            # Count each range
            tight_count = ((ratio >= 0.95) & (ratio <= 1.05)).sum()
            acceptable_count = ((ratio >= 0.7) & (ratio <= 1.3)).sum()
            broad_count = ((ratio >= 0.3) & (ratio <= 3.0)).sum()
            
            # Calculate MUTUALLY EXCLUSIVE proportions
            proportions['0.95-1.05'] = tight_count / total
            proportions['0.7-1.3'] = (acceptable_count - tight_count) / total  # Exclude tight
            proportions['0.3-3.0'] = (broad_count - acceptable_count) / total  # Exclude acceptable and tight
            
            config_proportions.append(proportions)
        
        # Convert to DataFrame and sort by target proportion (descending)
        prop_df = pd.DataFrame(config_proportions)
         # Sort if requested
        if sort_by_best:
            prop_df = prop_df.sort_values('0.95-1.05', ascending=False)

        
        # Create labels for x-axis (only variable conditions)
        if variable_cols:
            prop_df['label'] = prop_df['config'].apply(
                lambda x: ', '.join([f"{col}={x[i]}" for i, col in enumerate(variable_cols)] + 
                                [f"exp={x[len(variable_cols)]}", f"{x[len(variable_cols)+1]}"])
            )
        else:
            prop_df['label'] = prop_df['config'].apply(
                lambda x: f"exp={x[0]}, {x[1]}" if isinstance(x, tuple) else f"{x}"
            )
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=figsize)
        x_pos = np.arange(len(prop_df))
        
        # Stack from bottom to top
        ax.bar(x_pos, prop_df['0.95-1.05'], color='black', label='0.95-1.05', width=0.8)
        
        ax.bar(x_pos, prop_df['0.7-1.3'], bottom=prop_df['0.95-1.05'],
            color='gray', label='0.7-1.3', width=0.8)
        
        ax.bar(x_pos, prop_df['0.3-3.0'], 
            bottom=prop_df['0.95-1.05'] + prop_df['0.7-1.3'],
            color='lightgray', label='0.3-3.0', width=0.8)
        
        # Create title with constant conditions
        title = "Proportion of Ratio Observed/Expected by Range"
        if constant_conditions:
            const_str = ', '.join([f"{k}={v}" for k, v in constant_conditions.items()])
            title += f"\n({const_str})"
        
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel('Proportion', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(prop_df['label'], rotation=45, ha='right', fontsize=8)
        ax.legend(title='Ratio Range', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_nielson_effect(self, figsize=(10, 8), range_broad=(0.5, 2.0)):
        """
        Create a scatter plot comparing model performance with and without Nielson data.
        X-axis: Proportion within 0.95-1.05 when include_nielson=True
        Y-axis: Proportion within range_broad when include_nielson=False
        
        Args:
            figsize (tuple): Figure size
            
        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects
        """
        
        # Clean model names
        self.df['model_clean'] = self.df['model'].apply(
            lambda x: x.split('/')[-1] if '/' in x else x
        )
        
        model_performance = []
        
        for model in self.df['model_clean'].unique():
            model_data = self.df[self.df['model_clean'] == model]
            
            # Performance WITH Nielson (include_nielson=True)
            with_nielson = model_data[model_data['include_nielson'] == True]
            if len(with_nielson) > 0:
                ratio_with = with_nielson['ratio_observed_expected']
                prop_tight_with = ((ratio_with >= 0.95) & (ratio_with <= 1.05)).sum() / len(ratio_with)
            else:
                prop_tight_with = None
            
            # Performance WITHOUT Nielson (include_nielson=False)
            without_nielson = model_data[model_data['include_nielson'] == False]
            if len(without_nielson) > 0:
                ratio_without = without_nielson['ratio_observed_expected']
                prop_broad_without = ((ratio_without >= range_broad[0]) & (ratio_without <= range_broad[1])).sum() / len(ratio_without)
            else:
                prop_broad_without = None
            
            # Only add if both conditions exist
            if prop_tight_with is not None and prop_broad_without is not None:
                model_performance.append({
                    'model': model,
                    'with_nielson': prop_tight_with,
                    'without_nielson': prop_broad_without
                })
        
        # Convert to DataFrame
        perf_df = pd.DataFrame(model_performance)
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(perf_df['with_nielson'], perf_df['without_nielson'], 
                s=50, alpha=0.6, color='black', zorder=3)
        
        # Add model labels with adjust_text for better positioning
        texts = []
        for _, row in perf_df.iterrows():
            text = ax.text(row['with_nielson'], row['without_nielson'], 
                        row['model'],
                        fontsize=7, ha='center', va='bottom')
            texts.append(text)
        
        # Try to import adjust_text for automatic label adjustment
        try:
            from adjustText import adjust_text
            adjust_text(texts, 
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                    expand_points=(1.5, 1.5),
                    force_points=(0.5, 0.5))
        except ImportError:
            print("Install adjustText for better label positioning: pip install adjustText")
            # Fallback: just offset labels slightly
            for text in texts:
                text.set_position((text.get_position()[0], text.get_position()[1] + 0.01))
        
        # Add diagonal reference line (y=x) only within data range
        x_min = perf_df['without_nielson'].min()
        x_max = perf_df['without_nielson'].max()
        y_min = perf_df['with_nielson'].min()
        y_max = perf_df['with_nielson'].max()

        
        diag_min = max(x_min, y_min)
        diag_max = min(x_max, y_max)
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.3, label='Equal proportion', zorder=1)
        
        # Build title based on experiment IDs in the batch
        exp_ids = sorted(self.df['experiment_id'].unique())
        if len(exp_ids) == 1:
            exp_name = self.experiment_names.get(exp_ids[0], f"Experiment {exp_ids[0]}")
            title_suffix = f"\n({exp_name})"
        else:
            exp_names = [self.experiment_names.get(eid, f"Exp {eid}") for eid in exp_ids]
            title_suffix = f"\n({', '.join(exp_names)})"

        ax.set_xlabel('Proportion within 0.95-1.05\n(with Nielson conversion table)', fontsize=9)
        ax.set_ylabel(f'Proportion within {range_broad[0]}-{range_broad[1]}\n(without Nielson conversion table)', fontsize=9)
        ax.set_title(f'Effect of Including Nielson Conversion Table on Model Performance{title_suffix}', 
                    fontsize=10, pad=15)
        
        # Expand axis limits to give space for labels
        x_margin = (x_max - x_min) * 0.15  # 15% margin
        y_margin = (y_max - y_min) * 0.15
        
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        ax.grid(True, alpha=0.3, zorder=0)
        ax.legend(fontsize=7)
        ax.tick_params(axis='both', labelsize=8)
        
        plt.tight_layout()
        return fig, ax
        
    def plot_ome_effect(self, figsize=(10, 8), range_broad=(0.6, 1.7), include_nielson=True):
        """
        Create a scatter plot comparing model performance on OME vs non-OME experiments.
        X-axis: Proportion within range_broad for OME experiments (experiment_id == 1)
        Y-axis: Proportion within range_broad for non-OME experiments (experiment_id == 2)
        
        Args:
            figsize (tuple): Figure size
            range_broad (tuple): Range for both axes metric (default: 0.6-1.7)
            include_nielson (bool): Filter for include_nielson condition (default: True)
            
        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects
        """
        import numpy as np
        
        # Clean model names
        self.df['model_clean'] = self.df['model'].apply(
            lambda x: x.split('/')[-1] if '/' in x else x
        )
        
        # Filter for the specified nielson condition
        df_filtered = self.df[self.df['include_nielson'] == include_nielson].copy()
        
        model_performance = []
        
        for model in df_filtered['model_clean'].unique():
            model_data = df_filtered[df_filtered['model_clean'] == model]
            
            # Performance on OME experiments (experiment_id == 1)
            ome_data = model_data[model_data['experiment_id'] == 1]
            if len(ome_data) > 0:
                ratio_ome = ome_data['ratio_observed_expected']
                prop_ome = ((ratio_ome >= range_broad[0]) & (ratio_ome <= range_broad[1])).sum() / len(ratio_ome)
            else:
                prop_ome = None
            
            # Performance on non-OME experiments (experiment_id == 2)
            nonome_data = model_data[model_data['experiment_id'] == 2]
            if len(nonome_data) > 0:
                ratio_nonome = nonome_data['ratio_observed_expected']
                prop_nonome = ((ratio_nonome >= range_broad[0]) & (ratio_nonome <= range_broad[1])).sum() / len(ratio_nonome)
            else:
                prop_nonome = None
            
            # Only add if both conditions exist
            if prop_ome is not None and prop_nonome is not None:
                model_performance.append({
                    'model': model,
                    'ome': prop_ome,
                    'nonome': prop_nonome
                })
        
        # Convert to DataFrame
        perf_df = pd.DataFrame(model_performance)
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(perf_df['ome'], perf_df['nonome'], 
                s=50, alpha=0.6, color='black', zorder=3)
        
        # Add model labels with adjust_text for better positioning
        texts = []
        for _, row in perf_df.iterrows():
            text = ax.text(row['ome'], row['nonome'], 
                        row['model'],
                        fontsize=7, ha='center', va='bottom')
            texts.append(text)
        
        # Try to import adjust_text for automatic label adjustment
        try:
            from adjustText import adjust_text
            adjust_text(texts, 
                    arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5),
                    expand_points=(1.5, 1.5),
                    force_points=(0.5, 0.5))
        except ImportError:
            print("Install adjustText for better label positioning: pip install adjustText")
            # Fallback: just offset labels slightly
            for text in texts:
                text.set_position((text.get_position()[0], text.get_position()[1] + 0.01))
        
        # Add diagonal reference line (y=x) only within data range
        x_min = perf_df['ome'].min()
        x_max = perf_df['ome'].max()
        y_min = perf_df['nonome'].min()
        y_max = perf_df['nonome'].max()
        
        diag_min = max(x_min, y_min)
        diag_max = min(x_max, y_max)
        ax.plot([diag_min, diag_max], [diag_min, diag_max], 'r--', alpha=0.3, label='Equal performance', zorder=1)
        
        nielson_str = "with Nielson table" if include_nielson else "without Nielson table"
        ax.set_xlabel(f'Proportion within {range_broad[0]}-{range_broad[1]}\n(OME: oral morphine included)', fontsize=9)
        ax.set_ylabel(f'Proportion within {range_broad[0]}-{range_broad[1]}\n(Non-OME: oral morphine excluded)', fontsize=9)
        ax.set_title(f'OME vs Non-OME Performance Comparison ({nielson_str})', 
                    fontsize=10, pad=15)
        
        # Expand axis limits to give space for labels
        x_margin = (x_max - x_min) * 0.15  # 15% margin
        y_margin = (y_max - y_min) * 0.15
        
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        ax.grid(True, alpha=0.3, zorder=0)
        ax.legend(fontsize=7)
        ax.tick_params(axis='both', labelsize=8)
        
        plt.tight_layout()
        return fig, ax

    def generate_pdf(
        self,
        output_pdf: str,
        plot_types: List[str] = None,
        figsize: tuple = (10, 48),
        metadata: Dict = None,
        include_summary_plots: bool = True,
        only_summary_plots: bool = False,
        colormap: str = 'parula',
        sort_by_best: bool = False
    ):
        """
        Generate a PDF with visualizations for the experiment batch.

        Args:
            output_pdf (str): Path to the output PDF file.
            plot_types (list): List of plot types to generate (e.g., ["scatter", "ratio_regression", "cdf", "histogram"]).
            figsize (tuple): Figure size for the plots.
            metadata (dict): Metadata for the PDF.
            include_summary_plots (bool): Whether to include summary plots.
            colormap (str): Colormap to use for the plots.
        """
        prompt_cols = list(self.prompt_conditions_list[0].keys()) if self.prompt_conditions_list else []
        
        with PdfPages(output_pdf, metadata=metadata) as pdf:
            # Add summary plots at the beginning if requested
            if include_summary_plots:
                try:
                    # Overall CDF
                    fig_cdf, ax_cdf = self.plot_overall_cdf(figsize=(12, 8), colormap=colormap)
                    pdf.savefig(fig_cdf, bbox_inches='tight')
                    plt.close(fig_cdf)
                    
                    # Overall Histogram
                    fig_hist, ax_hist = self.plot_overall_histogram(figsize=(12, 8), colormap=colormap, sort_by_best=sort_by_best)
                    pdf.savefig(fig_hist, bbox_inches='tight')
                    plt.close(fig_hist)
                    
                except Exception as e:
                    print(f"Error generating summary plots: {e}")
                    import traceback
                    traceback.print_exc()
            
            if only_summary_plots:
                print(f"PDF generated with only summary plots: {output_pdf}")
                return

            # Generate individual plots
            for plot_type in plot_types:
                if plot_type in ["scatter", "ratio_regression"]:
                    # Group by prompt condition columns and experiment ID
                    if prompt_cols:
                        groupby_cols = prompt_cols + ["experiment_id"]
                    else:
                        groupby_cols = ["experiment_id"]
                    
                    for group_keys, exp_group in self.df.groupby(groupby_cols):
                        try:
                            # Check if the group has data
                            if exp_group.empty:
                                print(f"Skipping plot for {plot_type}, group={group_keys}: No data.")
                                continue

                            # Create the plot
                            fig, ax = plt.subplots(figsize=figsize)
                            if plot_type == "scatter":
                                self._plot_scatter(ax, exp_group, colormap)
                            elif plot_type == "ratio_regression":
                                self._plot_ratio_regression(ax, exp_group, colormap)

                            # Create title with group information
                            if len(groupby_cols) > 1:
                                group_info = dict(zip(groupby_cols, group_keys if isinstance(group_keys, tuple) else [group_keys]))
                                title = f"{plot_type.capitalize()} - " + ", ".join([f"{k}={v}" for k, v in group_info.items()])
                            else:
                                title = f"{plot_type.capitalize()} - Experiment ID: {group_keys}"
                            
                            ax.set_title(title)
                            plt.tight_layout()
                            pdf.savefig(fig)
                            plt.close(fig)

                        except Exception as e:
                            print(f"Error generating {plot_type} for group={group_keys}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue

                elif plot_type in ["cdf", "histogram"]:
                    try:
                        # Generate summary plots
                        fig, ax = plt.subplots(figsize=figsize)
                        if plot_type == "cdf":
                            sns.ecdfplot(data=self.df, x="ratio_observed_expected", hue="model", ax=ax, palette=colormap)
                            ax.set_title("CDF of Ratio Observed/Expected by Model")
                        elif plot_type == "histogram":
                            sns.histplot(data=self.df, x="ratio_observed_expected", hue="model", ax=ax, palette=colormap, kde=True)
                            ax.set_title("Histogram of Ratio Observed/Expected by Model")
                        plt.tight_layout()
                        pdf.savefig(fig)
                        plt.close(fig)

                    except Exception as e:
                        print(f"Error generating {plot_type}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

        print(f"PDF generated: {output_pdf}")

    def _plot_scatter(self, ax, group, colormap):
        """Helper function to create scatter plots."""
        for (from_opioid, to_opioid), sub_group in group.groupby(["from_opioid", "to_opioid"]):
            ax.scatter(
                sub_group["expected_to_dose"],
                sub_group["observed_to_dose"],
                label=f"{from_opioid} → {to_opioid}",
                alpha=0.7
            )
        ax.set_xlabel("Expected Dose")
        ax.set_ylabel("Observed Dose")
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')

    def _plot_ratio_regression(self, ax, group, colormap):
        """Helper function to create ratio regression plots."""
        from scipy import stats
        for (from_opioid, to_opioid), sub_group in group.groupby(["from_opioid", "to_opioid"]):
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                sub_group["expected_to_dose"], sub_group["ratio_observed_expected"]
            )
            mean_ratio = sub_group["ratio_observed_expected"].mean()
            std_ratio = sub_group["ratio_observed_expected"].std()
            ax.errorbar(
                slope, mean_ratio, yerr=std_ratio,
                fmt='o', alpha=0.7,
                label=f"{from_opioid} → {to_opioid}"
            )
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel("Regression Slope")
        ax.set_ylabel("Mean Ratio (Observed/Expected)")
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    @staticmethod
    def reorganize_csv(input_file: str, output_file: str):
        """
        Reorganize the CSV by combining all experiment_ids into a single dataset,
        adding a new column 'ome' to indicate if 'oral_morphine' is one of the opioids,
        and assigning new experiment_ids:
        - experiment_id=1 for ome=True
        - experiment_id=2 for ome=False

        Args:
            input_file (str): Path to the input CSV file.
            output_file (str): Path to the output CSV file.
        """
        import pandas as pd

        # Read the input CSV
        df = pd.read_csv(input_file)

        # Add the 'ome' column: True if 'oral_morphine' is in either 'from_opioid' or 'to_opioid'
        df['ome'] = df.apply(
            lambda row: 'oral_morphine' in (row['from_opioid'], row['to_opioid']),
            axis=1
        )

        # Assign new experiment_id based on the 'ome' column
        df['experiment_id'] = df['ome'].apply(lambda x: 1 if x else 2)

        # Save the reorganized DataFrame to the output CSV
        df.to_csv(output_file, index=False)

        print(f"Reorganized CSV saved to {output_file}")

    @staticmethod
    def summarize_csv(csv_file="experiment_results.csv"):
        """
        Summarize ALL entries in the CSV file with detailed experiment metadata.
        Returns a comprehensive summary DataFrame with experiment configurations.
        """
        import pandas as pd
        
        df = pd.read_csv(csv_file)
        
        print("=" * 80)
        print(f"FULL CSV SUMMARY: {csv_file}")
        print("=" * 80)
        print(f"\nTotal entries: {len(df)}")
        print(f"Unique experiments: {df['experiment_id'].nunique()}")
        print(f"Unique models: {df['model'].nunique()}")
        print(f"Experiment IDs: {sorted(df['experiment_id'].unique())}")
        print(f"Models: {sorted(df['model'].unique())}")
        
        # Build comprehensive summary DataFrame
        summary_data = []
        
        priority_1_opioids = ['oral_hydromorphone', 'oral_morphine', 'oral_oxycodone', 
                              'iv_hydromorphone', 'iv_morphine']
        priority_2_opioids = ['oral_codeine', 'oral_hydrocodone', 'oral_methadone', 
                              'oral_oxymorphone', 'oral_tapentadol', 'oral_tramadol',
                              'sublingual_buprenorphine', 'transdermal_fentanyl',
                              'iv_fentanyl', 'sc_fentanyl']
        
        for exp_id in sorted(df['experiment_id'].unique()):
            exp_df = df[df['experiment_id'] == exp_id]
            
            # Get unique models for this experiment
            models = sorted(exp_df['model'].unique())
            
            # Get prompt conditions
            nielson_vals = exp_df['include_nielson'].unique()
            examples_vals = exp_df['include_examples'].unique()
            
            # Determine conversion type and priorities
            to_opioids = exp_df['to_opioid'].unique()
            from_opioids = exp_df['from_opioid'].unique()
            all_opioids = set(from_opioids) | set(to_opioids)
            
            # Determine if pairs
            is_pairs = not (len(to_opioids) == 1 and 'oral_morphine' in to_opioids)
            
            # Check if permutations were used
            uses_permutations = False
            if is_pairs:
                pairs = set()
                for _, row in exp_df.iterrows():
                    pairs.add((row['from_opioid'], row['to_opioid']))
                
                for from_op, to_op in pairs:
                    if (to_op, from_op) in pairs:
                        uses_permutations = True
                        break
            
            # Determine priority level
            if all_opioids.issubset(set(priority_1_opioids)):
                priority = "Priority 1"
            elif all_opioids.issubset(set(priority_1_opioids + priority_2_opioids)):
                has_p1 = bool(all_opioids.intersection(set(priority_1_opioids)))
                has_p2 = bool(all_opioids.intersection(set(priority_2_opioids)))
                if has_p1 and has_p2:
                    priority = "Priority 1 & 2"
                elif has_p2:
                    priority = "Priority 2"
                else:
                    priority = "Priority 1"
            else:
                priority = "Mixed/Unknown"
            
            # Add row for each model in this experiment
            for model in models:
                model_df = exp_df[exp_df['model'] == model]
                
                summary_data.append({
                    'experiment_id': exp_id,
                    'model': model,
                    'count': len(model_df),
                    'priority': priority,
                    'pairs': is_pairs,
                    'use_permutations': uses_permutations if is_pairs else None,
                    'include_nielson': ', '.join(map(str, sorted(nielson_vals))),
                    'include_examples': ', '.join(map(str, sorted(examples_vals))),
                    'from_opioids_count': len(from_opioids),
                    'to_opioids_count': len(to_opioids)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE SUMMARY TABLE")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        
        # Also show the pivot table
        pivot = df.pivot_table(
            index='experiment_id',
            columns='model',
            values='from_dose',
            aggfunc='count',
            fill_value=0
        )
        
        print("\n" + "=" * 80)
        print("PIVOT TABLE (experiment_id vs model)")
        print("=" * 80)
        print(pivot)
        print("\n" + "=" * 80)
        
        return summary_df
    
    def summarize_batch(self):
        """
        Summarize the filtered data in this batch with detailed experiment metadata.
        Shows prompt conditions, opioid priorities, and conversion types for batch data only.
        """
        print("=" * 80)
        print("BATCH SUMMARY (Filtered Data)")
        print("=" * 80)
        print(f"\nTotal entries: {len(self.df)}")
        print(f"Unique experiments in batch: {self.df['experiment_id'].nunique()}")
        print(f"Unique models in batch: {self.df['model'].nunique()}")
        print(f"Experiment IDs: {sorted(self.df['experiment_id'].unique())}")
        print(f"Models: {sorted(self.df['model'].unique())}")
        
        # Detailed experiment configuration
        print("\n" + "=" * 80)
        print("EXPERIMENT CONFIGURATIONS")
        print("=" * 80)
        
        for exp_id in sorted(self.df['experiment_id'].unique()):
            exp_df = self.df[self.df['experiment_id'] == exp_id]
            
            # Get experiment name if available
            exp_name = self.experiment_names.get(exp_id, "")
            if exp_name:
                print(f"\n--- Experiment {exp_id}: {exp_name} ---")
            else:
                print(f"\n--- Experiment {exp_id} ---")
            
            print(f"  Total conversions: {len(exp_df)}")
            print(f"  Models: {sorted(exp_df['model'].unique())}")
            
            # Prompt conditions
            nielson_vals = exp_df['include_nielson'].unique()
            examples_vals = exp_df['include_examples'].unique()
            print(f"  Prompt conditions:")
            print(f"    - include_nielson: {nielson_vals}")
            print(f"    - include_examples: {examples_vals}")
            
            # Determine if pairs or to oral morphine
            to_opioids = exp_df['to_opioid'].unique()
            from_opioids = exp_df['from_opioid'].unique()
            
            if len(to_opioids) == 1 and 'oral_morphine' in to_opioids:
                print(f"  Conversion type: All opioids → oral morphine")
                print(f"  From opioids ({len(from_opioids)}): {sorted(from_opioids)}")
            else:
                print(f"  Conversion type: Opioid pairs")
                print(f"  From opioids ({len(from_opioids)}): {sorted(from_opioids)}")
                print(f"  To opioids ({len(to_opioids)}): {sorted(to_opioids)}")
                
                # Check if permutations were used
                pairs = set()
                for _, row in exp_df.iterrows():
                    pairs.add((row['from_opioid'], row['to_opioid']))
                
                reverse_pairs = 0
                for from_op, to_op in pairs:
                    if (to_op, from_op) in pairs:
                        reverse_pairs += 1
                
                if reverse_pairs > 0:
                    print(f"  Uses permutations: YES (both A→B and B→A)")
                else:
                    print(f"  Uses permutations: NO (one direction only)")
            
            # Determine priority level (infer from opioid names)
            priority_1_opioids = ['oral_hydromorphone', 'oral_morphine', 'oral_oxycodone', 
                                  'iv_hydromorphone', 'iv_morphine']
            priority_2_opioids = ['oral_codeine', 'oral_hydrocodone', 'oral_methadone', 
                                  'oral_oxymorphone', 'oral_tapentadol', 'oral_tramadol',
                                  'sublingual_buprenorphine', 'transdermal_fentanyl',
                                  'iv_fentanyl', 'sc_fentanyl']
            
            all_opioids = set(from_opioids) | set(to_opioids)
            
            if all_opioids.issubset(set(priority_1_opioids)):
                print(f"  Opioid priority: Priority 1 only")
            elif all_opioids.issubset(set(priority_1_opioids + priority_2_opioids)):
                has_p1 = bool(all_opioids.intersection(set(priority_1_opioids)))
                has_p2 = bool(all_opioids.intersection(set(priority_2_opioids)))
                if has_p1 and has_p2:
                    print(f"  Opioid priority: Priority 1 & 2")
                elif has_p2:
                    print(f"  Opioid priority: Priority 2 only")
                else:
                    print(f"  Opioid priority: Priority 1 only")
            else:
                print(f"  Opioid priority: Mixed/Unknown")
        
        # Summary by experiment_id and model
        print("\n" + "=" * 80)
        print("COUNTS BY EXPERIMENT AND MODEL")
        print("=" * 80)
        summary = self.df.groupby(['experiment_id', 'model']).size().reset_index(name='count')
        print(summary.to_string(index=False))
        
        # Detailed summary
        print("\n" + "=" * 80)
        print("DETAILED BREAKDOWN (with prompt conditions)")
        print("=" * 80)
        detailed = self.df.groupby([
            'experiment_id', 'model', 'include_nielson', 'include_examples'
        ]).size().reset_index(name='count')
        print(detailed.to_string(index=False))
        
        # Pivot table
        pivot = self.df.pivot_table(
            index='experiment_id',
            columns='model',
            values='from_dose',
            aggfunc='count',
            fill_value=0
        )
        
        print("\n" + "=" * 80)
        print("PIVOT TABLE (experiment_id vs model)")
        print("=" * 80)
        print(pivot)
        print("\n" + "=" * 80)
        
        return summary