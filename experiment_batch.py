import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from itertools import product
from typing import List, Dict, Optional
from experiment import ExperimentFromCSVConverter
import numpy as np


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
    
    def plot_overall_histogram(self, figsize=(7, 3.5), ax=None, colormap='parula'):
        """
        Create a stacked histogram comparing overall performance across all experiment configurations.
        
        Args:
            figsize: Figure size
            ax: Optional axes to plot on
            colormap: Colormap to use (ignored for histogram, kept for consistency)
            
        Returns:
            fig, ax
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        labels = []
        tight_proportions = []
        acceptable_proportions = []
        broad_proportions = []
        
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
            nielson = "N" if config['prompt_conditions'].get('include_nielson') else "no-N"
            examples = "E" if config['prompt_conditions'].get('include_examples') else "no-E"
            exp_name = self._get_experiment_name(config["experiment_id"])
            labels.append(f"{config['model']}\n{exp_name}\n{nielson}, {examples}")
            
            # Calculate proportions
            total = len(all_ratios)
            tight_count = ((all_ratios >= 0.95) & (all_ratios <= 1.05)).sum()
            tight_prop = tight_count / total if total > 0 else 0
            tight_proportions.append(tight_prop)
            
            acceptable_count = ((all_ratios >= 0.7) & (all_ratios <= 1.3)).sum() - tight_count
            acceptable_prop = acceptable_count / total if total > 0 else 0
            acceptable_proportions.append(acceptable_prop)
            
            broad_count = ((all_ratios >= 0.3) & (all_ratios <= 3)).sum() - tight_count - acceptable_count
            broad_prop = broad_count / total if total > 0 else 0
            broad_proportions.append(broad_prop)
        
        # Create stacked bar chart with black bars
        x_pos = np.arange(len(labels))
        
        # Bottom bar: tight range (alpha=1)
        ax.bar(x_pos, tight_proportions, color='black', alpha=1.0, label="0.95-1.05")
        
        # Middle bar: acceptable range (alpha=0.6)
        ax.bar(x_pos, acceptable_proportions, bottom=tight_proportions,
               color='black', alpha=0.6, label="0.7-1.3")
        
        # Top bar: broad range (alpha=0.3)
        ax.bar(x_pos, broad_proportions,
               bottom=np.array(tight_proportions) + np.array(acceptable_proportions),
               color='black', alpha=0.3, label="0.3-3")
        
        ax.set_ylabel("Proportion", fontsize=10)
        ax.set_title("Overall Proportion Comparison Across Experiments", fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # Add more space at the bottom for rotated labels
        plt.tight_layout()
        
        return fig, ax
    
    def generate_pdf(
        self,
        output_pdf: str,
        plot_types: List[str] = None,
        figsize: tuple = (10, 48),
        metadata: Dict = None,
        include_summary_plots: bool = True,
        colormap: str = 'parula'
    ):
        """
        Generate a multi-page PDF with experiment comparison plots.
        
        Args:
            output_pdf: Path to the output PDF file
            plot_types: List of plot types to generate (e.g., ["scatter", "cdf", "histogram"])
            figsize: Figure size for each plot page
            metadata: Dictionary of PDF metadata
            include_summary_plots: Whether to include overall CDF and histogram summary plots
            colormap: Colormap to use (default: 'parula', alternatives: 'viridis', 'plasma', 'turbo')
        """
        if plot_types is None:
            plot_types = ["scatter", "cdf", "histogram"]
        
        with PdfPages(output_pdf) as pdf:
            # Individual experiment plots
            for plot_type in plot_types:
                fig, axes = ExperimentFromCSVConverter.plot_comparison(
                    self.csv_file,
                    self.experiment_configs,
                    plot_type=plot_type,
                    figsize=figsize
                )
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Summary plots
            if include_summary_plots:
                # Overall CDF
                fig, ax = self.plot_overall_cdf(figsize=(10, 4), colormap=colormap)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                # Overall Histogram
                fig, ax = self.plot_overall_histogram(figsize=(10, 4), colormap=colormap)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Add metadata
            if metadata:
                d = pdf.infodict()
                d['Title'] = metadata.get('title', 'Experiment Comparison')
                d['Author'] = metadata.get('author', '')
                d['Subject'] = metadata.get('subject', '')
                d['Keywords'] = metadata.get('keywords', '')
                d['Creator'] = metadata.get('creator', 'Python matplotlib')
    
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