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
                fig, ax = self.plot_overall_cdf(figsize=(14, 8), colormap=colormap)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                
                # Overall Histogram
                fig, ax = self.plot_overall_histogram(figsize=(14, 8), colormap=colormap)
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