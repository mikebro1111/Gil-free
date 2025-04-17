import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List
import seaborn as sns

class ResultsProcessor:
    """Process and visualize benchmark results"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        
    def load_results(self, benchmark_type: str) -> pd.DataFrame:
        """Load results from CSV file"""
        latest_run = max((d for d in (self.results_dir / benchmark_type).iterdir() if d.is_dir()))
        return pd.read_csv(latest_run / 'results.csv')
    
    def plot_basic_operations(self):
        """Plot basic operations benchmark results"""
        df = self.load_results('basic_ops')
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='size', y='mean_time', hue='implementation')
        plt.title('Basic Operations Performance')
        plt.xlabel('Array Size')
        plt.ylabel('Time (seconds)')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        
        return plt.gcf()
    
    def plot_linear_regression(self):
        """Plot linear regression benchmark results"""
        df = self.load_results('linear_regression')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot training time
        sns.lineplot(data=df, x='n_samples', y='mean_fit_time', 
                    hue='implementation', ax=ax1)
        ax1.set_title('Training Time')
        ax1.set_xlabel('Number of Samples')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True)
        
        # Plot prediction time
        sns.lineplot(data=df, x='n_samples', y='mean_predict_time', 
                    hue='implementation', ax=ax2)
        ax2.set_title('Prediction Time')
        ax2.set_xlabel('Number of Samples')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        return fig 