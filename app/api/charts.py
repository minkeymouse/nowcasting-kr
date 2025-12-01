"""Chart generation utilities using seaborn."""

import io
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any
from fastapi.responses import Response

from services import ModelRegistry, TrainingManager


def generate_model_distribution_chart(model_registry: ModelRegistry) -> Response:
    """Generate model distribution chart using seaborn."""
    models = model_registry.list_models()
    models_by_type = {"DFM": 0, "DDFM": 0}
    for model in models:
        model_type = model.get("model_type", "dfm").upper()
        if model_type == "DFM":
            models_by_type["DFM"] += 1
        elif model_type == "DDFM":
            models_by_type["DDFM"] += 1
    
    # Create seaborn chart
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    data = list(models_by_type.items())
    df = pd.DataFrame(data)
    df.columns = ['Type', 'Count']
    sns.barplot(data=df, x='Type', y='Count', ax=ax, palette=['#006BB4', '#00A69B'])
    ax.set_title('Models by Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return Response(content=buf.read(), media_type="image/png")


def generate_experiment_usage_chart(model_registry: ModelRegistry) -> Response:
    """Generate experiment usage chart using seaborn."""
    usage = model_registry.get_experiment_usage()
    
    if not usage:
        # Return empty chart
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'No experiment usage data', 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
    else:
        # Sort by count, take top 5
        sorted_usage = sorted(usage.items(), key=lambda x: x[1], reverse=True)[:5]
        experiments = [item[0] for item in sorted_usage]
        counts = [item[1] for item in sorted_usage]
        
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        df = pd.DataFrame({'Experiment': experiments, 'Count': counts})
        sns.barplot(data=df, x='Experiment', y='Count', ax=ax, palette='viridis')
        ax.set_title('Most Used Experiments', fontsize=14, fontweight='bold')
        ax.set_xlabel('Experiment ID', fontsize=12)
        ax.set_ylabel('Model Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return Response(content=buf.read(), media_type="image/png")


def generate_training_timeline_chart(training_manager: TrainingManager) -> Response:
    """Generate training timeline chart using seaborn."""
    jobs = training_manager.get_recent_jobs(limit=10)
    
    if not jobs:
        # Return empty chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No training jobs yet', 
               ha='center', va='center', fontsize=14)
        ax.axis('off')
    else:
        # Prepare data
        job_data = []
        for job in jobs:
            job_data.append({
                'Model': job.get('model_name', 'Unnamed')[:20],
                'Progress': job.get('progress', 0),
                'Status': job.get('status', 'unknown')
            })
        
        df = pd.DataFrame(job_data)
        
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color mapping
        color_map = {'completed': '#00A69B', 'running': '#006BB4', 
                    'failed': '#BD271E', 'unknown': '#D3DAE6'}
        colors = [color_map.get(status, '#D3DAE6') for status in df['Status']]
        
        sns.barplot(data=df, x='Model', y='Progress', ax=ax, palette=colors)
        ax.set_title('Recent Training Jobs Progress', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model Name', fontsize=12)
        ax.set_ylabel('Progress (%)', fontsize=12)
        ax.set_ylim(0, 100)
        plt.xticks(rotation=45, ha='right')
        
        # Add status legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[k], label=k.capitalize()) 
                         for k in color_map.keys()]
        ax.legend(handles=legend_elements, loc='upper right')
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return Response(content=buf.read(), media_type="image/png")

