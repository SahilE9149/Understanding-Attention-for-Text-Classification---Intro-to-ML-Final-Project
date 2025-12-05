"""
Generate comparison tables and summaries from extension_results.csv.
"""

import pandas as pd
import numpy as np


def load_results(csv_file="extension_results.csv"):
    """Load and parse results CSV."""
    df = pd.read_csv(csv_file)
    return df


def create_accuracy_comparison_table(df):
    """
    Create accuracy table by encoder, dataset, and scaling factor.
    
    Output:
    ┌─────────┬────────┬────────┬────────┐
    │Encoder  │τ=1    │τ=10   │τ=100  │
    ├─────────┼────────┼────────┼────────┤
    │LSTM     │0.820  │0.822  │0.815  │
    │Affine   │0.810  │0.811  │0.805  │
    └─────────┴────────┴────────┴────────┘
    """
    print("\n" + "="*80)
    print("TABLE 1: Test Accuracy by Encoder and Scaling Factor τ")
    print("="*80)
    
    for dataset in df['dataset'].unique():
        print(f"\n📊 Dataset: {dataset.upper()}")
        print("-" * 80)
        
        df_dataset = df[df['dataset'] == dataset]
        
        # Pivot: rows=encoder, columns=scaling_tau
        pivot_table = df_dataset.pivot_table(
            index='encoder',
            columns='scaling_tau',
            values='test_accuracy',
            aggfunc='mean'
        )
        
        print(pivot_table.to_string())
        print()


def create_correlation_comparison_table(df):
    """Create polarity-attention correlation comparison."""
    print("\n" + "="*80)
    print("TABLE 2: Polarity-Attention Correlation r by Encoder")
    print("="*80)
    
    for dataset in df['dataset'].unique():
        print(f"\n📊 Dataset: {dataset.upper()}")
        print("-" * 80)
        
        df_dataset = df[df['dataset'] == dataset]
        
        pivot_table = df_dataset.pivot_table(
            index='encoder',
            columns='scaling_tau',
            values='correlation_r',
            aggfunc='mean'
        )
        
        print(pivot_table.to_string())
        print()


def create_timing_comparison_table(df):
    """Create training time comparison."""
    print("\n" + "="*80)
    print("TABLE 3: Training Time (seconds) by Encoder")
    print("="*80)
    
    for dataset in df['dataset'].unique():
        print(f"\n📊 Dataset: {dataset.upper()}")
        print("-" * 80)
        
        df_dataset = df[df['dataset'] == dataset]
        
        pivot_table = df_dataset.pivot_table(
            index='encoder',
            columns='scaling_tau',
            values='training_time_sec',
            aggfunc='mean'
        )
        
        print(pivot_table.to_string())
        print()


def create_speedup_table(df):
    """
    Calculate speedup relative to LSTM baseline.
    Speedup = LSTM_time / Encoder_time
    """
    print("\n" + "="*80)
    print("TABLE 4: Speedup Relative to LSTM Baseline")
    print("="*80)
    print("(Values > 1.0 indicate faster than LSTM)")
    
    for dataset in df['dataset'].unique():
        print(f"\n📊 Dataset: {dataset.upper()}")
        print("-" * 80)
        
        df_dataset = df[df['dataset'] == dataset].copy()
        
        # Get LSTM baseline times
        lstm_times = df_dataset[df_dataset['encoder'] == 'lstm'].set_index('scaling_tau')['training_time_sec']
        
        speedups = []
        for encoder in ['affine', 'gru', 'cnn']:
            for tau in sorted(df_dataset['scaling_tau'].unique()):
                encoder_time = df_dataset[
                    (df_dataset['encoder'] == encoder) & 
                    (df_dataset['scaling_tau'] == tau)
                ]['training_time_sec'].values
                
                if len(encoder_time) > 0 and tau in lstm_times.index:
                    speedup = lstm_times[tau] / encoder_time[0]
                    speedups.append({
                        'encoder': encoder.upper(),
                        'tau': tau,
                        'speedup': speedup
                    })
        
        speedup_df = pd.DataFrame(speedups)
        if not speedup_df.empty:
            pivot = speedup_df.pivot_table(index='encoder', columns='tau', values='speedup')
            print(pivot.to_string())
        print()


def create_best_config_table(df):
    """Find best configuration for each encoder."""
    print("\n" + "="*80)
    print("TABLE 5: Best Configuration per Encoder (Highest Accuracy on Test Set)")
    print("="*80)
    
    for dataset in df['dataset'].unique():
        print(f"\n📊 Dataset: {dataset.upper()}")
        print("-" * 80)
        
        df_dataset = df[df['dataset'] == dataset]
        
        best_configs = df_dataset.loc[df_dataset.groupby('encoder')['test_accuracy'].idxmax()]
        
        summary = best_configs[[
            'encoder_display', 'scaling_tau', 'test_accuracy', 
            'correlation_r', 'training_time_sec'
        ]].copy()
        summary.columns = ['Encoder', 'τ', 'Accuracy', 'Correlation r', 'Time (s)']
        summary = summary.set_index('Encoder')
        
        print(summary.to_string())
        print()


def create_hypothesis_summary(df):
    """
    Summarize findings for Extension 1 hypothesis.
    H_ext1: "Modern encoders (GRU, CNN) match Affine speed while improving accuracy."
    """
    print("\n" + "="*80)
    print("EXTENSION 1 HYPOTHESIS EVALUATION")
    print("="*80)
    print("\nH_ext1: Modern encoders (GRU, CNN) will maintain or improve accuracy")
    print("        over Affine while approaching similar speedup vs. LSTM.\n")
    
    # Compare average metrics
    for dataset in df['dataset'].unique():
        df_dataset = df[df['dataset'] == dataset]
        
        print(f"\n📊 Dataset: {dataset.upper()}")
        print("-" * 60)
        
        summary_stats = df_dataset.groupby('encoder').agg({
            'test_accuracy': ['mean', 'std'],
            'correlation_r': ['mean', 'std'],
            'training_time_sec': ['mean', 'std']
        }).round(4)
        
        print(summary_stats)
        
        # Speedup calculation
        lstm_avg_time = df_dataset[df_dataset['encoder'] == 'lstm']['training_time_sec'].mean()
        for encoder in ['affine', 'gru', 'cnn']:
            enc_avg_time = df_dataset[df_dataset['encoder'] == encoder]['training_time_sec'].mean()
            speedup = lstm_avg_time / enc_avg_time
            print(f"\n{encoder.upper()} speedup vs LSTM: {speedup:.2f}x")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Load results
    df = load_results()
    
    # Generate all tables
    create_accuracy_comparison_table(df)
    create_correlation_comparison_table(df)
    create_timing_comparison_table(df)
    create_speedup_table(df)
    create_best_config_table(df)
    create_hypothesis_summary(df)
    
    print("\n✅ All tables generated successfully!")
