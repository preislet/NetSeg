import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def load_metrics_from_jsonl(filepath):
    """Loads and processes metrics from a JSONL file into a pandas DataFrame."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found at {filepath}")
        return None
        
    with open(filepath, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # The 'dev' column contains dictionaries, we need to expand them
    dev_metrics = pd.json_normalize(df['dev'])
    
    # Combine the main df with the expanded dev metrics
    df = pd.concat([df.drop(columns=['dev']), dev_metrics], axis=1)
    
    return df

def plot_training_curves():
    """
    Generates and saves plots for training loss, validation accuracy, 
    and other validation metrics.
    """
    unet_path = os.path.join('checkpoints', 'unet_metrics.jsonl')
    resunet_path = os.path.join('checkpoints', 'resunet_metrics.jsonl')
    
    unet_df = load_metrics_from_jsonl(unet_path)
    resunet_df = load_metrics_from_jsonl(resunet_path)
    
    if unet_df is None or resunet_df is None:
        print("Could not generate plots because one or both metric files are missing.")
        return

    # Set a consistent style for plots
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- 1. Plot Training Loss ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(unet_df['epoch'], unet_df['train_loss'], 'o-', label='U-Net', color='royalblue')
    ax1.plot(resunet_df['epoch'], resunet_df['train_loss'], 's-', label='ResU-Net', color='darkorange')
    ax1.set_title('Training Loss per Epoch', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Weighted Cross-Entropy Loss', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.set_xlim(left=0)
    fig1.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    print("Saved training_loss.png")

    # --- 2. Plot Validation Accuracy ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(unet_df['epoch'], unet_df['accuracy'], 'o-', label='U-Net', color='royalblue')
    ax2.plot(resunet_df['epoch'], resunet_df['accuracy'], 's-', label='ResU-Net', color='darkorange')
    ax2.set_title('Validation Accuracy per Epoch', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Categorical Accuracy', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.set_xlim(left=0)
    fig2.savefig('validation_accuracy.png', dpi=300, bbox_inches='tight')
    print("Saved validation_accuracy.png")

    # --- 3. Plot Validation Macro Precision and Recall ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(unet_df['epoch'], unet_df['macro_precision'], 'o--', label='U-Net Macro Precision', color='cornflowerblue')
    ax3.plot(unet_df['epoch'], unet_df['macro_recall'], 'o-', label='U-Net Macro Recall', color='royalblue')
    ax3.plot(resunet_df['epoch'], resunet_df['macro_precision'], 's--', label='ResU-Net Macro Precision', color='sandybrown')
    ax3.plot(resunet_df['epoch'], resunet_df['macro_recall'], 's-', label='ResU-Net Macro Recall', color='darkorange')
    ax3.set_title('Validation Macro-Averaged Metrics per Epoch', fontsize=16)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.legend(fontsize=12)
    ax3.set_xlim(left=0)
    ax3.set_ylim(bottom=0.7, top=1.0) # Adjust if needed
    fig3.savefig('validation_metrics.png', dpi=300, bbox_inches='tight')
    print("Saved validation_metrics.png")
    
    plt.close('all')

if __name__ == '__main__':
    plot_training_curves()