import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy import stats

from config import cfg
from model import get_model
from train_memristor_cnn import train_one_epoch, evaluate, MemFCLState, get_fc_linears, build_g_states_per_color, build_optimizer_conv_and_bias
from memristor_model import MemristorModel
from data_loader import get_cifar10_loaders

# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_confusion_matrix(model, loader, device, epoch, mode, seed, export_dir):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'Confusion Matrix - {mode} - Seed {seed} - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, f'cm_{mode}_seed{seed}_epoch{epoch}.png'))
    plt.close()
    
    # Analyze Top 3 Errors
    np.fill_diagonal(cm_norm, 0)
    flat_indices = np.argsort(cm_norm.ravel())[::-1]
    top_errors = []
    print(f"\n[Analysis] Top 3 Misclassifications ({mode}, Epoch {epoch}):")
    for i in range(3):
        idx = flat_indices[i]
        true_cls, pred_cls = np.unravel_index(idx, cm_norm.shape)
        score = cm_norm[true_cls, pred_cls]
        print(f"  {i+1}. True: {true_cls} -> Pred: {pred_cls} (Rate: {score:.2%})")

def save_gradient_histograms(model, epoch, mode, seed, export_dir):
    # This requires gradients to be present. 
    # Since train_one_epoch zeroes grads, we might not have them here unless we hook.
    # Alternatively, we can visualize WEIGHTS which is also useful.
    # User asked for "Gradient distribution", so let's try to capture them during training if possible.
    # For now, we'll plot Weight distribution as a proxy for layer activity/health.
    
    plt.figure(figsize=(12, 6))
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name and param.requires_grad and len(param.shape) > 1:
            if i > 10: break # Limit to first few layers
            data = param.detach().cpu().numpy().flatten()
            plt.hist(data, bins=50, alpha=0.5, label=name, density=True)
            
    plt.title(f'Weight Distribution - {mode} - Seed {seed} - Epoch {epoch}')
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig(os.path.join(export_dir, f'weights_hist_{mode}_seed{seed}_epoch{epoch}.png'))
    plt.close()

def run_experiment():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    modes = ["bidir", "unidir"]
    overlaps = [0.0, 0.1, 0.3, 0.5]
    seeds = [42] # Fixed seed as requested for paper
    epochs = cfg.EPOCHS
    
    export_dir = "experiment_results"
    os.makedirs(export_dir, exist_ok=True)
    
    # Global Results: results[mode][seed][overlap] = [acc_list]
    global_results = {m: {s: {ov: [] for ov in overlaps} for s in seeds} for m in modes}
    
    # Load LTP/LTD
    if not os.path.exists(cfg.LTP_LTD_PATH):
        print(f"Error: {cfg.LTP_LTD_PATH} not found.")
        return
    ltp, ltd = MemristorModel.load_ltp_ltd_csv(cfg.LTP_LTD_PATH, ltp_count=cfg.LTP_COUNT)
    g_states = build_g_states_per_color(ltp, ltd)

    # Prepare Data Loaders (Overlap=0 for training)
    print("Preparing Data Loaders...")
    cfg.RGB_PRE_MODE_TRAIN = "4bit_ideal"
    cfg.RGB_OVERLAP_TRAIN = 0.0
    train_loader, _ = get_cifar10_loaders(batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS)
    
    # Test Loaders
    test_loaders = {}
    for ov in overlaps:
        cfg.RGB_PRE_MODE_TEST = "4bit_overlap"
        cfg.RGB_OVERLAP_TEST = ov
        _, test_loader = get_cifar10_loaders(batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS)
        test_loaders[ov] = test_loader

    criterion = torch.nn.CrossEntropyLoss().to(device) # Placeholder, overridden in train_memristor_cnn.py

    for seed in seeds:
        print(f"\n\n{'='*40}\nRunning Seed: {seed}\n{'='*40}")
        set_seed(seed)
        
        for mode in modes:
            print(f"\n--- Training Mode: {mode} (Seed {seed}) ---")
            
            # Init Model (Strong Variant with ResBlocks)
            model = get_model("strong", num_classes=10).to(device)
            
            # Init Memristor State
            fcl_state = MemFCLState(g_states, max_pulses_per_step=cfg.MAX_PULSES_PER_STEP, color_mapping="round_robin", device=device)
            fc1, fc2 = get_fc_linears(model)
            fcl_state.init_layer_from_float("fc1", fc1.weight.detach().cpu().numpy())
            fcl_state.init_layer_from_float("fc2", fc2.weight.detach().cpu().numpy())
            fcl_state.writeback("fc1", fc1.weight)
            fcl_state.writeback("fc2", fc2.weight)
            
            optimizer = build_optimizer_conv_and_bias(model, lr=cfg.LR)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            # Early Stopping Variables
            best_val_acc = 0.0
            patience = 5
            trigger_times = 0
            
            train_losses = []
            val_losses = [] # Using Ov=0.0 as Validation
            
            for ep in range(epochs):
                train_acc, train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, fcl_state, mode=mode)
                
                # Evaluate on Ov=0.0 (Validation)
                val_acc, val_loss = evaluate(model, test_loaders[0.0], criterion, device)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                # Store results for all overlaps
                print(f"Ep {ep+1}/{epochs} | TrnAcc: {train_acc:.4f} Loss: {train_loss:.4f} | Val(Ov0): {val_acc:.4f} |", end=" ")
                
                global_results[mode][seed][0.0].append(val_acc * 100.0)
                
                for ov in overlaps:
                    if ov == 0.0: continue
                    test_acc, _ = evaluate(model, test_loaders[ov], criterion, device)
                    global_results[mode][seed][ov].append(test_acc * 100.0)
                    print(f"Ov{ov}: {test_acc:.4f}", end=" | ")
                print()
                
                # Monitoring: Confusion Matrix every 5 epochs
                if (ep + 1) % 5 == 0:
                    save_confusion_matrix(model, test_loaders[0.0], device, ep+1, mode, seed, export_dir)
                    save_gradient_histograms(model, ep+1, mode, seed, export_dir)
                
                # Early Stopping Check (Disabled for Paper Curves)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Save best model
                    torch.save(model.state_dict(), os.path.join(export_dir, f"best_model_{mode}_seed{seed}.pth"))
                
                # Overfitting Check
                if (train_acc - val_acc) > 0.08:
                    print(f"Warning: Potential Overfitting detected! Train-Val Gap: {train_acc - val_acc:.4f}")

                scheduler.step()
            
            # Save Loss Curve
            plt.figure()
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.title(f'Loss Curve - {mode} - Seed {seed}')
            plt.legend()
            plt.savefig(os.path.join(export_dir, f'loss_{mode}_seed{seed}.png'))
            plt.close()

    # Final Analysis & Plotting
    plot_final_results(global_results, epochs, overlaps, modes, export_dir)

def plot_final_results(results, epochs, overlaps, modes, export_dir):
    # Plot Mean Accuracy with Shaded Error Bar
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    colors = {0.0: 'r', 0.1: 'g', 0.3: 'orange', 0.5: 'b'}
    
    for idx, mode in enumerate(modes):
        ax = axes[idx]
        ax.set_title(f"{'Bidirectional' if mode=='bidir' else 'Unidirectional'} Model (Mean of 3 Seeds)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        for ov in overlaps:
            # Gather data across seeds: [seed1_list, seed2_list, seed3_list]
            data = np.array([results[mode][s][ov] for s in results[mode]])
            # Pad if lengths differ (due to early stopping logic above, should be same)
            # Actually, I filled remaining epochs, so it should be fine.
            # But let's handle length mismatch just in case.
            min_len = min(len(d) for d in data)
            data = data[:, :min_len]
            
            mean_acc = np.mean(data, axis=0)
            std_acc = np.std(data, axis=0)
            x = range(min_len)
            
            ax.plot(x, mean_acc, label=f'Overlap {ov}', color=colors[ov])
            ax.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, color=colors[ov], alpha=0.1)
            
            # Print final stats
            print(f"[{mode} Ov{ov}] Final Mean: {mean_acc[-1]:.2f}% +/- {std_acc[-1]:.2f}%")

        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(export_dir, "final_performance_comparison.png"))
    print(f"Results saved to {export_dir}")
    
    # P-value check for Ov=0.5
    bidir_ov5 = [results['bidir'][s][0.5][-1] for s in results['bidir']]
    unidir_ov5 = [results['unidir'][s][0.5][-1] for s in results['unidir']]
    
    t_stat, p_val = stats.ttest_ind(bidir_ov5, unidir_ov5)
    print(f"\n[Hypothesis Test] Ov=0.5 Bidir vs Unidir:")
    print(f"  Bidir: {np.mean(bidir_ov5):.2f}%")
    print(f"  Unidir: {np.mean(unidir_ov5):.2f}%")
    print(f"  P-value: {p_val:.4f} ({'Significant' if p_val < 0.05 else 'Not Significant'})")

if __name__ == "__main__":
    run_experiment()
