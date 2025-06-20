
# Dict[str, Dict[str, List[Dict[str, Tensor]]]]
# dataset_name -> model_name -> list of entries

import os
import torch

# Final structure:
# Dict[str, Dict[str, Dict[str, Any]]]
# results[dataset_name][model_name] = {
#     "file_contents": <string from txt file>,
#     "data": <list of dicts with tensors from .pt>
# }

import numpy as np

def load_results(results_root="results"):
    '''
    Go through results/ and create a dict for each dataset-model combination, save them in a model_name-indexed dict for each dataset,
    and save those in a dataset_name-indexed dict. Each inner dict contains the data read from the .pt file in that directory.
    returns said dict.
    '''
    results = {}

    for dataset_name in os.listdir(results_root):
        dataset_path = os.path.join(results_root, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        results[dataset_name] = {}

        for model_name in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model_name)
            if not os.path.isdir(model_path):
                continue

            pt_files = [f for f in os.listdir(model_path) if f.endswith(".pt")]
            txt_files = [f for f in os.listdir(model_path) if f.endswith(".txt")]

            if not pt_files:
                continue
            if len(pt_files) > 1:
                raise ValueError(f"Multiple .pt files in {model_path}, expected only one.")
            if len(txt_files) != 1:
                raise ValueError(f"Expected exactly one .txt file in {model_path}, found {len(txt_files)}.")

            pt_path = os.path.join(model_path, pt_files[0])
            txt_path = os.path.join(model_path, txt_files[0])

            data = torch.load(pt_path)
            with open(txt_path, "r") as f:
                file_contents = f.read()

            results[dataset_name][model_name] = {
                "hf_model_name": file_contents,
                "data": data
            }

    return results
# The reason we have to precompute the cosines during training is that 
# we need the model embedding layer to produce the vector embeddings. 
# The tokenizer doesnt have that, only the model

import matplotlib.pyplot as plt
import numpy as np

def process_experiment_results(experiment_results, remove_zero_one_points):
    '''
    Do the correlation analysis for one model-dataset combination.
    Compute pearson and spearman correlation scores.
    Create plot with median cosines and standard deviation
    '''
    entropies = []
    cosines = []

    for prompt_result in experiment_results["data"]:
        prompt_entropies = prompt_result["entropies"]
        prompt_cosines = prompt_result["cosines"]
        entropies.extend(prompt_entropies)
        cosines.extend(prompt_cosines)
    
    entropies = np.array(entropies, dtype=float)
    cosines = np.array(cosines, dtype=float)

    #mask = (entropies == 0) & np.isnan(cosines)
    #cosines[mask] = 1.0

    
    if remove_zero_one_points:
        mask = ~((entropies == 0) & (cosines == 1))
        entropies = entropies[mask]
        cosines = cosines[mask]

    # compute pearson and spearman
    from scipy.stats import pearsonr
    r, p = pearsonr(entropies, cosines)
    print("pearson coefficient")
    print(r)
    print(p)
    from scipy.stats import spearmanr
    rho, p = spearmanr(entropies, cosines)
    print("Spearman coefficient")
    print(rho)
    print(p)


    #coeffs = np.polyfit(entropies, cosines, deg=4)
    #print(coeffs)
    # Generate x values from 0 to 1
    #x_fit = np.linspace(0, 8, 500)

    # Compute the polynomial values
    #y_fit = np.polyval(coeffs, x_fit)

    # Plot
    #plt.figure(figsize=(8, 5))
    #plt.plot(x_fit, y_fit, label='Cubic Polynomial Fit', color='darkred')
    #plt.xlabel("Normalized X")
    #plt.ylabel("Predicted Y")
    #plt.title("Cubic Polynomial Regression Fit")
    #plt.legend()
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()


    # Create scatter plot
    #plt.figure(figsize=(8, 6))
    #plt.scatter(entropies, cosines, alpha=0.1)
    #plt.xlabel('Entropy')
    #plt.ylabel('Cosine Similarity')
    #plt.title('Entropy vs. Cosine Similarity')
    #plt.tight_layout()
    #plt.show()
    # Create line plot

    from scipy.stats import binned_statistic

    # Assuming entropies and cosines are already numpy arrays
    num_bins = 50
    bin_means, bin_edges, _ = binned_statistic(entropies, cosines, statistic='mean', bins=num_bins)
    bin_medians, _, _ = binned_statistic(entropies, cosines, statistic='median', bins=num_bins)

    # Compute standard deviation per bin
    bin_indices = np.digitize(entropies, bin_edges)
    bin_std = np.zeros(num_bins)
    for i in range(1, num_bins + 1):
        bin_values = cosines[bin_indices == i]
        if len(bin_values) > 0:
            bin_std[i-1] = np.std(bin_values)

    # Bin centers for plotting
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, bin_medians, label='Median Cosine Similarity', color='blue')
    plt.fill_between(bin_centers,
                    bin_medians - bin_std,
                    bin_medians + bin_std,
                    color='blue',
                    alpha=0.2,
                    label='±1 Std Dev')
    plt.xlabel('Entropy')
    plt.ylabel('Cosine Similarity')
    plt.title('Entropy vs. Cosine Similarity (Median ± Std Dev)')
    plt.legend()
    plt.tight_layout()
    plt.show()


from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def unimodal_fitting(entropies, cosines):
    # Example input (replace with your actual data)
    X = np.array(entropies)
    y = np.array(cosines)

    # Define the unimodal function
    def unimodal_func(x, a, b, c):
        return a * x * np.exp(-b * x) + c

    # Fit the curve
    popt, _ = curve_fit(unimodal_func, X, y, maxfev=10000)
    y_pred = unimodal_func(X, *popt)
    r2 = r2_score(y, y_pred)
    print(f"Fitted params: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")
    print(f"R² = {r2:.3f}")

    # Sort for plotting
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    # Plot data and fit
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X, y, alpha=0.3, label='Data')
    ax.plot(X_sorted, y_pred_sorted, color='red', label='Unimodal Fit')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Unimodal Fit: $a x e^{-bx} + c$')
    ax.legend()
    fig.tight_layout()
    fig.show()

# Dict[str, Dict[str, List[Dict[str, Tensor]]]]
# dataset_name -> model_name -> list of entries

import os
import torch

# Final structure:
# Dict[str, Dict[str, Dict[str, Any]]]
# results[dataset_name][model_name] = {
#     "file_contents": <string from txt file>,
#     "data": <list of dicts with tensors from .pt>
# }

import numpy as np

def load_results(results_root="results"):
    '''
    Go through results/ and create a dict for each dataset-model combination, save them in a model_name-indexed dict for each dataset,
    and save those in a dataset_name-indexed dict. Each inner dict contains the data read from the .pt file in that directory.
    returns said dict.
    '''
    results = {}

    for dataset_name in os.listdir(results_root):
        dataset_path = os.path.join(results_root, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        results[dataset_name] = {}

        for model_name in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model_name)
            if not os.path.isdir(model_path):
                continue

            pt_files = [f for f in os.listdir(model_path) if f.endswith(".pt")]
            txt_files = [f for f in os.listdir(model_path) if f.endswith(".txt")]

            if not pt_files:
                continue
            if len(pt_files) > 1:
                raise ValueError(f"Multiple .pt files in {model_path}, expected only one.")
            if len(txt_files) != 1:
                raise ValueError(f"Expected exactly one .txt file in {model_path}, found {len(txt_files)}.")

            pt_path = os.path.join(model_path, pt_files[0])
            txt_path = os.path.join(model_path, txt_files[0])

            data = torch.load(pt_path)
            with open(txt_path, "r") as f:
                file_contents = f.read()

            results[dataset_name][model_name] = {
                "hf_model_name": file_contents,
                "data": data
            }

    return results

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic, pearsonr, spearmanr
import matplotlib.cm as cm
import matplotlib.colors as mcolors

global_model_data_points = {}
global_dataset_data_points = {}
global_model_data_median = {}
global_dataset_data_median = {}

def process_and_collect(experiment_results, remove_zero_one_points, dataset_name, model_name):
    entropies = []
    cosines = []

    for prompt_result in experiment_results["data"]:
        entropies.extend(prompt_result["entropies"])
        cosines.extend(prompt_result["cosines"])

    entropies = np.array(entropies, dtype=float)
    cosines = np.array(cosines, dtype=float)




    if remove_zero_one_points:
        mask = ~((entropies == 0) & (cosines == 1))
        entropies = entropies[mask]
        cosines = cosines[mask]

    # Add to global 
    global_model_data_points[model_name] = global_model_data_points.get(model_name, {"entropies" : np.array([], dtype=float), "cosines" : np.array([], dtype=float)})
    global_model_data_points[model_name]["entropies"] = np.concatenate([global_model_data_points[model_name]["entropies"] , entropies])
    global_model_data_points[model_name]["cosines"] = np.concatenate([global_model_data_points[model_name]["cosines"] , cosines])

    global_dataset_data_points[dataset_name] = global_dataset_data_points.get(dataset_name, {"entropies" : np.array([], dtype=float), "cosines" : np.array([], dtype=float)})
    global_dataset_data_points[dataset_name]["entropies"] = np.concatenate([global_dataset_data_points[dataset_name]["entropies"] , entropies])
    global_dataset_data_points[dataset_name]["cosines"] = np.concatenate([global_dataset_data_points[dataset_name]["cosines"] , cosines])

    # Stats
    r, p = pearsonr(entropies, cosines)
    rho, p2 = spearmanr(entropies, cosines)
    print(f"  Pearson r={r:.3f}, Spearman rho={rho:.3f}")
    

    # Binning
    num_bins = 50
    bin_medians, bin_edges, _ = binned_statistic(entropies, cosines, statistic='median', bins=num_bins)
    bin_indices = np.digitize(entropies, bin_edges)
    bin_std = np.zeros(num_bins)
    for i in range(1, num_bins + 1):
        bin_values = cosines[bin_indices == i]
        if len(bin_values) > 0:
            bin_std[i-1] = np.std(bin_values)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    #TODO also try mean instead of median


    mask = np.isfinite(bin_centers) & np.isfinite(bin_medians)
    cleaned_bin_centers = bin_centers[mask]
    cleaned_bin_medians = bin_medians[mask]
    # Add to global 
    global_model_data_median[model_name] = global_model_data_median.get(model_name, {"entropies" : np.array([], dtype=float), "cosines" : np.array([], dtype=float)})
    global_model_data_median[model_name]["entropies"] = np.concatenate([global_model_data_median[model_name]["entropies"] , cleaned_bin_centers])
    global_model_data_median[model_name]["cosines"] = np.concatenate([global_model_data_median[model_name]["cosines"] , cleaned_bin_medians])

    global_dataset_data_median[dataset_name] = global_dataset_data_median.get(dataset_name, {"entropies" : np.array([], dtype=float), "cosines" : np.array([], dtype=float)})
    global_dataset_data_median[dataset_name]["entropies"] = np.concatenate([global_dataset_data_median[dataset_name]["entropies"] , cleaned_bin_centers])
    global_dataset_data_median[dataset_name]["cosines"] = np.concatenate([global_dataset_data_median[dataset_name]["cosines"] , cleaned_bin_medians])


    #return bin_centers, bin_medians, bin_std
    return bin_centers, bin_medians, bin_std, r, p, rho, p2



root_dir = "results/" #TODO use this in .py
#root_dir = "./../results/" #TODO use this in .ipynb
results = load_results(root_dir)

# Collect datasets/models
datasets = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
models = set()
for d in datasets:
    d_path = os.path.join(root_dir, d)
    for m in os.listdir(d_path):
        if os.path.isdir(os.path.join(d_path, m)):
            models.add(m)
models = list(models)

# Assign fixed color per model
model_colors = dict(zip(models, cm.get_cmap("tab10").colors[:len(models)]))

fig, axes = plt.subplots(len(models), len(datasets), figsize=(5 * len(datasets), 4 * len(models)), sharex=True, sharey=True)
if len(datasets) == 1 and len(models) == 1:
    axes = np.array([[axes]])
elif len(models) == 1:
    axes = axes[np.newaxis, :]
elif len(datasets) == 1:
    axes = axes[:, np.newaxis]


import pandas as pd

stats_rows = []



for i, model_name in enumerate(models):
    for j, dataset_name in enumerate(datasets):
        ax = axes[i][j]
        model_path = os.path.join(root_dir, dataset_name, model_name)
        if not os.path.isdir(model_path):
            ax.set_visible(False)
            continue

        print(f"Model: {model_name} Dataset: {dataset_name}")
        experiment_results = results[dataset_name][model_name]
        bin_centers, bin_medians, bin_std, r, p, rho, p2 = process_and_collect(experiment_results, True, dataset_name, model_name)
        
        # Unimodal model fitting
        bin_centers2 = np.nan_to_num(bin_centers, nan=0.0)
        bin_medians2 = np.nan_to_num(bin_medians, nan=0.0)
        # TODO dont replace nan with 0, is there a way to remove nan?
        unimodal_fitting(bin_centers2, bin_medians2)

        # Add row to stats table
        stats_rows.append({
            "Model": model_name,
            "Dataset": dataset_name,
            "Pearson r": round(r, 4),
            "Pearson p": round(p, 4),
            "Spearman r": round(rho, 4),
            "Spearman p": round(p2, 4)
        })
        lower = np.clip(bin_medians - bin_std, 0, None)
        upper = bin_medians + bin_std
        color = model_colors[model_name]

        ax.plot(bin_centers, bin_medians, label=model_name, color=color)
        ax.fill_between(bin_centers, lower, upper, alpha=0.2, color=color)

        # Left-side model name
        if j == 0:
            ax.set_ylabel(f"{model_name}\nCosine Similarity", fontsize=10)

        # Right-side model name again for clarity (optional)
        if j == len(datasets) - 1:
            ax.text(1.02, 0.5, model_name,
                    transform=ax.transAxes,
                    fontsize=11, fontweight='bold',
                    va='center', ha='left')

        if i == len(models) - 1:
            ax.set_xlabel("Entropy")
        if i == 0:
            ax.set_title(dataset_name)

        ax.grid(True)

# Shared legend (optional, add outside grid if needed)
# fig.legend(models, loc="upper right")

fig.suptitle("Entropy vs. Cosine Similarity (Median ± Std Dev)", fontsize=16)
fig.tight_layout(rect=[0, 0, 0.96, 0.97])
fig.savefig("correlation_eval/correlation.png", dpi=300, bbox_inches="tight")

fig.show()


# Build stats table
stats_df = pd.DataFrame(stats_rows)
stats_df.set_index(["Model", "Dataset"], inplace=True)

# Print or save
print("\n=== Correlation Statistics Table ===")
print(stats_df.to_string())

# Optional: save to CSV
stats_df.to_csv("correlation_eval/correlation_stats.csv")

with open("correlation_eval/correlation_stats.tex", "w") as f:
    f.write(stats_df.to_latex(multicolumn=True, multirow=True, float_format="%.2f"))

