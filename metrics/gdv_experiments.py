import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import csv
from sklearn.decomposition import PCA
import matplotlib.animation as animation
import pickle

# ---- GDV Helper Functions ----import numpy as np
from scipy.spatial.distance import pdist

def compute_mean_intra_class_distance(X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.array([label.strip().lower() for label in labels])  # Normalize labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Unique labels: {unique_labels}, Counts: {counts}")
    intra_dists = []
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        print(f"Label: {label}, Indices: {idx}, Count: {len(idx)}")
        if len(idx) < 2:  # Skip only if there are fewer than 2 samples
            print(f"Skipping label {label} due to insufficient samples.")
            continue
        subset = X[idx]
        print(f"Computing intra-class distances for label: {label}")
        print(f"Subset activations shape: {subset.shape}")
        dists = pdist(subset, metric="euclidean")
        print(f"Pairwise distances: {dists}")
        intra_dists.append(np.mean(dists))
    return float(np.mean(intra_dists)) if intra_dists else 0.0

def compute_mean_inter_class_distance(X: np.ndarray, labels: np.ndarray) -> float:
    labels = np.array([label.strip().lower() for label in labels])  # Normalize labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Unique labels: {unique_labels}, Counts: {counts}")
    if len(unique_labels) < 2:
        print("Not enough unique labels for inter-class distance computation.")
        return 0.0
    inter_dists = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            idx1 = np.where(labels == unique_labels[i])[0]
            idx2 = np.where(labels == unique_labels[j])[0]
            print(f"Pair ({unique_labels[i]}, {unique_labels[j]}): Indices1: {idx1}, Indices2: {idx2}")
            if len(idx1) == 0 or len(idx2) == 0:
                print(f"Skipping pair ({unique_labels[i]}, {unique_labels[j]}) due to insufficient samples.")
                continue
            diff = X[idx1][:, None, :] - X[idx2][None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            print(f"Pairwise distances between {unique_labels[i]} and {unique_labels[j]}: {dists}")
            inter_dists.append(np.mean(dists))
    return float(np.mean(inter_dists)) if inter_dists else 0.0

def compute_gdv(X: np.ndarray, labels: np.ndarray) -> float:
    # --- STEP 1: z-score each dim ---
    mu = X.mean(axis=0, keepdims=True)       # (1, D)
    sigma = X.std(axis=0, keepdims=True) + 1e-12
    #print(mu, sigma)
    Xz = (X - mu) / sigma
    Xz *= 0.5  # Scale by 1/2 as per the paper

    # --- STEP 2: Compute intra/inter distances ---
    intra = compute_mean_intra_class_distance(Xz, labels)
    inter = compute_mean_inter_class_distance(Xz, labels)
    print(f"Intra-class distance: {intra}")
    print(f"Inter-class distance: {inter}")

    # --- STEP 3: Combine with scaling factor ---
    D = Xz.shape[1]
    if len(np.unique(labels)) < 2:
        print("Not enough unique labels for GDV computation.")
        return 0.0
    gdv = (1/np.sqrt(D)) * ((1/len(np.unique(labels))) * intra - (2/(len(np.unique(labels)) * (len(np.unique(labels)) - 1))) * inter)
    print(f"GDV value: {gdv}")
    return float(gdv)

# ---- Plotting Function ----
def plot_layer_activations(activations: np.ndarray, labels: np.ndarray, layer_idx: int, gdv_value: float, output_dir='results/gdv/plots'):
    # Ensure labels is a 1D NumPy array
    labels = np.array(labels)
    print(f"Labels shape: {labels.shape}, Labels type: {type(labels)}")
    print(f"Labels: {labels}")

    # PCA transformation
    pca = PCA(n_components=2)
    activations_2d = pca.fit_transform(activations)
    print(f"Layer {layer_idx}: Activations 2D shape = {activations_2d.shape}")
    
    plt.figure(figsize=(6, 6))
    
    # Iterate over unique groups
    unique_groups = np.unique(labels)
    print(f"Unique labels for plotting: {unique_groups}")
    for group in unique_groups:
        print(f"Group: {group}, Group type: {type(group)}")
        group_indices = np.where(labels == group)[0]  # Ensure labels == group comparison works
        print(f"Group {group}: Number of samples = {len(group_indices)}")
        plt.scatter(activations_2d[group_indices, 0], activations_2d[group_indices, 1], label=group)
    
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"Layer {layer_idx}\nGDV = {gdv_value:.4f}")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"layer_{layer_idx}_activations.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()

# ---- Animation Function ----
def animate_layers_smooth(activations_dict: dict, semantic_labels: np.ndarray, hold_count: int = 3, interp_count: int = 5, interval: int = 500):
    sorted_layers = sorted(activations_dict.keys())
    pca_data = {}
    gdv_data = {}
    for layer in sorted_layers:
        X = activations_dict[layer].numpy()
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        pca_data[layer] = X_2d
        gdv_data[layer] = compute_gdv(X, semantic_labels)
    
    schedule = []
    L = len(sorted_layers)
    for i in range(L - 1):
        for _ in range(hold_count):
            schedule.append(("recorded", i, 0))
        for j in range(1, interp_count):
            frac = j / (interp_count - 1)
            schedule.append(("interp", i, frac))
    for _ in range(hold_count):
        schedule.append(("recorded", L - 1, 0))
    
    total_frames = len(schedule)
    
    unique_groups = np.unique(semantic_labels)
    colors = plt.cm.get_cmap("tab10", len(unique_groups))
    
    fig, ax = plt.subplots(figsize=(6, 6))
    scatters = {}
    for idx, group in enumerate(unique_groups):
        scat = ax.scatter([], [], color=colors(idx), label=f"Group {group}")
        scatters[group] = scat
    title = ax.set_title("")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    
    all_data = np.concatenate(list(pca_data.values()), axis=0)
    x_min, x_max = all_data[:,0].min(), all_data[:,0].max()
    y_min, y_max = all_data[:,1].min(), all_data[:,1].max()
    ax.set_xlim(x_min - 0.1*(x_max - x_min), x_max + 0.1*(x_max - x_min))
    ax.set_ylim(y_min - 0.1*(y_max - y_min), y_max + 0.1*(y_max - y_min))
    
    def init():
        for group in unique_groups:
            scatters[group].set_offsets(np.empty((0, 2)))
        title.set_text("")
        return list(scatters.values()) + [title]
    
    def update(frame):
        mode, base_idx, frac = schedule[frame]
        if mode == "recorded":
            cur_layer = sorted_layers[base_idx]
            data_2d = pca_data[cur_layer]
            cur_gdv = gdv_data[cur_layer]
            title_text = f"Layer {cur_layer}: GDV = {cur_gdv:.4f}"
        elif mode == "interp":
            layer_a = sorted_layers[base_idx]
            layer_b = sorted_layers[base_idx+1]
            data_2d = (1-frac)*pca_data[layer_a] + frac*pca_data[layer_b]
            cur_gdv = (1-frac)*gdv_data[layer_a] + frac*gdv_data[layer_b]
            title_text = f"Transition: {layer_a}→{layer_b} (t={frac:.2f})"
        else:
            data_2d = None
            title_text = ""
        for group in unique_groups:
            mask = semantic_labels == group
            points = data_2d[mask]
            scatters[group].set_offsets(points)
        title.set_text(title_text)
        return list(scatters.values()) + [title]
    
    ani = animation.FuncAnimation(fig, update, frames=total_frames, init_func=init,
                                  interval=interval, blit=False, repeat=True)
    plt.show()
    return ani

# ---- Main Function to Run GDV Experiment and Save Data for Dash ----

def run_gdv_experiment(data_path):
    # Load the precomputed activations from the numpy file.
    loaded_results = np.load(data_path, allow_pickle=True).item()
    results_list = loaded_results['results']

    # Initialize dictionaries to store aggregated activations and GDV values.
    layer_activations = {}
    gdv_all = {}
    layer_info = {}

    # Aggregate activations per layer across samples.
    for result in results_list:
        embeddings = result['embeddings']  # dict of layer_idx -> activations
        for layer_idx, act_tensor in enumerate(embeddings.values()):
            X = np.array(act_tensor)  # shape = (n_tokens+cls, D)
            layer_activations.setdefault(layer_idx, []).append(X)

    # Compute GDV and build 2D positions using CLS-token PCA.
    for layer_idx, activations_list in layer_activations.items():
        # Stack tokens (including CLS) for GDV calculation
        X_all = np.vstack(activations_list)
        labels = [r['interestingness'] for r in results_list]
        gdv_value = compute_gdv(X_all, labels)
        gdv_all[layer_idx] = gdv_value

        # Extract CLS token (position 0) from each sample's activations
        cls_vectors = np.vstack([arr[0] for arr in activations_list])  # shape = (n_samples, D)

        # PCA down to 2D on the CLS vectors
        pca = PCA(n_components=2)
        cls_2d = pca.fit_transform(cls_vectors)  # shape = (n_samples, 2)

        # Save one point per sample for Dash
        layer_info[layer_idx] = {
            'x':        cls_2d[:, 0].tolist(),
            'y':        cls_2d[:, 1].tolist(),
            'group':    [r['interestingness'] for r in results_list],
            'sentence': [r['explanation']    for r in results_list]
        }

    # Prepare metadata for Dash
    sorted_layers = sorted(layer_activations.keys())
    max_gdv_layer = max(gdv_all, key=gdv_all.get)
    meta = {
        'labels':       [r['interestingness'] for r in results_list],
        'total_layers': len(sorted_layers),
        'max_gdv_layer': max_gdv_layer,
        'explanations': [r['explanation'] for r in results_list]
    }

    # Save GDV CSV
    output_dir = 'results/_gdv/'
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'gdv_values.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Layer', 'GDV'])
        for layer_idx, gdv in sorted(gdv_all.items()):
            writer.writerow([layer_idx, gdv])

    # Dump pickle for Dash
    output_data = {
        'sorted_layers': sorted_layers,
        'layer_data':    layer_info,
        'gdv_per_layer': gdv_all,
        'meta':          meta
    }
    with open(os.path.join(output_dir, 'gdv.pkl'), 'wb') as f:
        pickle.dump(output_data, f)

    print(f"Saved GDV and layer data to {os.path.join(output_dir, 'gdv.pkl')}")
    return gdv_all
