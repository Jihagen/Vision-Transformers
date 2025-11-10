import numpy as np
from metrics import run_metrics


if __name__ == "__main__":
    # Run over shards: data/results-FAU_25_0.npy ... data/results-FAU_25_9.npy
    for i in range(10):
        data_path = f"data/results-FAU_100_{i}.npy"
        out_root  = f"results/gdv_100_{i}"
        print(f"\n=== Running GDV for {data_path} → {out_root} ===")

        gdv_results = run_metrics(
            data_path=data_path,
            vision_layer_threshold=50,
            output_root=out_root,
            # Optional: pass custom grids; otherwise module defaults are used
            # umap_setups=[
            #     {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean", "n_components": 2},
            #     {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine",    "n_components": 2},
            # ],
            # tsne_setups=[
            #     {"perplexity": 30, "learning_rate": "auto", "metric": "euclidean", "init": "pca"},
            #     {"perplexity": 30, "learning_rate": "auto", "metric": "cosine",    "init": "pca"},
            # ],
        )


        # Per-shard summary
        print("\nGDV Results Summary:")
        print(f"Total layers analyzed: {len(gdv_results)}")
        max_gdv_layer = max(gdv_results, key=gdv_results.get)
        print(f"Layer with highest GDV: {max_gdv_layer} (GDV = {gdv_results[max_gdv_layer]:.4f})")

