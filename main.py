import numpy as np
from metrics import run_gdv_experiment
"""
if __name__ == "__main__":
    data_path = "data/results-FAU_100.npy"
    gdv_results = run_gdv_experiment(data_path, save_plots=True)
    
    # Optional: Print summary of results
    print(f"\nGDV Results Summary:")
    print(f"Total layers analyzed: {len(gdv_results)}")
    max_gdv_layer = max(gdv_results, key=gdv_results.get)
    print(f"Layer with highest GDV: {max_gdv_layer} (GDV = {gdv_results[max_gdv_layer]:.4f})")
    
    

"""



#────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run over shards: data/results-FAU_25_0.npy ... data/results-FAU_25_9.npy
    for i in range(10):
        data_path = f"data/results-FAU_100_{i}.npy"
        out_root  = f"results/gdv_100_{i}"
        print(f"\n=== Running GDV for {data_path} → {out_root} ===")

        gdv_results = run_gdv_experiment(
            data_path=data_path,
            vision_layer_threshold=50,
            output_root=out_root
        )

        # Per-shard summary
        print("\nGDV Results Summary:")
        print(f"Total layers analyzed: {len(gdv_results)}")
        max_gdv_layer = max(gdv_results, key=gdv_results.get)
        print(f"Layer with highest GDV: {max_gdv_layer} (GDV = {gdv_results[max_gdv_layer]:.4f})")
