#!/usr/bin/env python3
import os, json, argparse

p = argparse.ArgumentParser()
p.add_argument("--model_path", required=True,
               help="Path to your snapshot folder (where config.json lives)")
args = p.parse_args()
cfg_file = os.path.join(args.model_path, "config.json")
print("→ Reading config:", cfg_file)

with open(cfg_file, "r") as f:
    cfg = json.load(f)

print("\nTop‐level keys in config.json:")
for k in cfg.keys():
    print(" •", k)
# 4) dump vision_config contents
vision_cfg = cfg.get("vision_config") or cfg.get("vision_config", {})
print("\nvision_config keys and values:")
for k,v in vision_cfg.items():
    print(f"  {k}: {v!r}")

# 5) if it's a nested dict, drill one layer deeper
if isinstance(vision_cfg, dict):
    for k,v in vision_cfg.items():
        if isinstance(v, dict):
            print(f"\nInside vision_config['{k}']:")
            for subk in v:
                print("  ", subk)