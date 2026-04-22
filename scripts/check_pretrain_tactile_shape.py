#!/usr/bin/env python3
"""Check tactile dimensions in pretrain.zarr.zip"""
import sys
import os
import json

def main():
    zarr_path = "/home/zinan/Documents/zinan/data/touch_in_the_wild-dataset/pretrain_data/pretrain.zarr.zip"
    if not os.path.exists(zarr_path):
        print(f"File not found: {zarr_path}")
        return 1

    print(f"Opening: {zarr_path}")
    
    # Read zarr metadata directly from zip (avoids loading codecs for rgb)
    import zipfile
    with zipfile.ZipFile(zarr_path, 'r') as zf:
        names = zf.namelist()
        print("\n--- Zarr structure (sample) ---")
        for n in sorted(names)[:50]:
            if '.zarray' in n or '.zattrs' in n:
                print(f"  {n}")
        
        # Find tactile array metadata
        tactile_meta = [n for n in names if 'tactile' in n and '.zarray' in n]
        print(f"\n--- Tactile array metadata files: {tactile_meta} ---")
        for path in tactile_meta:
            with zf.open(path) as f:
                meta = json.load(f)
                print(f"\n  {path}:")
                print(f"    {json.dumps(meta, indent=4)}")
        
        # Also check .zattrs for data group
        for n in names:
            if 'data/.zattrs' == n or 'meta/.zattrs' == n:
                with zf.open(n) as f:
                    attrs = json.load(f)
                    print(f"\n  {n}: {list(attrs.keys())}")
    
    # Try loading with zarr (tactile might not use jpegxl)
    import zarr
    with zarr.ZipStore(zarr_path, mode='r') as zip_store:
        root = zarr.group(zip_store)
        if 'data' in root:
            data_grp = root['data']
            for k in data_grp.keys():
                if 'tactile' in k.lower():
                    try:
                        arr = data_grp[k]
                        print(f"\n--- Loaded data['{k}'] ---")
                        print(f"  shape: {arr.shape}")
                        print(f"  dtype: {arr.dtype}")
                        if arr.shape[0] > 0:
                            sample = arr[0]
                            print(f"  sample[0] shape: {sample.shape}")
                            print(f"  sample[0] min/max: {float(sample.min()):.3f} / {float(sample.max()):.3f}")
                    except Exception as e:
                        print(f"  Error loading {k}: {e}")
    
    print("\n--- Done ---")
    return 0

if __name__ == "__main__":
    sys.exit(main())
