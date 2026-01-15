import fiftyone as fo

# ✅ EDIT THIS LIST to exactly what you want to delete
TO_DELETE = [
    "bdd100k",          # likely old/full
    "bdd100k_hf_10k",   # the broken HF-cache one
    # "bdd100k_10k",    # DON'T delete if this is your good local one
    # "bdd100k_hf_10k_local",  # DON'T delete if this is your good one
]

print("Existing datasets:")
print(fo.list_datasets())

for name in TO_DELETE:
    if fo.dataset_exists(name):
        print(f"Deleting: {name}")
        fo.delete_dataset(name)
    else:
        print(f"Skip (not found): {name}")

print("\nRemaining datasets:")
print(fo.list_datasets())
