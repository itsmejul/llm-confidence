import torch

# Load the dictionary
data = torch.load("output_2025-05-13_17-12.pt")  # replace with your actual filename

# Print high-level structure
print(f"Top-level keys ({len(data)}):", list(data.keys())[:5])  # print first 5 keys

# Inspect first entry in detail
first_key = list(data.keys())[0]
first_entry = data[first_key]

print(f"\nStructure of entry '{first_key}':")
if isinstance(first_entry, dict):
    for k, v in first_entry.items():
        print(f"  {k}: {type(v)}", end='')
        if isinstance(v, torch.Tensor):
            print(f" shape={v.shape}, dtype={v.dtype}")
        else:
            print()
else:
    print(f"  {type(first_entry)}: {first_entry}")

print(data["prompt92"]["correct"])