import os
import torchvision
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="stl10", choices=['stl10', 'cifar10'])
args = parser.parse_args()

# --- Define the exact same base path used in base.py ---
base_data_path = "/opt/dlami/nvme/ERA-V3-S23-assignment/torchvision_datasets"
# ---

if args.dataset_name.lower() == 'stl10':
    dataset_root = os.path.join(base_data_path, "stl10")
    print(f"Ensuring STL-10 exists at: {dataset_root}")
    torchvision.datasets.STL10(root=dataset_root, split='train', download=True)
    torchvision.datasets.STL10(root=dataset_root, split='test', download=True) # Download test too if needed later
elif args.dataset_name.lower() == 'cifar10':
    dataset_root = os.path.join(base_data_path, "cifar10")
    print(f"Ensuring CIFAR-10 exists at: {dataset_root}")
    torchvision.datasets.CIFAR10(root=dataset_root, train=True, download=True)
    torchvision.datasets.CIFAR10(root=dataset_root, train=False, download=True) # Download test too

print(f"{args.dataset_name} dataset download/verification complete.")