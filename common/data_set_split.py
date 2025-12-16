import random
import torch

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Custom function to split the dataset into train, validation, and test sets
    based on provided ratios.
    """
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."

    # Get the total number of samples in the dataset
    total_size = len(dataset)

    # Calculate split sizes
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    # Generate random indices for splitting
    indices = list(range(total_size))
    random.shuffle(indices)

    # Create subsets by selecting indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    # Use Subset class to create datasets from these indices
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    return train_subset, val_subset, test_subset
