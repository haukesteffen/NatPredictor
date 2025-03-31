from typing import Dict, Tuple, Any, Optional
import pandas as pd
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info

class NameNationalityData(Dataset):
    """Dataset for loading all data at once."""
    
    def __init__(self, data_file: str, transform=None) -> None:
        """Initialize dataset.
        
        Args:
            data_file: Path to CSV file containing names and country codes
            transform: Optional transform to apply to data
        """
        # Load all data
        self.df = pd.read_csv(data_file)
        print(f'Dataset has {len(self.df)} records.')
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row['name']
        country_code = row['alpha2']
        
        if self.transform:
            return self.transform(name, country_code)
        return name, country_code

class NameNationalityDataStream(IterableDataset):
    """Dataset for streaming data in chunks."""
    
    def __init__(self, data_file: str, chunksize: int, transform=None) -> None:
        """Initialize dataset.
        
        Args:
            data_file: Path to CSV file containing names and country codes
            chunksize: Number of rows to load at once
            transform: Optional transform to apply to data
        """
        self.data_file = data_file
        self.chunksize = chunksize
        self.transform = transform

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        
        # Read CSV in chunks
        for chunk in pd.read_csv(self.data_file, chunksize=self.chunksize):
            # Partition chunk rows among workers
            chunk = chunk.iloc[worker_id::num_workers]
            # Shuffle chunk rows
            chunk = chunk.sample(frac=1)
            
            for _, row in chunk.iterrows():
                name = row['name']
                country_code = row['alpha2']
                
                if self.transform:
                    yield self.transform(name, country_code)
                else:
                    yield name, country_code

def create_dataloaders(
    transform: Any,
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int,
    num_workers: int = 8
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation and test dataloaders.
    
    Args:
        transform: Transform to apply to data
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        batch_size: Batch size for training
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    # Create datasets with transform
    train_data = NameNationalityDataStream(
        data_file=train_path,
        chunksize=batch_size,
        transform=transform
    )
    
    val_data = NameNationalityData(
        data_file=val_path,
        transform=transform
    )
    
    test_data = NameNationalityData(
        data_file=test_path,
        transform=transform
    )
    
    return (
        DataLoader(
            train_data, 
            batch_size=batch_size, 
            num_workers=num_workers,
            persistent_workers=True
        ),
        DataLoader(
            val_data, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            persistent_workers=True
        ),
        DataLoader(
            test_data, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            persistent_workers=True
        )
    )