from typing import Dict, Tuple, Any
import pandas as pd
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

class NameNationalityDataStream(IterableDataset):
    """Dataset for streaming data in chunks."""
    
    def __init__(self, data_file: str, batch_size: int) -> None:
        self.data_file = data_file
        self.batch_size = batch_size

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        for chunk in pd.read_csv(self.data_file, chunksize=self.batch_size):
            chunk = chunk.iloc[worker_id::num_workers]
            chunk = chunk.sample(frac=1)
            
            for _, row in chunk.iterrows():
                yield row['name'], row['alpha2']

def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int,
    num_workers: int = 8
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation and test dataloaders."""
    train_data = NameNationalityDataStream(
        data_file=train_path,
        batch_size=batch_size
    )
    val_data = NameNationalityDataStream(
        data_file=val_path,
        batch_size=batch_size
    )
    test_data = NameNationalityDataStream(
        data_file=test_path,
        batch_size=batch_size
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