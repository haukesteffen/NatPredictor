# Standard library imports
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Third-party imports
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info
import country_converter as coco

@dataclass
class DataConfig:
    """Configuration for data preprocessing and loading."""
    maximum_name_length: int
    vocabulary: str
    country_codes: List[str]
    country_mapping: Dict[str, str]
    
    @classmethod
    def from_files(cls, maximum_name_length: int, vocab_path: str, 
                  country_codes_path: str, target_class: str = 'UNregion') -> 'DataConfig':
        """Create DataConfig from files and parameters."""
        with open(vocab_path, 'r') as f:
            vocabulary = f.read()
            
        with open(country_codes_path, 'r') as f:
            country_codes = f.read().splitlines()
            
        country_mapping = {cc: coco.convert(names=cc, to=target_class) 
                         for cc in country_codes}
        
        return cls(
            maximum_name_length=maximum_name_length,
            vocabulary=vocabulary,
            country_codes=country_codes,
            country_mapping=country_mapping
        )

class BaseNationalityDataset:
    """Base class for nationality datasets with common encoding/decoding methods."""
    
    def __init__(self, maximum_name_length: int, vocabulary: str, 
                 country_codes: list, country_mapping: dict) -> None:
        self.maximum_name_length = maximum_name_length
        self.vocabulary = vocabulary
        self.country_codes = country_codes
        self.country_mapping = country_mapping
        self.padding_index = 0
        
        # Generate mappings
        self.character_to_index, self.index_to_character, self.vocabulary_length = \
            self._generate_name_mapping(self.vocabulary)
        
        self.class_to_index, self.index_to_class, self.alpha2_to_index, self.number_of_classes = \
            self._generate_country_mapping(self.country_codes, self.country_mapping)
    
    def _generate_name_mapping(self, vocabulary):
        """
        Each character of the vocabulary is assigned an integer index, starting at 1 so that 0 can be 
        used as a padding index. The vocabulary is a list of unique characters generated from the dataset.

        Parameters
        ----------
        vocabulary : str
            A string of unique characters generated from the input dataset.

        Returns
        -------
        tuple of (dict, dict, int):
            ctoi : dict
                Mapping from character to integer index.
            itoc : dict
                Mapping from integer index back to character.
            vocabulary_length : int
                The number of unique characters in vocabulary.
        """
        vocabulary_length = len(vocabulary)
        ctoi = {c:i for i, c in enumerate(vocabulary, 1)} # start enumeration at 1 because 0 is padding index
        itoc = {i:c for i, c in enumerate(vocabulary, 1)}
        return ctoi, itoc, vocabulary_length

    def _generate_country_mapping(self, country_codes, country_mapping):
        """
        Generate mappings between target classes and their indices based on country codes.

        This method creates three mappings:
        - A mapping from each target class (e.g., UN region) to a unique index (starting at 1,
            since index 0 is reserved for padding).
        - The inverse mapping from these indices back to the target classes.
        - A mapping from each country code in the provided list to the corresponding class index,
            as determined by the country_mapping dictionary.

        Additionally, it calculates the total number of distinct target classes.

        Parameters
        ----------
        country_codes : list
            A list of all country codes present in the dataset.
        country_mapping : dict
            A dictionary mapping each country code to its target class.

        Returns
        -------
        Tuple of (dict, dict, dict, int):
            class_to_index : dict
                Mapping from target class to its unique index.
            index_to_class : dict
                Mapping from index to target class.
            alpha2_to_index : dict
                Mapping from country codes to their corresponding class indices.
            number_of_classes : int
                The total number of distinct target classes.
        """
        output_classes = set(country_mapping.values())
        number_of_classes = len(output_classes)
        class_to_index = {c:i for i, c in enumerate(output_classes, 1)} # start enumeration at 1 because 0 is padding index
        index_to_class = {i:c for c, i in class_to_index.items()}
        alpha2_to_index = {country: class_to_index[country_mapping[country]] for country in country_codes}
        return class_to_index, index_to_class, alpha2_to_index, number_of_classes 
    
    def _encode_name(self, seq):
        """
        Encodes a single string or a list of strings into integer indices based on `self.character_to_index`,
        replacing unmapped characters with `self.padding_index`. Each encoded sequence is then padded
        to `self.maximum_name_length`, and the original (unpadded) lengths are recorded.

        Parameters
        ----------
        seq : str or list of str
            The input string(s) to be converted.

        Returns
        -------
        Tuple of (torch.Tensor, torch.Tensor):
            If input is a single string:
                padded_tensor : torch.Tensor of shape (self.maximum_name_length,)
                sequence_length : torch.Tensor (scalar) indicating the original length
            If input is a list of strings:
                padded_tensors : torch.Tensor of shape (batch_size, self.maximum_name_length)
                sequence_lengths : torch.Tensor of shape (batch_size,) indicating the original lengths
        """
        assert isinstance(seq, (str, list)), "Input must be string or list of strings"
        
        if isinstance(seq, str):
            # Process single string without wrapping it in a list.
            encoded = [self.character_to_index.get(char, self.padding_index) for char in seq]
            seq_len = len(encoded)
            padded_tensor = torch.full((self.maximum_name_length,), self.padding_index, dtype=torch.int32)
            max_len = min(seq_len, self.maximum_name_length)
            padded_tensor[:max_len] = torch.tensor(encoded[:max_len], dtype=torch.int32)
            return padded_tensor, torch.tensor(max_len, dtype=torch.int32)
        
        else:  # seq is a list of strings
            encoded_input = []
            for s in seq:
                assert isinstance(s, str), "Each element in the list must be a string"
                encoded_input.append([self.character_to_index.get(char, self.padding_index) for char in s])
            sequence_lengths = torch.tensor(
                [min(len(encoding), self.maximum_name_length) for encoding in encoded_input],
                dtype=torch.int32
            )
            batch_size = len(encoded_input)
            padded_tensors = torch.full(
                (batch_size, self.maximum_name_length),
                self.padding_index,
                dtype=torch.int32
            )
            for i, encoding in enumerate(encoded_input):
                seq_len = len(encoding)
                max_len = min(seq_len, self.maximum_name_length)
                padded_tensors[i, :max_len] = torch.tensor(encoding[:max_len], dtype=torch.int32)
            
        return padded_tensors, sequence_lengths

    def _decode_name(self, seq_tensor):
        """
        Decodes a 1D or 2D tensor of integer indices into characters using the `self.index_to_character` mapping.
        
        - If `seq_tensor` is 1D (shape: [N]), it decodes a single sequence of characters.
        - If `seq_tensor` is 2D (shape: [B, N]), it decodes multiple sequences (one per row).

        Parameters
        ----------
        seq_tensor : torch.Tensor
            A 1D or 2D tensor of integer indices.

        Returns
        -------
        list of str:
            If the input is 1D, returns a single-element list with the decoded name string.
            If the input is 2D, returns a list of decoded names, one per row.
        """
        if not isinstance(seq_tensor, torch.Tensor):
            raise TypeError("seq_tensor must be a torch.Tensor of integer indices.")
        if seq_tensor.dim() == 1:
            return [''.join([self.index_to_character.get(int(idx), '') for idx in seq_tensor])]
        elif seq_tensor.dim() == 2:
            decoded_sequences = []
            for row in seq_tensor:
                decoded_sequences.append(''.join([self.index_to_character.get(int(idx), '') for idx in row]))
            return decoded_sequences
        else:
            raise ValueError("seq_tensor must be a 1D or 2D tensor of integer indices.")

    def _encode_country(self, country_input):
        """
        Encode a country code or a list of country codes into one-hot vectors.

        This method maps the input country code(s) to their corresponding indices using
        `self.alpha2_to_index`. If a country code is not found in the mapping, the padding
        index (`self.padding_index`) is used. The resulting indices are then converted into
        one-hot encoded tensors with a dimensionality of `self.number_of_classes + 1` (to account
        for the padding index).

        Parameters
        ----------
        country_input : str or list of str
            A single country code or a list of country codes.

        Returns
        -------
        encoded_tensors : torch.Tensor
            A tensor containing the one-hot encoded representation(s) of the input.
        """
        assert isinstance(country_input, (str, list)), 'Input must be string or list of strings'
        if isinstance(country_input, str):
            encoded_output = self.alpha2_to_index.get(country_input, self.padding_index)
        elif isinstance(country_input, list):
            encoded_output = []
            for c in country_input:
                assert isinstance(c, str), 'Input must be string or list of strings'
                encoded_output.append(self.alpha2_to_index.get(c, self.padding_index))
        index_tensors = torch.tensor(encoded_output, dtype=torch.int64)
        encoded_tensors = F.one_hot(index_tensors, num_classes=self.number_of_classes+1).to(torch.float32)
        return encoded_tensors      

    def _decode_country(self, country_code_tensor):
        """
        Decodes a 1D or 2D one-hot-encoded tensor into its corresponding country code strings.

        For a 1D tensor (shape: [num_classes]), it finds the index of the maximum value 
        (the argmax) and returns a list containing the corresponding country code.

        For a 2D tensor (shape: [batch_size, num_classes]), it applies the same argmax 
        operation along each row, returning a list of country codes for the entire batch.

        Parameters
        ----------
        country_code_tensor : torch.Tensor
            A 1D or 2D tensor representing one-hot-encoded country codes.

        Returns
        -------
        list of str
            - If the input is 1D, returns a single-element list with the decoded country code.
            - If the input is 2D, returns a list of decoded country codes, one per row.

        Raises
        ------
        TypeError
            If `country_code_tensor` is not a torch.Tensor.
        ValueError
            If `country_code_tensor` is neither 1D nor 2D.
        """
        if not isinstance(country_code_tensor, torch.Tensor):
            raise TypeError("country_code_tensor must be a torch.Tensor of integer indices.")
        if country_code_tensor.dim() == 1:
            return [self.index_to_class.get(torch.argmax(country_code_tensor).item(), 'Unknown')]
        elif country_code_tensor.dim() == 2:
            decoded_output = []
            for encoding in country_code_tensor:
                index = torch.argmax(encoding).item()
                decoded_output.append(self.index_to_class.get(index, 'Unknown'))
            return decoded_output
        else:
            raise ValueError("country_code_tensor must be a 1D or 2D tensor of integer indices.")

class NameNationalityData(BaseNationalityDataset, Dataset):
    """Dataset for loading all data at once."""
    
    def __init__(self, data_file: str, **kwargs) -> None:
        super().__init__(**kwargs)
        
        # Load and process all data
        df = pd.read_csv(data_file)
        print(f'Dataset has {len(df)} records.')
        
        self.X, self.sequence_lengths = self._encode_name(df['name'].to_list())
        self.y = self._encode_country(df['alpha2'].to_list())
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.sequence_lengths[idx]

class NameNationalityDataStream(BaseNationalityDataset, IterableDataset):
    """Dataset for streaming data in chunks."""
    
    def __init__(self, data_file: str, chunksize: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_file = data_file
        self.chunksize = chunksize

    def __iter__(self):
        worker_info = get_worker_info()
        # If in a worker process, partition workload among workers.
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        
        # Read CSV in chunks
        for chunk_idx, chunk in enumerate(pd.read_csv(self.data_file, chunksize=self.chunksize)):
            chunk = chunk.iloc[worker_id::num_workers] # partition chunk rows
            chunk = chunk.sample(frac=1)  # shuffle chunk rows
            for _, row in chunk.iterrows():
                X, sequence_lengths = self._encode_name(row['name'])
                y = self._encode_country(row['alpha2'])
                yield X, y, sequence_lengths
                
def create_dataloaders(
    config: DataConfig,
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int,
    num_workers: int = 8
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation and test dataloaders."""
    
    dataset_kwargs = {
        'maximum_name_length': config.maximum_name_length,
        'vocabulary': config.vocabulary,
        'country_codes': config.country_codes,
        'country_mapping': config.country_mapping
    }
    
    train_data = NameNationalityDataStream(
        data_file=train_path,
        chunksize=batch_size,
        **dataset_kwargs
    )
    
    val_data = NameNationalityData(
        data_file=val_path,
        **dataset_kwargs
    )
    
    test_data = NameNationalityData(
        data_file=test_path,
        **dataset_kwargs
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