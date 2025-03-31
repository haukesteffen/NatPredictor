from pathlib import Path
import json
import pandas as pd
from typing import Set, Tuple, Dict, Any, Optional
import country_converter as coco
from lightning.pytorch.loggers import MLFlowLogger

def _create_vocabulary_and_codes(
    train_path: Path,
    vocab_path: Path,
    country_codes_path: Path,
    chunksize: int = 100_000
) -> Tuple[Set[str], Set[str]]:
    """Create vocabulary and country codes files from training data.
    
    Args:
        train_path: Path to training data CSV
        vocab_path: Path to save vocabulary
        country_codes_path: Path to save country codes
        chunksize: Number of rows to process at a time
        
    Returns:
        Tuple of (vocabulary set, country codes set)
    """
    vocabulary = set()
    country_codes = set()
    
    # Process CSV in chunks to handle large files
    for chunk in pd.read_csv(train_path, chunksize=chunksize):
        # Add new characters from names
        for name in chunk['name']:
            vocabulary.update(str(name))
        
        # Add new country codes
        country_codes.update(chunk['alpha2'].unique())
    
    # Sort and save vocabulary
    sorted_vocab = sorted(vocabulary)
    with vocab_path.open('w', encoding='utf-8') as f:
        f.write(''.join(sorted_vocab))
    
    # Sort and save country codes
    sorted_codes = sorted(country_codes)
    with country_codes_path.open('w', encoding='utf-8') as f:
        f.write('\n'.join(sorted_codes))
    return sorted_vocab, sorted_codes

def _create_and_save_mappings(
    vocab_path: Path,
    country_codes_path: Path,
    mappings_path: Path,
    target_class: str = 'UNregion'
) -> Dict[str, Any]:
    """Create and save all mappings required for the model.
    
    Args:
        vocab_path: Path to vocabulary file
        country_codes_path: Path to country codes file
        mappings_path: Path to save mappings
        target_class: Target classification type
    """
    # Load vocabulary
    with vocab_path.open('r') as f:
        vocabulary = f.read()
    
    # Create character mappings (with 0 reserved for padding)
    char_to_index = {char: idx for idx, char in enumerate(sorted(vocabulary), 1)}
    
    # Load country codes
    with country_codes_path.open('r') as f:
        country_codes = f.read().splitlines()
    
    # Create country mappings (with 0 reserved for unknown)
    country_mapping = {cc: coco.convert(names=cc, to=target_class) 
                      for cc in country_codes}
    output_classes = sorted(set(country_mapping.values()))
    class_to_index = {c: i for i, c in enumerate(output_classes, 1)}
    alpha2_to_index = {
        alpha2: class_to_index[country_mapping[alpha2]]
        for alpha2 in country_codes
    }
    
    # Save mappings
    mappings = {
        'character': {
            'char_to_index': char_to_index,
        },
        'country': {
            'class_to_index': class_to_index,
            'country_mapping': country_mapping,
            'alpha2_to_index': alpha2_to_index
        }
    }
    
    mappings_path.parent.mkdir(parents=True, exist_ok=True)
    with mappings_path.open('w') as f:
        json.dump(mappings, f, indent=2)
    return mappings

def initialize_metadata(config: Dict[str, Any], mlflow_logger: Optional[MLFlowLogger] = None) -> None:
    """Initialize the project by creating necessary files if they don't exist."""
    # Get paths from config
    vocab_path = Path(config.metadata_parameters.vocab_path)
    country_codes_path = Path(config.metadata_parameters.country_codes_path)
    mappings_path = Path(config.metadata_parameters.mappings_path)
    train_path = Path(config.data_parameters.train_path)
    
    # Check if vocabulary and country codes files need to be created
    if not vocab_path.exists() or not country_codes_path.exists():
        print("Creating vocabulary and country codes files...")
        sorted_vocab, sorted_codes = _create_vocabulary_and_codes(
            train_path=train_path,
            vocab_path=vocab_path,
            country_codes_path=country_codes_path
        )
    else:
        # Read vocabulary and country codes from existing files
        print("Vocabulary and country codes files already exist.")
        with open(vocab_path, 'r') as f:
            sorted_vocab = f.read()
        with open(country_codes_path, 'r') as f:
            sorted_codes = f.read().splitlines() 
    
    # Check if mappings file needs to be created
    if not mappings_path.exists():
        print("Creating mappings file...")
        mappings = _create_and_save_mappings(
            vocab_path=vocab_path,
            country_codes_path=country_codes_path,
            mappings_path=mappings_path,
            target_class=config.metadata_parameters.target_class
        )
    else:
        print("Mappings file already exists.")
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)

    # Log files to MLflow if logger is provided
    if mlflow_logger:
        mlflow_logger.experiment.log_artifact(
            local_path=str(vocab_path),
            run_id=mlflow_logger.run_id
        )
        mlflow_logger.experiment.log_artifact(
            local_path=str(country_codes_path),
            run_id=mlflow_logger.run_id
        )
        mlflow_logger.experiment.log_artifact(
            local_path=str(mappings_path),
            run_id=mlflow_logger.run_id
        )
    print("Initialization complete!")
    return sorted_vocab, sorted_codes, mappings

if __name__ == '__main__':
    import yaml
    
    # Load config from the same location main.py uses it
    project_root = Path(__file__).parent.parent
    with open(project_root / 'config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    initialize_metadata(config)