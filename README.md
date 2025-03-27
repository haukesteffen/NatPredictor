# Name Nationality Predictor

A PyTorch-based RNN model for predicting a person’s nationality based solely on their name. This project processes a large dataset of names (from multiple CSV files), encodes names at the character level, and trains a multi-layer RNN. For downstream tasks, predicted country labels can be remapped (e.g., to UN Geoscheme regions) to reduce the number of output classes. Uses a [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) Wrapper and [MLflow](https://mlflow.org/) for logging training experiments.

## Features

- **Data Preparation:**  
  - Original data from [here](https://github.com/philipperemy/name-dataset)
  - The original CSV files (~500M rows) were concatenated, shuffled and split into training (98%), validation (1%) and test (1%) data.
  - Uses an `IterableDataset` to stream training data in manageable chunks from the large CSV file.
  - Uses a `Dataset` to keep validation and test data in a single batch.
  - Constructs vocabularies from names (with character-to-index mapping) and encodes variable-length names with padding.
  - Constructs mappings for countries (alpha2-to-class, class-to-index and index-to-class) and encodes indices as one-hot-tensors.
  - Original labels (alpha2 country codes) can be mapped to broader regions (e.g., UN Geoscheme regions) to simplify downstream classification.
  
- **Model Architecture:**  
  - An RNN-based model with embedding layers, multi-layer RNN (with dropout), and a linear output layer.
  - Uses `pack_padded_sequence` to efficiently process sequences of varying lengths.
  
- **Inference Pipeline:**  
  - Tools for encoding a single name or batches of names and running inference are being worked on.
