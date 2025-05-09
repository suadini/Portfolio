from datasets import load_dataset

def load_landmark_dataset():
    """
    Loads the Azerbaijan landmark dataset from Hugging Face Hub.

    Returns:
    - dataset (DatasetDict): Hugging Face dataset object.
    """
    dataset = load_dataset("khaleed-mammad/azerbaijan-landmarks-dataset")
    return dataset
