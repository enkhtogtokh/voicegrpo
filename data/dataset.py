import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import  TensorDataset
 

class Dataset:
    def __init__(self, dataset_path):
        """
        Constructor for the Dataset class.

        Parameters
        ----------
        dataset_path : str
            The path to the dataset Excel file.
        """
        self.dataset_path = dataset_path

    def prepare(self):
        
        """
        Prepares the dataset for use.

        Returns
        -------
        train_dataset : TensorDataset
            The training dataset.
        test_dataset : TensorDataset
            The testing dataset.
        """
        df = pd.read_excel(self.dataset_path)
        features, labels = df[['pitch', 'jitter', 'shimmer', 'hnr', 'age', 'disease_severity']], df['pathology']
 
        # --- Data Preprocessing ---
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to tensors
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train.values))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test.values))    

        return train_dataset, test_dataset
