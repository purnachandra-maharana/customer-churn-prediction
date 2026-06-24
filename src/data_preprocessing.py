"""
Data Preprocessing Module
"""

import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load dataset
    """
    return pd.read_csv(file_path)


def preprocess_data(df):
    """
    Data cleaning and preprocessing
    """
    print("Preprocessing started...")

    # Future preprocessing code here

    return df


if __name__ == "__main__":
    print("Data Preprocessing Module")
