import numpy as np

class Norm:
    def __init__(self, data):
        self.data=data
        if len(data.shape) == 1:
            # Reshape 1D array to 2D
            data = data.reshape(-1, 1)
            self.data = data
        
    
    def min_max_scaler(self):
        """
        Perform Min-Max Scaling to scale the data between 0 and 1.
        """
        min_val = np.min(self.data, axis=0)
        max_val = np.max(self.data, axis=0)
        
        # Avoid division by zero in case max == min
        scaled_data = (self.data - min_val) / (max_val - min_val)
        
        return scaled_data

    def standard_scaler(self):
        """
        Perform standard scaling (0 mean and unit variance).
        """
        mean = np.mean(self.data, axis=0)
        std_dev = np.std(self.data, axis=0)
        
        # Avoid division by zero in case std_dev == 0
        standardized_data = (self.data - mean) / (std_dev)
    
        return standardized_data

    def l1_normalize(self):
        """
        Perform L1 normalization.
        """
        l1_norm = np.sum(np.abs(self.data), axis=1, keepdims=True)
        
        # Avoid division by zero
        normalized_data = self.data / (l1_norm)
        
        return normalized_data

    def l2_normalize(self):
        """
        Perform L2 normalization.
        """
        l2_norm = np.sqrt(np.sum(self.data**2, axis=1, keepdims=True))
        
        # Avoid division by zero
        normalized_data = self.data / (l2_norm)
        
        return normalized_data