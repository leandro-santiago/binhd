import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
import torchhd
from torchhd import embeddings
from sklearn.model_selection import train_test_split

from binhd.embeddings import ScatterCode, CategoricalEncoder
from binhd.datasets.adult import Adult
from binhd.datasets import BaseDataset
from binhd.classifiers import BinHD
from binhd.functional import multibundle

import pandas as pd

# Use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

dimension = 1000
num_levels = 100
batch_size = 1000

dataset = Adult()
min_val, max_val = dataset.get_min_max_values()

X_categorical = dataset.features[dataset.categorical_features]
categorical_encoder = CategoricalEncoder(dimension)
X_categorical = categorical_encoder.fit_transform(X_categorical)
X_numeric = dataset.features[dataset.numeric_features]
X = pd.concat([X_categorical, X_numeric], axis=1) 

y = list(dataset.labels)
y = torch.tensor(y).squeeze().to(device)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)  

train_dataset = BaseDataset(X_train.values, y_train)
test_dataset = BaseDataset(X_test.values, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class RecordEncoder(nn.Module):
    def __init__(self, out_features, size, levels, low, high):
        super(RecordEncoder, self).__init__() 
        self.position = embeddings.Random(size, out_features, vsa="BSC", dtype=torch.int8)
        self.value = ScatterCode(levels, out_features, low = low, high = high)
    
    def forward(self, x_categorical, x_numeric):        
        sample_categorical = torchhd.bind(self.position.weight[:x_categorical.shape[1]], x_categorical)        
        sample_numeric= torchhd.bind(self.position.weight[x_categorical.shape[1]:], self.value(x_numeric))                
        sample_hv = multibundle(sample_categorical, sample_numeric)                      
        return sample_hv
    
record_encode = RecordEncoder(dimension, dataset.num_features, num_levels, min_val, max_val)
record_encode = record_encode.to(device)

model = BinHD(dimension, dataset.num_classes)

with torch.no_grad():
    for samples, labels in train_loader:
        samples = samples.to(device)
        labels = labels.to(device)
        samples_cat_hv = categorical_encoder(samples[:,:categorical_encoder.num_features])
        samples_hv = record_encode(samples_cat_hv, samples[:, categorical_encoder.num_features:])
        model.fit(samples_hv, labels)

    accuracy = torchmetrics.Accuracy("multiclass", num_classes=dataset.num_classes)    

    for samples, labels in test_loader:
        samples = samples.to(device)
        labels = labels.to(device)    
        samples_cat_hv = categorical_encoder(samples[:,:categorical_encoder.num_features])
        samples_hv = record_encode(samples_cat_hv, samples[:, categorical_encoder.num_features:])        
        predictions = model.predict(samples_hv)  
        accuracy.update(predictions, labels)
    
    acc = accuracy.compute().item()    
    print("BinHD: Accuracy = ", acc)
