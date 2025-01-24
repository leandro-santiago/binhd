import torch
import torch.nn as nn
import torchhd
from torchhd import embeddings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from binhd.embeddings import ScatterCode, CategoricalEncoder
from binhd.datasets.statlog import Statlog
from binhd.classifiers import BinHD
from binhd.functional import multibundle

# Use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

dimension = 2000
num_levels = 100

dataset = Statlog()

# Transform categorical features
X_categorical = dataset.features[dataset.categorical_features]
categorical_encoder = CategoricalEncoder(dimension)
X_categorical = categorical_encoder.fit_transform(X_categorical)
X_categorical = torch.tensor(X_categorical.values)
X_cat_hv = categorical_encoder(X_categorical)

# Transform numeric features
X_numeric = dataset.features[dataset.numeric_features]
X_numeric = torch.tensor(X_numeric.values, dtype=torch.uint8).to(device)
min_val, max_val = dataset.get_min_max_values()
print(min_val, max_val)

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

y = list(dataset.labels)
y = torch.tensor(y).squeeze().to(device)

record_encode = RecordEncoder(dimension, dataset.num_features, num_levels, min_val, max_val)
record_encode = record_encode.to(device)


with torch.no_grad():
    X_hv = record_encode(X_cat_hv, X_numeric)

X_train, X_test, y_train, y_test = train_test_split(X_hv, y, test_size=0.3, random_state = 0)  

model = BinHD(dimension, dataset.num_classes)

with torch.no_grad():
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)  
    acc = accuracy_score(predictions, y_test)
    print("BinHD: Accuracy = ", acc)
