import torch
import torch.nn as nn
import torch.utils.data as data
import torchhd
from torchhd import embeddings
from torchhd.tensors.bsc import BSCTensor
from torchhd.models import Centroid
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from binhd.embeddings import ScatterCode
from binhd.datasets.iris import Iris
from binhd.classifiers import BinHD, NeuralHD

# Use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

dataset = Iris()

X = dataset.features
X = torch.tensor(X.values).to(dtype=torch.float32, device=device)
y = dataset.labels
y = torch.tensor(y).squeeze().to(device)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)  

dimension = 1000
num_levels = 100

min_val, max_val = dataset.get_min_max_values()
print(min_val, max_val)

class RecordEncoder(nn.Module):
    def __init__(self, out_features, size, levels, low, high):
        super(RecordEncoder, self).__init__() 
        self.position = embeddings.Random(size, out_features, vsa="BSC", dtype=torch.uint8)
        self.value = ScatterCode(levels, out_features, low = low, high = high)
    
    def forward(self, x):
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return sample_hv

class NGramEncoder(nn.Module):
    def __init__(self, out_features, levels, low, high):
        super(NGramEncoder, self).__init__()
        self.value = ScatterCode(levels, out_features, low = low, high = high)              

    def forward(self, x, oper = "bind"):
        if oper == "bind":
            sample_hv = torchhd.bind_sequence(self.value(x))
        elif oper == "bundle":
            sample_hv = torchhd.bundle_sequence(self.value(x))
        return sample_hv

class RandomProjectionEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(RandomProjectionEncoder, self).__init__() 
        self.projection = embeddings.Sinusoid(in_features, out_features)        
        #self.projection = embeddings.Projection(in_features, out_features)        
    
    def forward(self, x):
        sample_hv = self.projection(x)
        return sample_hv

    def forward_bin(self, x):
        sample_hv = self.projection(x)
        return torch.where(sample_hv > 0, 1, 0).to(dtype = torch.int8).as_subclass(BSCTensor)        
    
record_encode = RecordEncoder(dimension, dataset.num_features, num_levels, min_val, max_val)
record_encode = record_encode.to(device)

ngram_encode = NGramEncoder(dimension, num_levels, min_val, max_val)
ngram_encode = ngram_encode.to(device)

projection_encode = RandomProjectionEncoder(dataset.num_features, dimension)
projection_encode = projection_encode.to(device)

model_record = BinHD(dimension, dataset.num_classes)
model_ngram = BinHD(dimension, dataset.num_classes)
model_ngram2 = BinHD(dimension, dataset.num_classes)

model_projection = Centroid(dimension, dataset.num_classes)
model_projection2 = BinHD(dimension, dataset.num_classes)

model_neural = NeuralHD(dataset.num_features, dimension, dataset.num_classes)

with torch.no_grad():
    
    X_train_record = record_encode(X_train)
    X_test_record = record_encode(X_test)
    model_record.fit(X_train_record,y_train)
    predictions = model_record.predict(X_test_record)  
    acc = accuracy_score(predictions, y_test)
    print("BinHD - Record-based Enconding: Accuracy = ", acc)

    X_train_ngram = ngram_encode(X_train)
    X_test_ngram = ngram_encode(X_test)
    model_ngram.fit(X_train_ngram,y_train)
    predictions = model_ngram.predict(X_test_ngram)  
    acc = accuracy_score(predictions, y_test)
    print("BinHD - Ngram Enconding (bind): Accuracy = ", acc)

    X_train_ngram = ngram_encode(X_train, oper = "bundle")
    X_test_ngram = ngram_encode(X_test, oper = "bundle")
    model_ngram2.fit(X_train_ngram,y_train)
    predictions = model_ngram2.predict(X_test_ngram)  
    acc = accuracy_score(predictions, y_test)
    print("BinHD - Ngram Enconding (bundle): Accuracy = ", acc)

    X_train_proj = projection_encode(X_train)      
    X_test_proj = projection_encode(X_test)  
    model_projection.add(X_train_proj,y_train) 
    predictions = torch.argmax(model_projection(X_test_proj), dim=-1)
    acc = accuracy_score(predictions, y_test)
    print("Centroid - Random Projection (Sinusoid): Accuracy = ", acc)  
    
    X_train_proj = projection_encode.forward_bin(X_train)      
    X_test_proj = projection_encode.forward_bin(X_test) 
    model_projection2.fit(X_train_proj,y_train)        
    predictions = model_projection2.predict(X_test_proj)  
    acc = accuracy_score(predictions, y_test)
    print("BinHD - Random Projection (Sinusoid): Accuracy = ", acc)

    #X_train_proj = projection_encode(X_train)      
    #X_test_proj = projection_encode(X_test) 
    model_neural.fit(X_train,y_train)        
    predictions = model_neural.predict(X_test)  
    acc = accuracy_score(predictions, y_test)
    print("NeuralHD - Random Projection (Sinusoid): Accuracy = ", acc)     
    
