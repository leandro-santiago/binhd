import torch
import torch.nn as nn
import torchhd
from torchhd import embeddings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from binhd.embeddings import ScatterCode
from binhd.datasets.iris import Iris
from binhd.classifiers import BinHD

# Use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

iris = Iris()
dimension = 1000
num_levels = 100

min_val, max_val = iris.get_min_max_values()
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

X = iris.features
y = list(iris.labels)

record_encode = RecordEncoder(dimension, iris.num_features, num_levels, min_val, max_val)
record_encode = record_encode.to(device)

with torch.no_grad():
    samples = torch.tensor(X.values).to(device)
    labels = torch.tensor(y).squeeze().to(device)

    X_hv = record_encode(samples)

X_train, X_test, y_train, y_test = train_test_split(X_hv, labels, test_size=0.3, random_state = 0)  

model = BinHD(dimension, iris.num_classes)

with torch.no_grad():
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)  
    acc = accuracy_score(predictions, y_test)
    print("BinHD: Accuracy = ", acc)

    model.fit_adapt(X_train,y_train)
    predictions = model.predict(X_test)  
    acc = accuracy_score(predictions, y_test)
    print("BinHD - Adapt: Accuracy = ", acc)
    