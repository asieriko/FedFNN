# FedFNN
This is the codes for my paper "Federated Fuzzy Neural Network with Evolutionary Rule Learning"

## Modifications:

### 1. Data loading

I've modified it to work with predefined client partitions described in a json file and with the data from a keel .dat file.

json file:

```json

{
  "C1": {
    "train": [...],
    "test": [...]
  },
  "C2": {
    "train": [...],
    "test": [...]
  },
  "C3": {
    "train": [...],
    "test": [...]
  }
}
```

These indices refer to the row positions of the dataset when loaded from the corresponding .dat file.

Example:

```
@relation iris
@attribute SepalLength real [4.3, 7.9]
@attribute SepalWidth real [2.0, 4.4]
@attribute PetalLength real [1.0, 6.9]
@attribute PetalWidth real [0.1, 2.5]
@attribute Class {Iris-setosa, Iris-versicolor, Iris-virginica}
@inputs SepalLength, SepalWidth, PetalLength, PetalWidth
@outputs Class
@data
5.1, 3.5, 1.4, 0.2, Iris-setosa
4.9, 3.0, 1.4, 0.2, Iris-setosa
4.6, 3.1, 1.5, 0.2, Iris-setosa
...
```

### 2. Results

Now the code also reports f1 score and the complexity of the learned rule base (number of rules and antencedents)
