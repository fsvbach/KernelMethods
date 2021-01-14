from abc import ABC, abstractmethod #abstract classes
import pandas as pd
import numpy as np

class Data(ABC):
    
    def __init__(self, index):
        self.index = index
        self.label = ""
        self.strings = None
        self.vectors = None

    @abstractmethod
    def name(self):
        pass

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name() == other.name()
        else:
            return False

    def as_strings(self):
        if self.strings is None:
            self.strings = pd.read_csv(f'Data/X{self.label}{self.index}.csv', index_col=0)['seq']
        return self.strings

    def as_bag_of_words(self):
        if self.vectors is None:
            self.vectors = np.array(pd.read_csv(f'Data/X{self.label}{self.index}_mat100.csv', header=None, sep=' '))
        return self.vectors

    def as_int_encoded_strings(self):
        to_int = { 'A' : 0, 'C': 1, 'G' : 2, 'T' : 3}
        convert_string = lambda s: list(map(lambda c: to_int[c], s))
        res = list(map(convert_string, self.as_strings()))
        return np.array(res)

class TestData(Data):

    def __init__(self, index):
        super().__init__(index)
        self.label = "te"
    
    def name(self):
        return f'test{self.index}'

class TrainingData(Data):

    def __init__(self, index):
        super().__init__(index)
        self.label = "tr"
        self.y = None

    def name(self):
        return f'train{self.index}'

    def labels(self):
        if self.y is None:
            y = pd.read_csv(f'Data/Y{self.label}{self.index}.csv', index_col=0)
            self.y = np.array(y['Bound'])
        return self.y

def load_data():
    X = [TrainingData(i) for i in range(3)]
    Z = [TestData(i) for i in range(3)]
    return (X, Z)
    