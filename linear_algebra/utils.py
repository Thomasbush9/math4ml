import random

class Matrix():
    def __init__(self, shape:tuple) -> None:
        self.shape = shape
        self.data = None
    def __repr__(self) -> str:
        return "\n".join(str(row) for row in self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    @classmethod
    def from_data(cls, data, copy=True, strict=True):
        if strict:
            assert all(type(val) == type(data[0]) for col in data for val in col), "Elements must be same type!"
        shape = (len(data), len(data[0]))
        temp = cls(shape)
        if copy:
            temp.data = data.copy()
        return temp

#TODO make it with a new matrix
    @property
    def T(self):
        rows, cols = self.shape
        m = [[] for _ in range(cols)]
        for row in self.data:
            for i, e in enumerate(row):
                m[i].append(e)
        self.data = m
    def __add__(self, x):
        # add dim checker
        if isinstance(x, int|float):
            temp_m = Matrix((self.shape))
            temp_m.data = [[e + x for e in col] for col in self.data]
            return temp_m
        if isinstance(x, Matrix):
            assert self.shape == x.shape, "Matrices must have same shape"
            data = [[e1+e2 for e1, e2 in zip(col1, col2)] for col1, col2 in zip(self.data, x.data)]
            shape= self.shape
            temp_m = Matrix(shape)
            temp_m.data= data
            return temp_m
    def __radd__(self, x):
    #TODO check the type to be the same in the future
        temp_m = Matrix(self.shape))
        temp_m.data = [[e + x for e in col] for col in self.data]
        return temp_m
    def __sub__(self, x):
        # add dim checker
        if isinstance(x, int|float):
            temp_m = Matrix((self.shape))
            temp_m.data = [[e - x for e in col] for col in self.data]
            return temp_m
        if isinstance(x, Matrix):
            assert self.shape == x.shape, "Matrices must have same shape"
            data = [[e1-e2 for e1, e2 in zip(col1, col2)] for col1, col2 in zip(self.data, x.data)]
            shape= self.shape
            temp_m = Matrix(shape)
            temp_m.data= data
            return temp_m




