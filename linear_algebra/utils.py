import random


class Matrix():
    def __init__(self, shape:tuple) -> None:
        self.shape = shape
        rows, cols = shape
        self.data = [[random.uniform(0, 1) for _ in range(cols)] for _ in range(rows)]
    def __repr__(self) -> str:
        return "\n".join(str(row) for row in self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    @property
    def T(self):
        rows, cols = self.shape
        m = [[] for _ in range(cols)]
        for row in self.data:
            for i, e in enumerate(row):
                m[i].append(e)
        self.data = m
    def __add__(self, x):
        assert len(self.shape) == len(x.shape), "Matrices must have same shape"
        # add dim checker
        if isinstance(x, Matrix):
            data = [[e1+e2 for e1, e2 in zip(col1, col2)] for col1, col2 in zip(self.data, x.data)]
            shape= self.shape
            temp_m = Matrix(shape)
            temp_m.data= data
            return temp_m




