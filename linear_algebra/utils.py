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




