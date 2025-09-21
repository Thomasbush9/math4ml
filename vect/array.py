import random
import array
from functools import reduce


class Array():
    def __init__(self, shape:tuple) -> None:
        self.shape = shape
        self.ndim = len(shape)
        multiply = lambda x, y: x*y
        self.size = reduce(multiply, shape)
        self.array = array.array("d", self.size)
        self.mem = memoryview(self.array)







