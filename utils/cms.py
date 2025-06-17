import random
import mmh3

class CountMinSketch:
    def __init__(self, width=1000, depth=10, seed=42):
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]
        self.seed = seed
        self.hash_seeds = [random.randint(1, 10000) for _ in range(depth)]

    def add(self, key, count=1):
        for i in range(self.depth):
            idx = mmh3.hash(str(key), self.hash_seeds[i]) % self.width
            self.table[i][idx] += count

    def estimate(self, key):
        min_est = float('inf')
        for i in range(self.depth):
            idx = mmh3.hash(str(key), self.hash_seeds[i]) % self.width
            min_est = min(min_est, self.table[i][idx])
        return min_est
