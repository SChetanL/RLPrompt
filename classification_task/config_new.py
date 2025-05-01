# utils.py

class Config:
    """
    Configuration class for prompt optimization.
    """
    def __init__(self):
        self.iteration_num = 10
        self.depth_limit = 6
        self.expand_width = 3
        self.exploration_weight = 1.0
        self.seed = 42