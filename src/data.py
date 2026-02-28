import numpy as np
from sklearn.model_selection import train_test_split
# Create the data set 

def create_data():
    # define the seed 
    rng = np.random.default_rng(5)

    # define the observations 
    x1 = rng.uniform(size = 500) - 0.5
    x2 = rng.uniform(size= 500) - 0.5 
    y = x1**2 -x2**2 > 0

    return x1, x2, y

