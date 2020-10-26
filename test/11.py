import pandas as pd
import numpy as np
@profile
def shuffle():
	df = pd.DataFrame(np.random.randn(100, 1000000))
	df = df.sample(frac=1).reset_index(drop=True)

if __name__ == '__main__':
	shuffle()