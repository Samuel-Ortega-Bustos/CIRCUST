import pandas as pd
import numpy as np
from preprocessing import load_expression_matrix

df = load_expression_matrix("/home/samu/Documents/Universidad/TFG-Statistics/CIRCUST/data/raw/matrixIn.parquet")
print(df.index.astype(str))