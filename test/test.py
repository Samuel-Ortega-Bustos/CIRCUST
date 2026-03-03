import pandas as pd
import numpy as np
import circust.preprocessing as prep

df = prep.load_expression_matrix("/home/samu/Documents/Universidad/TFG-Statistics/CIRCUST/data/raw/matrixIn.parquet",gene_column="gene_id")
result = prep.Preprocessor().run(df)