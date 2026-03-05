import pandas as pd
import numpy as np
from circust.preprocessing import load_expression_matrix, Preprocessor
from circust.cpca import CPCA
from circust.constants import SEED_GENES_DEFAULT

df = load_expression_matrix("/home/samu/Documents/Universidad/TFG-Statistics/CIRCUST/data/raw/matrixIn.parquet",gene_column="gene_id")
result = Preprocessor().run(df)
result  = CPCA().run(result.expr_norm)