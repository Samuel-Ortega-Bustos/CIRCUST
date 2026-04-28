import pandas as pd
import numpy as np
from circust.preprocessing import load_expression_matrix, Preprocessor
from circust.cpca import CPCA
from circust.core_genes import SEED_GENES_DEFAULT
from circust.visualization.cpca_plots import plot_pc_scatter,plot_gene_panels

df = load_expression_matrix("/home/samu/Documents/Universidad/TFG-Statistics/CIRCUST/data/matrixIn.parquet",gene_column="gene_id")
#df = load_expression_matrix("/home/samu/Documents/Universidad/TFG-Statistics/CIRCUST/data/BA46_glut_sample_no_minmax.csv")
result = Preprocessor().run(df)
resultCPCA  = CPCA().run(result.expr_norm)
fig1 = plot_pc_scatter(resultCPCA, title="BA46 glutamatergic")
fig1.savefig("cpca_scatter.png", dpi=150, bbox_inches="tight")

fig2 = plot_gene_panels(resultCPCA)
fig2.savefig("cpca_genes.png", dpi=150, bbox_inches="tight")