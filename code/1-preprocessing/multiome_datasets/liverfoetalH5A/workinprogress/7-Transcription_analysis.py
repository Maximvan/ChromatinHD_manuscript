
# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import scanpy as sc

import pathlib

import tqdm.auto as tqdm

import chromatinhd as chd
import polyptich as pp

import eyck
import scanpy as sc
import pandas as pd
import numpy as np
from gprofiler import GProfiler


# %%
import pickle
folder_root = chd.get_output()
folder_data = folder_root / "data"
folder_data_preproc = folder_data_preproc = folder_data / 'liverfoetalH'
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))

# %%
cells_of_interest = ["Macrophage_I", "Macrophage_II", "Macrophage_III", "Monocyte_I", "Monocyte_II"]
adata2 = adata[adata.obs["celltype"].isin(cells_of_interest)]

# %%
sc.tl.rank_genes_groups(
    adata, groupby="celltype", method="wilcoxon", key_added="wilcoxon", use_raw=False,
)

# %%
top_genes = {cluster: adata.uns["wilcoxon"]["names"][cluster] for cluster in adata.uns["wilcoxon"]["names"].dtype.names}
num_top_genes = 100 
top_genes_df = pd.DataFrame({f"Cluster {cluster}": top_genes[cluster][:num_top_genes] for cluster in top_genes})
target_clusters = ["Cluster Macrophage_I", "Cluster Macrophage_II", "Cluster Macrophage_III", "Cluster Monocyte_I", "Cluster Monocyte_II"]
all_target_cluster_genes = pd.concat([top_genes_df[col].dropna() for col in target_clusters]).unique().tolist()

# %%
gp = GProfiler(return_dataframe=True)

# %%
go_results = gp.profile(organism='hsapiens', query=all_target_cluster_genes,
                        #sources=['GO:BP', 'GO:MF', 'GO:CC'],
                        no_evidences=False)


# %%
adhesion_go_terms = go_results[
    (go_results['source'].isin(['GO:BP'])) &
    (go_results['name'].str.contains('adhesion|junction|integrin|cadherin|selectin|cell surface binding', case=False)) &
    ~(go_results['name'].str.contains('regulation of|positive regulation of|negative regulation of', case=False)) &
    (go_results['p_value'] < 0.05)
]


# %%
if not adhesion_go_terms.empty:
    print("\nSignificantly enriched adhesion-related GO terms:")
    print(adhesion_go_terms[['name', 'p_value', 'term_size', 'intersections']])


    adhesion_genes_from_go = set()
    for genes_str in adhesion_go_terms['intersections']:
        adhesion_genes_from_go.update(genes_str)

    adhesion_genes_from_go = list(adhesion_genes_from_go)
    print(f"\nIdentified {len(adhesion_genes_from_go)} adhesion-related genes from GO enrichment.")
    print("Example adhesion genes:", adhesion_genes_from_go[:10])

else:
    print("\nNo significantly enriched adhesion-related GO terms found with the current thresholds.")

# %%
adhesion_adata = adata[:, adata.var_names.isin(adhesion_genes_from_go)].copy()
adhesion_adata_monomac = adhesion_adata[adhesion_adata.obs["celltype"].isin(cells_of_interest)].copy()

# %%
def gene_id(adata, symbol):
    return adata.var.reset_index().set_index("symbol").loc[symbol, "gene_ids"]


def symbol(adata, gene):
    return adata.var.reset_index().set_index("gene_ids").loc[gene, "symbol"]

# %%
# Plotting options:
# a) Dot Plot: Good for showing average expression and proportion of expressing cells
sc.pl.dotplot(adhesion_adata_monomac, 
              symbol(adhesion_adata_monomac, adhesion_genes_from_go), 
              groupby="celltype", 
              layer="magic", 
              gene_symbols="symbol",
              dot_min=0.1, color_map='Reds',
              swap_axes=True,
              title='Expression of Adhesion Molecules in Monocytes and Macrophages')

# %%
# b) Heatmap: Shows expression of individual genes across cells (or averaged by group)
# For averaged by group:
sc.pl.heatmap(adhesion_adata_monomac,
              symbol(adhesion_adata_monomac, adhesion_genes_from_go), 
              groupby="celltype", 
              #layer="magic", 
              gene_symbols="symbol",
              cmap='viridis', dendrogram=False)

# %%
