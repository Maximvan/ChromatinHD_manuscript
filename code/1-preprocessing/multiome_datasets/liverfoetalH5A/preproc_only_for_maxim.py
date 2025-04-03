# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import sys

import polyptich as pp
pp.setup_ipython()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import pickle

import scanpy as sc

import polyptich as pp
import eyck

# %%
folder_data = pp.paths.get_output() / "data" / "suo22_kc"
folder_data.mkdir(exist_ok=True, parents=True)

# %%
h5ad_file = folder_data / "adata.h5ad"
# h5ad_file.unlink()
if not h5ad_file.exists():
    # # !wget https://datasets.cellxgene.cziscience.com/66e520ac-ee32-4f34-a55b-ab99ca6fda80.h5ad -O {h5ad_file}
    os.system(
        f"wget https://datasets.cellxgene.cziscience.com/527c50aa-22d2-48df-b876-b0e1ef6dacd5.h5ad -O {h5ad_file}"
    )

# %%
adataD = sc.read_h5ad(h5ad_file)

# %%
adataD = adataD[adataD.obs["tissue"] == "liver"]


# %%
def gene_id(adata, symbol):
    return adata.var.reset_index().set_index("symbol").loc[symbol, "gene"]


def symbol(adata, gene):
    return adata.var.reset_index().set_index("gene").loc[gene, "symbol"]


# %%
adataD.var["symbol"] = adataD.var["GeneName"]
adataD.var.index.name = "gene"

# %%
sc.pl.umap(adataD, color=gene_id(adataD, ["NR1H3", "CLEC4F"]))

# %%
pd.crosstab(adataD.obs["donor_id"], adataD.obs["development_stage"])

# %%
sc.pl.umap(adataD, color="donor_id")
sc.pl.umap(adataD, color="development_stage")

# %% [markdown]
# ## Focus on one donor

# %%
adata2 = adataD[(adataD.obs["donor_id"] == "F61") & (adataD.obs["Sort_id"] == "CD45P")].copy()
adata2 = adata2.raw.to_adata()
counts = adata2.X.copy()

# %%
adata2.raw = adata2
sc.pp.normalize_total(adata2)
sc.pp.log1p(adata2)
sc.pp.highly_variable_genes(adata2, n_top_genes=2000)
sc.pp.pca(adata2)
sc.pp.neighbors(adata2)
sc.tl.umap(adata2)

# %%
adata2.layers["counts"] = counts

# %%
adata2.obs.iloc[0]

# %%
adata2.obs["is_maternal_contaminant"] = adata2.obs["is_maternal_contaminant"].astype("category")

# %%
sc.pl.umap(
    adata2,
    color=[
        "Sort_id",
        "is_maternal_contaminant",
        "assay",
        "sex",
        "development_stage",
        "celltype_annotation",
    ],
)

# %%
adata2.obs["logn_counts"] = np.log1p(adata2.obs["n_counts"])

# %%
sc.pl.umap(adata2, color="logn_counts")

# %%
sc.pl.umap(adata2, color=["scvi_clusters"])

# %%
symbols = [
    "CDK1",
    "CLEC4F",
    "MARCO",
    "TIMD4",
    "NR1H3",
    "CDK1",
    "LYVE1",
    "MERTK",
    "FABP5",
    "CD38",
    "APOE",
    "FTH1",
    "MT1H",
    "SLC40A1",
    "HLA-DPB1",
    "CD34",
    "ID3",
    "SPIC",
    "C1QA",
    "S100A6",
    "NFKBIA",
    "ZMIZ1",
    "PPARG",
    "APOE",
]

symbolsB = [
    'CD34', 'SPINK2', 'PROM1', 'MLLT3',
    'TESPA1', 'GATA2', 'GATA1', 'ALAS2', 
    'HBA1', 'HDC', 'MKI67', 'TESPA1', 
    'GATA2', 'HDC', 'TESPA1', 'GATA2', 
    'ITGA2B', 'CD34', 'SPINK2', 'PROM1', 
    'MLLT3', 'ITGA2B', 'MPO', 'AZU1', 
    'SPI1', 'SPI1', 'FCN1', 'VCAN', 'CTSB',
    'SPI1', 'FCN1', 'VCAN', 'CTSB', 
    'IRF8', 'CLEC10A', 'MLLT3', 'IL2RB', 
    'NKG7', 'PRF1', 'IL7R', 'PAX5', 'IGHM', 'IRF8', 'CLEC4C'
]
sc.pl.umap(adata2, color=gene_id(adataD, symbolsB), title=symbolsB)

# %% [markdown]
# ### Exploring main diversity in PCs
# %%
pc_markers = []
for pc_ix in range(20):
    pc_markers.append(
        {"pc": pc_ix, "gene": adata2.var.index[np.argmax(adata2.varm["PCs"][:, pc_ix])]}
    )
    pc_markers.append(
        {"pc": pc_ix, "gene": adata2.var.index[np.argmin(adata2.varm["PCs"][:, pc_ix])]}
    )
pc_markers = pd.DataFrame(pc_markers)
pc_markers["symbol"] = symbol(adataD, pc_markers["gene"]).values
pc_markers["label"] = pc_markers["pc"].astype(str) + " " + pc_markers["symbol"].astype(str)

# %%
sc.pl.umap(adata2, color=pc_markers["gene"], title=pc_markers["label"])

# %% [markdown]
# ## Subselect only on KCs manually

# %%
adata2.var["symbol"] = adata2.var["feature_name"]

# %%
adata3 = adata2[adata2.obs["scvi_clusters"].isin([10, 5, 0, 12])]

# %%
gene_ids = ['ENSG00000196735',
 'ENSG00000186431',
 'ENSG00000125538',
 'ENSG00000005381',
 'ENSG00000018280',
 'ENSG00000010327',
 'ENSG00000196562',
 'ENSG00000178789',
 'ENSG00000182578',
 'ENSG00000216490',
 'ENSG00000179344',
 'ENSG00000085265',
 'ENSG00000140678',
 'ENSG00000110446',
 'ENSG00000100095',
 'ENSG00000173391',
 'ENSG00000107551',
 'ENSG00000172243',
 'ENSG00000276085',
 'ENSG00000132514',
 'ENSG00000152315',
 'ENSG00000131724',
 'ENSG00000110324',
 'ENSG00000124731',
 'ENSG00000131042',
 'ENSG00000177989',
 'ENSG00000204577',
 'ENSG00000116701']
sc.pl.umap(adata3, color=gene_ids, title=symbol(adataD, gene_ids))

# %%
genes_oi = []
for i in range(20):
    g = adata3.var.index[np.argmax(adata3.varm["PCs"][:, i])]
    if g not in genes_oi:
        genes_oi.append(g)
    g = adata3.var.index[np.argmin(adata3.varm["PCs"][:, i])]
    if g not in genes_oi:
        genes_oi.append(g)

# %%
eyck.modalities.transcriptome.plot_umap(
    adata3,
    ["celltype_annotation", *genes_oi],
).display()
# %%
