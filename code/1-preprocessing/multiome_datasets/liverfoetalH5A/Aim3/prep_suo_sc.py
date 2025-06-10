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
grouped_palette = [
    '#cc4c02',  # Monocyte
    '#fe9929',  # Early Macrophage
    '#fdd49e',  # Late Macrophage
]
desired_order = [
    'Monocyte', 
    'Early Macrophage',
    'Late Macrophage'
]

# %%
grouped_palette_original = [
    "#021832",
    "#052D57",
    '#2D5379',
    "#0028A1",
    '#1860b8',
    '#5298C9',
    '#8BBBDB', 
    "#CDE4F0"
]

desired_order_original = [
    "MONOCYTE_I_CXCR4",
    "MONOCYTE_II_CCR2",
    "MONOCYTE_III_IL1B",
    "MACROPHAGE_MHCII_HIGH",
    "MACROPHAGE_IRON_RECYCLING",
    "MACROPHAGE_PROLIFERATING",
    "MACROPHAGE_KUPFFER_LIKE",
    "MACROPHAGE_ERY"
]
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
pd.crosstab(adataD.obs["donor_id"], adataD.obs["development_stage"])

# %%
sc.pl.umap(adataD, color="donor_id")
sc.pl.umap(adataD, color="development_stage")


# %% [markdown]
# ## Focus on one donor; F61 8th week post-fertilization human stage

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
sc.pl.umap(adata2, color="donor_id")
sc.pl.umap(adata2, color="development_stage")

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

# %% [markdown]
# ## Subselect only on KCs manually
# %%
adata2.var["symbol"] = adata2.var["feature_name"]
adata3 = adata2[adata2.obs["celltype_annotation"].isin(["MONOCYTE_III_IL1B", "MONOCYTE_II_CCR2", "MONOCYTE_I_CXCR4", "MACROPHAGE_ERY", "MACROPHAGE_IRON_RECYCLING", "MACROPHAGE_KUPFFER_LIKE", "MACROPHAGE_MHCII_HIGH", "MACROPHAGE_PROLIFERATING"])]

# %%
adata3 = adata3.raw.to_adata()
counts = adata3.X.copy()
adata3.raw = adata3
sc.pp.normalize_total(adata3)
#sc.pp.log1p(adata3)
sc.pp.highly_variable_genes(adata3, n_top_genes=2000)
sc.pp.pca(adata3)
sc.pp.neighbors(adata3)
sc.tl.umap(adata3)

# %%
sc.pl.umap(adata3, color=["celltype_annotation"])

# %%
# leiden
sc.tl.leiden(adata3, resolution=0.4, flavor='igraph', n_iterations=-1)
sc.pl.umap(adata3, color=["leiden"], title=["leiden"])    

# %%
cluster_to_celltype = {
    '0': 'Late Macrophage',
    '1': 'Late Macrophage',
    '2': 'Late Macrophage',
    '3': 'Late Macrophage',
    '4': 'Early Macrophage',
    '5': 'Monocyte'
}

# %%
adata3.obs['annotation'] = adata3.obs['leiden'].map(cluster_to_celltype)
adata3.obs['annotation'] = adata3.obs['annotation'].astype('category')
adata3.obs['annotation'] = adata3.obs['annotation'].cat.reorder_categories(desired_order)
sc.pl.umap(adata3, color=["annotation"], palette=grouped_palette)

# %%
sc.pl.umap(adata3, color=["celltype_annotation", "annotation"], title=["celltype_annotation", "annotation"])

# %%
pickle.dump(adata3, (folder_data / "adata_sc_8_annotated.pkl").open("wb"))

# %% [markdown]
# ## Focus on one donor; F23 11th week post-fertilization human stage

# %%
adata2 = adataD[(adataD.obs["donor_id"] == "F23") & (adataD.obs["Sort_id"] == "CD45P")].copy()
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
sc.pl.umap(adata2, color="donor_id")
sc.pl.umap(adata2, color="development_stage")

# %%
adata2.layers["counts"] = counts

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

# %% [markdown]
# ## Subselect only on KCs manually
# %%
adata2.var["symbol"] = adata2.var["feature_name"]
adata3 = adata2[adata2.obs["celltype_annotation"].isin(["MONOCYTE_III_IL1B", "MONOCYTE_II_CCR2", "MONOCYTE_I_CXCR4", "MACROPHAGE_ERY", "MACROPHAGE_IRON_RECYCLING", "MACROPHAGE_KUPFFER_LIKE", "MACROPHAGE_MHCII_HIGH", "MACROPHAGE_PROLIFERATING"])]

# %%
adata3 = adata3.raw.to_adata()
counts = adata3.X.copy()
adata3.raw = adata3
sc.pp.normalize_total(adata3)
#sc.pp.log1p(adata3)
sc.pp.highly_variable_genes(adata3, n_top_genes=2000)
sc.pp.pca(adata3)
sc.pp.neighbors(adata3)
sc.tl.umap(adata3)

# %%
sc.pl.umap(adata3, color=["celltype_annotation"])

# %%
# leiden
sc.tl.leiden(adata3, resolution=0.4, flavor='igraph', n_iterations=-1)
sc.pl.umap(adata3, color=["leiden"], title=["leiden"])    

# %%
cluster_to_celltype = {
    '0': 'Late Macrophage',
    '1': 'Late Macrophage',
    '2': 'Late Macrophage',
    '3': 'Late Macrophage',
    '4': 'Early Macrophage',
    '5': 'Monocyte'
}

# %%
adata3.obs['annotation'] = adata3.obs['leiden'].map(cluster_to_celltype)
adata3.obs['annotation'] = adata3.obs['annotation'].astype('category')
adata3.obs['annotation'] = adata3.obs['annotation'].cat.reorder_categories(desired_order)
sc.pl.umap(adata3, color=["annotation"], palette=grouped_palette)

# %%
sc.pl.umap(adata3, color=["celltype_annotation", "annotation"], title=["celltype_annotation", "annotation"])

# %%
pickle.dump(adata3, (folder_data / "adata_sc_11_annotated.pkl").open("wb"))


# %% [markdown]
# ## Focus on one donor; F38 14th week post-fertilization human stage

# %%
adata2 = adataD[(adataD.obs["donor_id"] == "F38") & (adataD.obs["Sort_id"] == "CD45P")].copy()
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
sc.pl.umap(adata2, color="donor_id")
sc.pl.umap(adata2, color="development_stage")

# %%
adata2.layers["counts"] = counts

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

# %% [markdown]
# ## Subselect only on KCs manually
# %%
adata2.var["symbol"] = adata2.var["feature_name"]
adata3 = adata2[adata2.obs["celltype_annotation"].isin(["MONOCYTE_III_IL1B", "MONOCYTE_II_CCR2", "MONOCYTE_I_CXCR4", "MACROPHAGE_ERY", "MACROPHAGE_IRON_RECYCLING", "MACROPHAGE_KUPFFER_LIKE", "MACROPHAGE_MHCII_HIGH", "MACROPHAGE_PROLIFERATING"])]

# %%
adata3 = adata3.raw.to_adata()
counts = adata3.X.copy()
adata3.raw = adata3
sc.pp.normalize_total(adata3)
#sc.pp.log1p(adata3)
sc.pp.highly_variable_genes(adata3, n_top_genes=2000)
sc.pp.pca(adata3)
sc.pp.neighbors(adata3)
sc.tl.umap(adata3)

# %%
sc.pl.umap(adata3, color=["celltype_annotation"])

# %%
# leiden
sc.tl.leiden(adata3, resolution=0.3, flavor='igraph', n_iterations=-1)
sc.pl.umap(adata3, color=["leiden"], title=["leiden"])    

# %%
cluster_to_celltype = {
    '0': 'Late Macrophage',
    '1': 'Late Macrophage',
    '2': 'Late Macrophage',
    '3': 'Late Macrophage',
    '6': 'Late Macrophage',
    '4': 'Early Macrophage',
    '5': 'Monocyte'
}

# %%
adata3.obs['annotation'] = adata3.obs['leiden'].map(cluster_to_celltype)
adata3.obs['annotation'] = adata3.obs['annotation'].astype('category')
adata3.obs['annotation'] = adata3.obs['annotation'].cat.reorder_categories(desired_order)
sc.pl.umap(adata3, color=["annotation"], palette=grouped_palette)

# %%
sc.pl.umap(adata3, color=["celltype_annotation", "annotation"], title=["celltype_annotation", "annotation"])

# %%
pickle.dump(adata3, (folder_data / "adata_sc_14_annotated.pkl").open("wb"))

# %%

# %% [markdown]
# ## Focus on one donor; F41 17th week post-fertilization human stage

# %%
adata2 = adataD[(adataD.obs["donor_id"] == "F41") & (adataD.obs["Sort_id"] == "CD45P")].copy()
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
sc.pl.umap(adata2, color="donor_id")
sc.pl.umap(adata2, color="development_stage")

# %%
adata2.layers["counts"] = counts

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

# %% [markdown]
# ## Subselect only on KCs manually
# %%
adata2.var["symbol"] = adata2.var["feature_name"]
adata3 = adata2[adata2.obs["celltype_annotation"].isin(["MONOCYTE_III_IL1B", "MONOCYTE_II_CCR2", "MONOCYTE_I_CXCR4", "MACROPHAGE_ERY", "MACROPHAGE_IRON_RECYCLING", "MACROPHAGE_KUPFFER_LIKE", "MACROPHAGE_MHCII_HIGH", "MACROPHAGE_PROLIFERATING"])]
adata3 = adata3[adata3.obs["assay"] == "10x 3' v2"]

# %%
adata3 = adata3.raw.to_adata()
counts = adata3.X.copy()
adata3.raw = adata3
sc.pp.normalize_total(adata3)
#sc.pp.log1p(adata3)
sc.pp.highly_variable_genes(adata3, n_top_genes=2000)
sc.pp.pca(adata3)
sc.pp.neighbors(adata3)
sc.tl.umap(adata3)

# %%
sc.pl.umap(adata3, color=["celltype_annotation"])

# %%
# leiden
sc.tl.leiden(adata3, resolution=0.4, flavor='igraph', n_iterations=-1)
sc.pl.umap(adata3, color=["leiden"], title=["leiden"])    

# %%
cluster_to_celltype = {
    '0': 'Late Macrophage',
    '1': 'Late Macrophage',
    '3': 'Late Macrophage',
    '2': 'Early Macrophage',
    '4': 'Monocyte'
}

# %%
adata3.obs['annotation'] = adata3.obs['leiden'].map(cluster_to_celltype)
adata3.obs['annotation'] = adata3.obs['annotation'].astype('category')
adata3.obs['annotation'] = adata3.obs['annotation'].cat.reorder_categories(desired_order)
sc.pl.umap(adata3, color=["annotation"], palette=grouped_palette)

# %%
sc.pl.umap(adata3, color=["celltype_annotation", "annotation"], title=["celltype_annotation", "annotation"])

# %%
pickle.dump(adata3, (folder_data / "adata_sc_17_annotated.pkl").open("wb"))



# %%
adata = pickle.load((folder_data / "adata_sc_17_annotated.pkl").open("rb"))
adata.obs['celltype_annotation'] = adata.obs['celltype_annotation'].astype('category')
adata.obs['celltype_annotation'] = adata.obs['celltype_annotation'].cat.reorder_categories(desired_order_original)
sc.pl.umap(adata, color=["celltype_annotation"], palette=grouped_palette_original, frameon=False, legend_loc='left margin')


# %%
