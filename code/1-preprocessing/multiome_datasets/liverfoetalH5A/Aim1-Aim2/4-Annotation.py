# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: chromatinhd
#     language: python
#     name: python3
# ---

# %%
import polyptich as pp
pp.setup_ipython()

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import pickle

import scanpy as sc

import tqdm.auto as tqdm
import io

import chromatinhd as chd

import magic
import eyck
import scipy.sparse


# %%
sc.settings.verbosity = 0 # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=150, facecolor="white", figsize=[4,4])
# ugent_palette = [
#     "#1e62c7ff", "#3373ccff", "#4a83d1ff", "#78a1deff", "#6191d7ff",
#     "#8eb1e3ff", "#a4c1e8ff", "#bbcfeeff", "#d2def3ff", "#e7eff9ff"
# ]
# accent_colour = ["#2D8CA8"]
# sns.set_palette(ugent_palette)

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "liverfoetalH"
genome = "hg38"
organism = "hs"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

folder_dataset = chd.get_output() / "datasets" / dataset_name

# %% [markdown]
# ## Load in data

# %%
adata = pickle.load((folder_data_preproc / "adata_filtered.pkl").open("rb"))

# %% [markdown]
# ### Normalization

# %%
adata.raw = adata
adata.layers["counts"] = adata.raw.X
sc.pp.normalize_total(adata, target_sum=1e4)
adata.layers["normalized"] = adata.X
sc.pp.log1p(adata)
adata.layers["log"] = adata.X
sc.pp.scale(adata, max_value=10)
adata.layers["scale"] = adata.X

# %%
# Compute new total counts and detected genes after normalization
adata.obs["total_counts_norm"] = adata.X.sum(1)
adata.obs["n_genes_by_counts_norm"] = (adata.X > 0).sum(1)

# Scatter plot to compare before and after normalization
sc.pl.scatter(adata, 'total_counts', 'n_genes_by_counts')
sc.pl.scatter(adata, 'total_counts_norm', 'n_genes_by_counts_norm')

# %% [markdown]
# ### HVG

# %%
adata.X = adata.layers["log"]
sc.pp.highly_variable_genes(adata, min_mean=0.00125, max_mean=3, min_disp=0.5, n_top_genes= 2000, batch_key="sample")
sc.pl.highly_variable_genes(adata)
print('Number of HVGs: {:d}'.format(adata[:, adata.var["highly_variable"]].n_vars))

# %% [markdown]
# ### PCA

# %%
adata.X = adata.layers["scale"]
sc.pp.pca(adata, n_comps= 50, svd_solver='auto') # By default uses only highly variable genes if available
sc.pl.pca(adata)
sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50)

sc.pl.pca(
    adata,
    color=["sample", "sample", "batch", "batch", "pct_counts_mt", "pct_counts_mt"],
    dimensions=[(0, 1), (2, 3), (0, 1), (2, 3), (0, 1), (2, 3)],
    ncols=2,
    size=2,
)

# %% [markdown]
# ### Harmony

import scanpy.external as sce
sce.pp.harmony_integrate(adata, key='sample', theta=1)

adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']

sc.pl.pca(
    adata,
    color=["sample", "sample", "batch", "batch", "pct_counts_mt", "pct_counts_mt"],
    dimensions=[(0, 1), (2, 3), (0, 1), (2, 3), (0, 1), (2, 3)],
    ncols=2,
    size=2,
)

# %% [markdown]
# ### UMAP

# %%
sc.pp.neighbors(adata, 
                n_pcs=30, 
                use_rep="X_pca", 
                knn=True, 
                random_state=42, 
                method="umap", 
                metric="euclidean")
sc.tl.umap(adata)
sc.pl.umap(
    adata,
    color="sample",
    size=2,
)

# %%
magic_operator = magic.MAGIC(knn=30, solver="approximate")
X_smoothened = magic_operator.fit_transform(adata.X)
X_smoothened[X_smoothened < 0] = 0
adata.layers["magic"] = X_smoothened

# %%
pickle.dump(adata, (folder_data_preproc / "adata_preproc.pkl").open("wb"))


# %% [markdown]
# ## Interpret and subset
# %%
adata = pickle.load((folder_data_preproc / "adata_preproc.pkl").open("rb"))

# %%
sc.tl.leiden(adata,resolution=2.1, flavor="igraph", n_iterations=-1)
sc.pl.umap(
    adata, 
    color=['leiden'],
    legend_loc="on data",
    legend_fontsize = 6,
    show=False
)

# %% [markdown]
# ## Re-assesing quality control and cell filtering

# %%
sc.pl.umap(
    adata,
    color=["leiden", "doublet_score", "sample", "log1p_total_counts", "pct_counts_mt", "log1p_n_genes_by_counts"],
    # increase horizontal space between panels
    wspace=0.5,
    size=3,
    ncols=2,
)

# %% [markdown]
# ## Annotation

# %%
sc.tl.rank_genes_groups(
    adata, "leiden", method="wilcoxon", key_added="wilcoxon", use_raw=False
)

# %%
# Extract the dictionary of gene names from recarray
top_genes = {cluster: adata.uns["wilcoxon"]["names"][cluster] for cluster in adata.uns["wilcoxon"]["names"].dtype.names}

# Convert to DataFrame
num_top_genes = 20  # Adjust the number of genes to retrieve
top_genes_df = pd.DataFrame({f"Cluster {cluster}": top_genes[cluster][:num_top_genes] for cluster in top_genes})

# %%
print(top_genes_df)
# sc.pl.dotplot(adata, var_names=top_genes_df.iloc[:3].values.flatten(), groupby="leiden")

# %%
diffexp = (
    sc.get.rank_genes_groups_df(adata, group=None, key="wilcoxon")
    .sort_values("scores", ascending=False)
    .groupby("group")
    .head(10)
)
diffexp["symbol"] = diffexp["names"].apply(lambda x: adata.var.loc[x, "symbol"])
diffexp.set_index("group").loc["1"]

# %%
symbols_dict = {}
for i in range(27):
    symbols_dict[f"symbols{i}"] = diffexp[diffexp.group == str(i)]["symbol"].tolist()


# %%
import io

marker_annotation = pd.read_table(
    io.StringIO(
        """ix	symbols	celltype
0	CD34, SPINK2	Hematopoietic Stem Cell
0	AZU1, MPO	Common Myeloid Progenitor
0	CLIP2, BCL2, SFMBT2	Lymphoid Multipotent Progenitor

1	FAM178B, WNT5B	Early Erythroid
1	TFRC, PVT1	Mid Erythroid
1	HBA1, SLC4A1	Late Erythroid
1	MED12L, ITGA2B	Early Megakaryocyte
1	CALD1, ITGB3	Late Megakaryocyte

2	CD247, IL2RB	Natural Killer Cell

3	BACE2, HDC	Granulocyte

4	FCN1, PLAUR	Monocyte
4	CD163, HLA-DPB1, HMOX1	Macrophage

5	IRF8, CLEC4C, IL3RA	plasmacytoid Dendritic Cell
5	IL1R1, IL7R	ILC

6	PAX5, UHRF1	Pro-B Cell
6	PAX5, ARPP21	Pre-B Cell
6	PAX5, FCRL1	Immature B Cell

7	ASS1, APOB	Hepatocytes
7	STAB2, NRG3	LSECs
"""
    )
).set_index("celltype")

marker_annotation["symbols"] = marker_annotation["symbols"].astype(str).str.split(", ")

# %%
import chromatinhd.utils.scanpy

cluster_celltypes = chd.utils.scanpy.evaluate_partition(
    adata, marker_annotation["symbols"].to_dict(), "symbol", partition_key="leiden"
).idxmax()

adata.obs["celltype"] = adata.obs["celltype"] = cluster_celltypes[
    adata.obs["leiden"]
].values
adata.obs["celltype"] = adata.obs["celltype"] = adata.obs["celltype"].astype(str)
sc.pl.umap(adata, color=["leiden", "celltype"], wspace=0.5)

# %% [markdown]
# ### Refining the annotation

# %%
# Kupffer Cells
adata_KC = adata[adata.obs["leiden"].isin(["21"])]
sc.tl.leiden(adata_KC, resolution=0.4, flavor="igraph", n_iterations=-1)
sc.pl.umap(adata_KC, color=["leiden", "celltype", "sample"])

KC_0_cells = adata_KC.obs[adata_KC.obs['leiden'] == '0'].index
adata.obs['celltype'] = adata.obs['celltype'].cat.add_categories('Macrophage_III')
adata.obs.loc[KC_0_cells, 'celltype'] = 'Macrophage_III'

KC_1_cells = adata_KC.obs[adata_KC.obs['leiden'] == '1'].index
adata.obs['celltype'] = adata.obs['celltype'].cat.add_categories('Macrophage_II')
adata.obs.loc[KC_1_cells, 'celltype'] = 'Macrophage_II'

KC_2_cells = adata_KC.obs[adata_KC.obs['leiden'] == '2'].index
adata.obs['celltype'] = adata.obs['celltype'].cat.add_categories('Macrophage_I')
adata.obs.loc[KC_2_cells, 'celltype'] = 'Macrophage_I'

KC_3_cells = adata_KC.obs[adata_KC.obs['leiden'] == '3'].index
adata.obs.loc[KC_3_cells, 'celltype'] = 'Macrophage_I'


adata.obs['celltype'] = adata.obs['celltype'].cat.remove_categories('Macrophage')

# %%
# Monocytes
cluster_8_cells = adata.obs[adata.obs['leiden'] == '8'].index
adata.obs['celltype'] = adata.obs['celltype'].cat.add_categories('Monocyte_II')
adata.obs.loc[cluster_8_cells, 'celltype'] = 'Monocyte_II'

cluster_9_cells = adata.obs[adata.obs['leiden'] == '9'].index
adata.obs['celltype'] = adata.obs['celltype'].cat.add_categories('Monocyte_I')
adata.obs.loc[cluster_9_cells, 'celltype'] = 'Monocyte_I'

adata.obs['celltype'] = adata.obs['celltype'].cat.remove_categories('Monocyte')

# %%
# Low Expression cluster / No markers

cluster_1_cells = adata.obs[adata.obs['leiden'] == '1'].index
adata.obs['celltype'] = adata.obs['celltype'].cat.add_categories('Low Expression')
adata.obs.loc[cluster_1_cells, 'celltype'] = 'Low Expression'

# %%
# MEP
cluster_4_cells = adata.obs[adata.obs['leiden'] == '4'].index
cluster_11_cells = adata.obs[adata.obs['leiden'] == '11'].index
adata.obs['celltype'] = adata.obs['celltype'].cat.add_categories('Megakaryocyte Erythroid Progenitor')
adata.obs.loc[cluster_4_cells, 'celltype'] = 'Megakaryocyte Erythroid Progenitor'
adata.obs.loc[cluster_11_cells, 'celltype'] = 'Megakaryocyte Erythroid Progenitor'

# %%
# Weird cluster
cluster_6_cells = adata.obs[adata.obs['leiden'] == '6'].index
adata.obs['celltype'] = adata.obs['celltype'].cat.add_categories('EBI Macrophages')
adata.obs.loc[cluster_6_cells, 'celltype'] = 'EBI Macrophages'

# %%
adata.obs['celltype'] = adata.obs['celltype'].cat.reorder_categories(
    sorted(adata.obs['celltype'].cat.categories), ordered=True
)

grouped_palette = [
    # Erythroid (reds/pinks)
    '#67000d',  # HSC
    '#a50f15',  # CMP
    '#cb181d',  # MEP
    '#de2d26',  # Early Erythroid
    '#fb6a4a',  # Mid Erythroid
    '#fcae91',  # Late Erythroid

    # Megakaryocyte (purples)
    '#dd3497',  # Early Megakaryocyte
    '#f768a1',  # Late Megakaryocyte

    # Myeloid (browns/oranges)
    '#1a9850',  # Granulocyte
    '#cc4c02',  # Monocyte_I 
    '#993404',  # Monocyte_II 
    '#fe9929',  # Macrophage_I
    '#fdae6b',  # Macrophage_II 
    '#fdd49e',  # Macrophage_III 
    '#a6d854',  # Dendritic Cell
    '#4daf4a',  # EBI Macrophages

    # Lymphoid (blues)
    '#08519c',  # LMPP
    '#3182bd',  # Pro-B
    '#6baed6',  # Pre-B
    '#9ecae1',  # Immature B

    # NK / ILC (greens)
    '#1c9099',  # NK
    '#005b5b',  # ILC

    # Liver / Stromal (grays/yellows)
    '#969696',  # LSECs
    '#d9d9d9',  # Hepatocytes

    # Low Expression (neutral)
    '#bdbdbd'
]

desired_order = [
    'Hematopoietic Stem Cell', 'Common Myeloid Progenitor', 'Megakaryocyte Erythroid Progenitor',
    'Early Erythroid', 'Mid Erythroid', 'Late Erythroid',
    'Early Megakaryocyte', 'Late Megakaryocyte',
    'Granulocyte', 'Monocyte_I', 'Monocyte_II', 'Macrophage_I', 'Macrophage_II', 'Macrophage_III', 'plasmacytoid Dendritic Cell', 'EBI Macrophages',
    'Lymphoid Multipotent Progenitor', 'Pro-B Cell', 'Pre-B Cell', 'Immature B Cell',
    'Natural Killer Cell', 'ILC',
    'LSECs', 'Hepatocytes',
    'Low Expression'
]

adata.obs['celltype'] = adata.obs['celltype'].astype('category')
adata.obs['celltype'] = adata.obs['celltype'].cat.reorder_categories(desired_order)


sc.pl.umap(adata, color=["celltype"], palette=grouped_palette, frameon=False, size=2.5, legend_loc='right margin')

# %%
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Create a dictionary mapping cell types to colors
palette_dict = dict(zip(desired_order, grouped_palette))

# Create legend handles from the dictionary
handles = [Patch(color=color, label=label) for label, color in palette_dict.items()]

# Create a separate figure for the legend
fig, ax = plt.subplots(figsize=(max(1, len(handles)*0.3), 1))  # width adjusts with number of labels
ax.legend(handles=handles, loc='center', ncol=2, frameon=False)  # ncol can be adjusted
ax.axis('off')  # hide axes

plt.show()


# %%
pickle.dump(adata, (folder_data_preproc / "adata_annotated.pkl").open("wb"))

# %% [markdown]
# ### Plotting genes

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))

# %%
def gene_id(adata, symbol):
    return adata.var.reset_index().set_index("symbol").loc[symbol, "gene_ids"]


def symbol(adata, gene):
    return adata.var.reset_index().set_index("gene_ids").loc[gene, "symbol"]

# %%
sc.pl.umap(adata, color=adata.var.index[(adata.var["symbol"] == "CCR2")][0])

# %%
adata2 = adata[adata.obs["celltype"].isin(["Monocyte_I", "Monocyte_II", "Monocyte_III", "Macrophage_I", "Macrophage_II"])]

# %%
eyck.m.t.plot_umap(adata2, 
                   ["CXCR4", "CD74", "celltype"],
                   legend="under panel",
                   panel_size= 1.5,
                   ncol=1,
                   datashader=False).display()


# %%
# Dotplot of marker genes
marker_genes = [
        'CD34', 'SPINK2', 'MLLT3', 'RNF220', 'CALN1',
        'AZU1', 'MPO',
        'CLIP2', 'BCL2', 'SFMBT2',
        'FAM178B',  'WNT5B', 'PVT1',
        'TFRC', 'NCL',
        'HBA1', 'C17orf99','SLC4A1', 
        'MED12L', 'ITGA2B',
        'LTBP1', 'CALD1', 'ITGB3', 
        'PAX5', 'UHRF1', 'RRM2', 
        'ARPP21', 
        'FCRL1',
        'CD247', 'TOX2', 'NCAM1', 'IL2RB', 'NCR1',
        'BACE2', 'KIT', 'HDC',
        'FCN1', 'RETN', 'PLAUR', 'MYO1F', 
        'CD163', 'CTSB', 'HLA-DPB1', 'HMOX1', 
        'IRF8', 'CLEC4C', 'PTPRS', 'IL3RA', 
        'IL1R1', 'IL7R',
        'CPS1', 'GPC3', 'ASS1', 'APOB', 
        'LDB2', 'STAB2', 'NRG3'
]

celltypes = ["HSCs", "CMP/GMP", "LMPP", "MEP",
            "Early_Ery", "Mid_Ery", "Late_Ery", "Early_MK", "Late_MK", 
            "Doublet_Ery_B",
            "Pre-Pro-B", "Pro-B", "Immature B", 
            "NK cells", "ILC",
            "Granulocyte", "Monocyte_I", "Monocyte_II", "Monocyte_III", "Macrophage_I", "Macrophage_II", "pDCS", 
            "Hepatocytes", "LSECs",
            "Low_Expr" ]

sc.pl.dotplot(
            adata, 
            var_names=marker_genes, 
            gene_symbols="symbol", 
            categories_order=celltypes, 
            groupby="celltype"
            )


# %%
# Dotplot of most DEG between clusters

sc.tl.rank_genes_groups(
    adata, "celltype", method="wilcoxon", key_added="wilcoxon2", use_raw=False
)

# Extract the dictionary of gene names from recarray
top_genes = {cluster: adata.uns["wilcoxon2"]["names"][cluster] for cluster in adata.uns["wilcoxon2"]["names"].dtype.names}

#%%
# Convert to DataFrame
num_top_genes = 3  # Adjust the number of genes to retrieve
top_genes_df = pd.DataFrame({f"Cluster {cluster}": top_genes[cluster][:num_top_genes] for cluster in top_genes})

# %%
sc.pl.dotplot(
            adata, 
            var_names=symbol(adata, np.unique(top_genes_df.iloc[:3].values.flatten())), 
            gene_symbols="symbol",
            categories_order=celltypes, 
            groupby="celltype"
            )

# %% [markdown]
# ### Subset for macrophages

# %%
adata2 = adata[adata.obs["celltype"].isin(["Macrophage_II", "Macrophage_I", "EBI Macrophages"])]
sc.tl.rank_genes_groups(
    adata2, groupby="celltype", method="wilcoxon", key_added="wilcoxon3", use_raw=False,
)

# %%
# Extract the dictionary of gene names from recarray
top_genes = {cluster: adata2.uns["wilcoxon3"]["names"][cluster] for cluster in adata2.uns["wilcoxon3"]["names"].dtype.names}

# Convert to DataFrame
num_top_genes = 10  # Adjust the number of genes to retrieve
top_genes_df = pd.DataFrame({f"Cluster {cluster}": top_genes[cluster][:num_top_genes] for cluster in top_genes})

# %%
top_genes = symbol(adata2, top_genes_df.T.values.flatten())

sc.pl.dotplot(adata2, 
              var_names=top_genes, 
              gene_symbols="symbol",
              groupby="celltype",
              layer="magic",
              cmap="rocket",
              swap_axes=True,)

sc.pl.heatmap(adata2, 
              var_names=top_genes, 
              gene_symbols="symbol",
              groupby="celltype",
              layer="magic",
              cmap="rocket",
              swap_axes=True,)

# %%
# GOI = [ 
#         "ITGAM", "CD14", "CD68", "CD163", "FCGR2A", "ADGRE1", "MERTK", # macrophage markers
#         "CD5L", "SLC1A3", "CD163", "FOLR2", "TIMD4", "MARCO", "GFRA2", "ADRB1", "TMEM26", "SLC40A1", "HMOX1", "SLC16A9", "VCAM1", "SUCNR1", # conserved KC markers
#         "SPI1",  "ITGAM","MSR1", "SIGLEC1", "ADGRE1",  "CCR2", "ID3", "ID1", # more KC markers
#         "ANK1", "WNT5B","GATA2", "EPOR", "GATA1", "HBA2", "HBA1", "EMP2", "TIMD4",  # possible EBI macro markers
#         "SAT1", "SAMHD1",  "CLEC7A", "CD83",  "GFRA2", "MYOF", "CD86", "CD163", "MRC1", "HMOX1", "CLEC4F", "FCGR2A",  
#         "CX3CR1", "TLR4", "CD14", "CD68", "CCR2", "FCGR3A", "HLA-DRA", "ITGAX", "CLEC5A"
#        ]

GOI = [ 
        "EMP2", "EPOR", "ILF2", "SIGLEC1", "VCAM1"
]
sc.pl.heatmap(adata2, 
              var_names=GOI, 
              gene_symbols="symbol",
              groupby="celltype",
              swap_axes=True,
              layer="magic",
              cmap="rocket"
              )

sc.pl.dotplot(adata2, 
              var_names=GOI, 
              gene_symbols="symbol",
              groupby="celltype",
              swap_axes=True,
              layer="magic",
              cmap="rocket"
              )