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
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "liverfoetalH"
genome = "hg38"
organism = "hs"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

folder_dataset = chd.get_output() / "datasets" / dataset_name

# %% [markdown]
# ## TSS

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))

# %%
transcripts = pickle.load((folder_data_preproc / "transcripts.pkl").open("rb"))
transcripts = transcripts.loc[transcripts["ensembl_gene_id"].isin(adata.var.index)]

# %%
fragments_file = folder_data_preproc / "atac_fragments_H7A.tsv.gz"
selected_transcripts = chd.data.regions.select_tss_from_fragments(
    transcripts, fragments_file
)

# %%
np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max()).shape

# %%
plt.scatter(
    adata.var["means"],
    np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max())[
        adata.var.index
    ],
)
np.corrcoef(
    adata.var["means"],
    np.log(transcripts.groupby("ensembl_gene_id")["n_fragments"].max() + 1)[
        adata.var.index
    ],
)

# %%
pickle.dump(
    selected_transcripts, (folder_data_preproc / "selected_transcripts.pkl").open("wb")
)

# %% [markdown]
# ## Preprocess

# %%
dataset_folder = chd.get_output() / "datasets" / dataset_name
dataset_folder.mkdir(exist_ok=True, parents=True)

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))
adata.obs["cell_original"] = adata.obs.index
adata.obs.index = adata.obs.index.str.split("-").str[0] + "-1"
adata.obs['batches'] = adata.obs['batch'].cat.codes.astype(int)

# %% [markdown]
# ### Create transcriptome for top 5000 genes + genes of interest

# %%
genes_of_interest = [
    'CD34', 'SPINK2', 'MLLT3', 'RNF220', 'AZU1', 'MPO', 'SFMBT2', 'FAM178B', 'WNT5B', 'TFRC', 'HBA1',
    'C17orf99', 'SLC4A1', 'MED12L', 'LTBP1', 'ITGA2B', 'ITGB3', 'CD247', 'TOX2', 'NCAM1', 'NCR1', 'BACE2',
    'KIT', 'HDC', 'RETN', 'PLAUR', 'MYO1F', 'CD163', 'CTSB', 'HLA-DPB1', 'HMOX1', 'IRF8', 'CLEC4C', 'PTPRS',
    'IL3RA', 'IL1R1', 'UHRF1', 'RRM2', 'IL7R', 'PAX5', 'ARPP21', 'FCRL1', 'CPS1', 'GPC3', 'ASS1', 'APOB',
    'LDB2', 'STAB2', 'NRG3', 'NCL', 'CALN1', 'IL2RB', 'TIMD4', 'SLC16A9', 'NDST3', 'ITGAD', 'VCAM1', 'SELENBP1',
    'BCAM', 'CDH5', 'CETP', 'RND3', 'FEZ1', 'FABP3', 'SDC3', 'CXCL12', 'ITLN1', 'LYVE1',
    'SLC1A3', 'CD163', 'FOLR2', 'MARCO', 'GFRA2', 'ADRB1', 'TMEM26', 'SLC40A1', 'HMOX1', 'SLC16A9', 'VCAM1',
    'SUCNR1', 'ITGAL', 'TLR4', 'CDH1', 'CDH5', 'RAB32', 'RNASE1', 'ABCG2', 'RPS2', 'RPS17', 'CORO1A', 'RPS20',
    'STAB1', 'CREM', 'RPS4Y1', 'SRGN', 'C5AR1', 'VMO1', 'PPP1CB', 'FOSL2', 'SH3BP5', 'ELL2', 'TIMP1', 'HLA-DRB1',
    'CD74', 'HLA-DRA', 'HLA-DPA1', 'CST3', 'FGL2', 'HLA-DMA', 'CTSS', 'HLA-F', 'CXCL12', 'FLT3', 'IL7R', 'EBF1',
    'PAX5', 'RAG1', 'RAG2', 'BACH2', 'SPIB', 'FCRL1', 'FLI1', 'GATA1', 'GATA2', 'TFRC', 'GYPA', 'SPI1', 'MAFB',
    'KLF4', 'RUNX1', 'CSF1R', 'CD14', 'CD68', 'CD163', 'CD200R1', 'CD86', 'CD83', 'FCGR2A', 'ITGAM', 'CLEC4F',
    'ID3', 'ARL4C', 'MERTK', 'NR1H3', 'SIGLEC1', 'TIMD4', 'VCAM1', 'IL1B', 'CASP1', 'ADGRE1', 'PTPRC', 'FCGR1A',
    'TNF', 'HSP90AA1', 'C1QA', 'HSP90B1', 'HSP90AB1', 'TLR4', 'RUNX1', 'SPI1', 'CSF1R', 'MAFB', 'KLF2', 'KLF4',
    'TIMD4', 'CLEC4F', 'SPIC', 'NR1H3', 'AIF1', 'MERTK', 'ADGRE1', 'CLEC7A', 'CCR2', 'CX3CR1', 'MSR1', 'CD63',
    'MRC1', 'CD5L', 'ITGAM', 'CD14', 'CD68', 'CD163', 'FCGR2A', 'ADGRE1', 'MERTK', 'SPI1', 'SLC1A3', 'FOLR2',
    'TIMD4', 'MARCO', 'GFRA2', 'ADRB1', 'TMEM26', 'SLC40A1', 'HMOX1', 'SLC16A9', 'VCAM1', 'SUCNR1', 'SPI1',
    'MSR1', 'SIGLEC1', 'CCR2', 'ID3', 'ID1', 'GATA2', 'EPOR', 'GATA1', 'HBA2', 'HBA1', 'EMP2', 'TIMD4', 'SAT1',
    'SAMHD1', 'CLEC7A', 'CD83', 'GFRA2', 'MYOF', 'CD86', 'CD163', 'MRC1', 'HMOX1', 'CLEC4F', 'FCGR2A', 'CX3CR1',
    'TLR4', 'CD14', 'CD68', 'CCR2', 'FCGR3A', 'HLA-DRA', 'ITGAX', 'CLEC5A', 
    'TIMD4', 'MARCO', 'MSR1', 'CD68', 'CALR', 'CD36', 'AXL', 'MERTK', 'TYRO3', 'CD163', 'MRC1', 'CLEC7A', 'FCGR1A', 'APOL2', 'SCARA5'
]

def gene_id(adata, symbol):
    return adata.var.reset_index().set_index("symbol").loc[symbol, "gene_ids"]
genes_of_interest_indices = [gene_id(adata, gene) for gene in genes_of_interest]
top_5000_genes = adata.var.sort_values("dispersions_norm").tail(5000).index
combined_genes = pd.Index((set(genes_of_interest_indices)).union(set(top_5000_genes)))


# %%
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(
    adata[:, combined_genes],
    path=dataset_folder / "transcriptome",
    overwrite=True
)

# %%
sc.pl.umap(adata, color=["leiden", "celltype"])

# %% [markdown]
# ### 10k

# %%
selected_transcripts = pickle.load(
    (folder_data_preproc / "selected_transcripts.pkl").open("rb")
).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-10000, 10000], dataset_folder / "regions" / "10k10k"
)

# %%
samples = ["H5A", "H5B", "H6A", "H7A", "H7B", "H7C"]
fragments_files = []
for sample in samples:
    fragments_file = folder_data_preproc / f"atac_fragments_{sample}.tsv.gz"
    fragments_files.append(fragments_file)
fragments = chd.data.Fragments(dataset_folder / "fragments" / "10k10k")


# %%
fragments.regions = regions
fragments = chd.data.Fragments.from_multiple_fragments_tsv(
    fragments_files=fragments_files,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    overwrite=True,
    batch_column= "batches"
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ### 100k

# %%
selected_transcripts = pickle.load(
    (folder_data_preproc / "selected_transcripts.pkl").open("rb")
).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-100000, 100000], dataset_folder / "regions" / "100k100k"
)

# %%
samples = ["H5A", "H5B", "H6A", "H7A", "H7B", "H7C"]
fragments_files = []
for sample in samples:
    fragments_file = folder_data_preproc / f"atac_fragments_{sample}.tsv.gz"
    fragments_files.append(fragments_file)
fragments = chd.data.Fragments(dataset_folder / "fragments" / "100k100k")


# %%
fragments.regions = regions
fragments = chd.data.Fragments.from_multiple_fragments_tsv(
    fragments_files=fragments_files,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    overwrite=True,
    batch_column= "batches"
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ## Clustering

# %%
transcriptome = eyck.modalities.Transcriptome(folder_dataset / "transcriptome")

# %%
clustering = chd.data.Clustering.from_labels(
    transcriptome.obs["celltype"], path=folder_dataset / "clusterings" / "cluster",
    overwrite=True
)

# %% [markdown]
# ## Folds

# %%
folds = chd.data.folds.Folds(folder_dataset / "folds")
folds.folds = [
    {
        "cells_train": np.arange(len(transcriptome.obs)),
        "cells_test": np.arange(len(transcriptome.obs))[:500],
        "cells_validation": np.arange(len(transcriptome.obs))[:500],
    }
]

# %% [markdown]
# ### Create transcriptome for all genes

# %%
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(
    adata,
    path=dataset_folder / "transcriptome_all",
    overwrite=True
)

# %%
sc.pl.umap(adata, color=adata.var.index[(adata.var["symbol"] == "GLUL")][0])
sc.pl.umap(adata, color=["leiden", "celltype"])

# %% [markdown]
# ### 10k

# %%
selected_transcripts = pickle.load(
    (folder_data_preproc / "selected_transcripts.pkl").open("rb")
).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-10000, 10000], dataset_folder / "regions_all" / "10k10k"
)

# %%
samples = ["H5A", "H5B", "H6A", "H7A", "H7B", "H7C"]
fragments_files = []
for sample in samples:
    fragments_file = folder_data_preproc / f"atac_fragments_{sample}.tsv.gz"
    fragments_files.append(fragments_file)
fragments = chd.data.Fragments(dataset_folder / "fragments_all" / "10k10k")


# %%
fragments.regions = regions
fragments = chd.data.Fragments.from_multiple_fragments_tsv(
    fragments_files=fragments_files,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    overwrite=True,
    batch_column= "batches"
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ### 100k

# %%
selected_transcripts = pickle.load(
    (folder_data_preproc / "selected_transcripts.pkl").open("rb")
).loc[transcriptome.var.index]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-100000, 100000], dataset_folder / "regions_all" / "100k100k"
)

# %%
samples = ["H5A", "H5B", "H6A", "H7A", "H7B", "H7C"]
fragments_files = []
for sample in samples:
    fragments_file = folder_data_preproc / f"atac_fragments_{sample}.tsv.gz"
    fragments_files.append(fragments_file)
fragments = chd.data.Fragments(dataset_folder / "fragments_all" / "100k100k")


# %%
fragments.regions = regions
fragments = chd.data.Fragments.from_multiple_fragments_tsv(
    fragments_files=fragments_files,
    regions=regions,
    obs=transcriptome.obs,
    path=fragments.path,
    reuse=True,
    batch_column= "batches",
    batch_size=100000000
)

# %%
fragments.create_regionxcell_indptr()

# %% [markdown]
# ## Clustering

# %%
transcriptome = eyck.modalities.Transcriptome(folder_dataset / "transcriptome_all")

# %%
clustering = chd.data.Clustering.from_labels(
    transcriptome.obs["celltype"], path=folder_dataset / "clusterings_all" / "cluster",
    overwrite=True
)

# %% [markdown]
# ## Folds

# %%
folds = chd.data.folds.Folds(folder_dataset / "folds_all")
folds.folds = [
    {
        "cells_train": np.arange(len(transcriptome.obs)),
        "cells_test": np.arange(len(transcriptome.obs))[:500],
        "cells_validation": np.arange(len(transcriptome.obs))[:500],
    }
]
# %%
