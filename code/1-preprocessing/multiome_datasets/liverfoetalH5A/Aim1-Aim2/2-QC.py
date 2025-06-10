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
from matplotlib import rcParams
import seaborn as sns

sns.set_style("ticks")

import pickle

import scanpy as sc

import tqdm.auto as tqdm
import io

import chromatinhd as chd

import magic
import eyck
import scipy.sparse as sp
import anndata as ad

# %%
sc.settings.verbosity = 0 # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=150, facecolor="white", figsize=[4,4])
ugent_palette = [
    "#1e62c7ff", "#3373ccff", "#4a83d1ff", "#78a1deff", "#6191d7ff",
    "#8eb1e3ff", "#a4c1e8ff", "#bbcfeeff", "#d2def3ff", "#e7eff9ff"
]
accent_colour = ["2D8CA8"]
sns.set_palette(ugent_palette)

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "liverfoetalH"
genome = "hg38"
organism = "hs"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

folder_dataset = chd.get_output() / "dataset" / dataset_name

# %% [markdown]
# ## Load in data

# %%
adata = pickle.load((folder_data_preproc / "adata_raw.pkl").open("rb"))

# %%
fragments_data = pickle.load((folder_data_preproc / "atac_pandas.pkl").open("rb"))


# %% [markdown]
# ## Quality Control of snRNA-seq

# %% [markdown]
# ### Mitochondrial, ribosomal and hemglobin genes

# %%
# mitochondrial genes, "MT-" for human
adata.var["mt"] = adata.var["symbol"].str.startswith(("MT-", "MTRNR"))
# ribosomal genes
adata.var["ribo"] = adata.var["symbol"].str.startswith(("RPS", "RPL"))
# hemoglobin genes
adata.var["hb"] = adata.var["symbol"].str.contains("^HB[^(P)]")

sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt", "ribo", "hb"], inplace=True
)


n_samples = len(adata.obs['sample'].unique())
fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))

for idx, sample in enumerate(adata.obs['sample'].unique()):
    adata_sample = adata[adata.obs['sample'] == sample]
    
    # Plot for ribosomal genes
    sc.pl.violin(
        adata_sample,
        ["pct_counts_ribo"],
        jitter=0.4,
        multi_panel=True,
        groupby="batch",
        ax=axes[idx, 0], 
        show=False,
        ylabel="Percentage of ribosomal genes"
    )
    
    # Plot for hemoglobin genes
    sc.pl.violin(
        adata_sample,
        ["pct_counts_hb"],
        jitter=0.4,
        multi_panel=True,
        groupby="batch",
        ax=axes[idx, 1],
        show=False,
        ylabel="Percentage of hemoglobin genes"
    )
    
    # Plot for mitochondrial genes
    sc.pl.violin(
        adata_sample,
        ["pct_counts_mt"],
        jitter=0.4,
        multi_panel=True,
        groupby="batch",
        ax=axes[idx, 2],
        show=False,
        ylabel="Percentage of mitochondrial genes"
    )

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Total gene counts per cell, total counts in cell, highest expressed genes

# %%
fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
for idx, sample in enumerate(adata.obs['sample'].unique()):
    adata_sample = adata[adata.obs['sample'] == sample]
    
    # Plot for total counts in cell
    sns.histplot(
        adata_sample.obs['total_counts'], 
        bins=50,
        kde=True,
        ax=axes[idx, 0]
        )

    # Plot for counts of genes in all cells
    sns.histplot(
        adata_sample.obs['n_genes_by_counts'], 
        bins=50, 
        kde=True, 
        ax=axes[idx, 1])
    
    # Plot for highest expressed genes
    sc.pl.highest_expr_genes(
        adata_sample,
        gene_symbols="symbol",
        n_top=10,
        ax=axes[idx, 2],
        show=False)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Doublet detection
cells_that_error_scrublet = ['GCGGATACAATTTAGC-1', 'GTTGCATAGGTGAAAT-1', 'TGAGCTTAGCTCAAAC-1-1', 'GTGAATCTCAGGGCCT-1', 'GGTTAATGTTCATTTG-1']
adata = adata[~adata.obs_names.isin(cells_that_error_scrublet)].copy()
sc.pp.scrublet(adata, batch_key='batch')

#%%
sc.pl.scrublet_score_distribution(adata)


# %%
pickle.dump(adata, (folder_data_preproc / "adata_QC.pkl").open("wb"))

# %% [markdown]
# ## Quality Control of snATAC-seq

# %%
fragments_data = pickle.load((folder_data_preproc / "atac_pandas.pkl").open("rb"))

# %%
adata = pickle.load((folder_data_preproc / "adata_raw.pkl").open("rb"))


# %%
samples = ["H5A", "H5B", "H6A", "H7A", "H7B", "H7C"]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, sample in enumerate(samples):
    # Sample 10% of the data for the current sample
    sampled_df = fragments_data[sample].sample(frac=0.1, random_state=42)
    sns.histplot(x=sampled_df['fragment_length'], ax=axes[idx])
    axes[idx].axvspan(147, 294, color='lightsteelblue', alpha=0.3, label='Single Nucleosome-Bound Fragments (147-294 bp)')
    axes[idx].axvspan(0, 147, color='lightslategray', alpha=0.3, label='Nucleosome-Free Fragments (<147 bp)')
    axes[idx].set_title(f"Fragment Length Distribution (Sampled 10% of Data) - {sample}", fontsize=12)
    axes[idx].set_xlim(0, 1000)  # Focus on the first 1000 bp for the x-axis
    axes[idx].set_xlabel("Fragment Length (bp)", fontsize=10)
    axes[idx].set_ylabel("Count", fontsize=10)

plt.tight_layout()
plt.show()

# %%
# Count the number of fragments in each sample
fragment_counts = {sample: len(df) for sample, df in fragments_data.items()}
plt.figure(figsize=(8, 6))
sns.barplot(x=list(fragment_counts.keys()), y=list(fragment_counts.values()))
plt.title("Number of Fragments per Sample")
plt.xlabel("Sample")
plt.ylabel("Fragment Count")
plt.show()

# %%
# Check for duplicates based on 'chrom', 'start', 'end' columns
duplicates = {sample: df.duplicated(subset=['chrom', 'start', 'end']).sum() for sample, df in fragments_data.items()}
plt.figure(figsize=(8, 6))
sns.barplot(x=list(duplicates.keys()), y=list(duplicates.values()))
plt.title("Duplicate Fragments per Sample")
plt.xlabel("Sample")
plt.ylabel("Duplicate Count")
plt.show()

# %%