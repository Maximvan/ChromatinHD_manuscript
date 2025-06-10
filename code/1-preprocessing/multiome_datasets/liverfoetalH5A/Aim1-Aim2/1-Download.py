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
    "#1E64C8", "#215BB4", "#2352A0", "#26498C", "#284078",
    "#2B3764", "#2D2E50", "#30253C", "#321C28", "#341314"
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

folder_dataset = chd.get_output() / "datasets" / dataset_name

# %% [markdown]
# ## Download snATAC-seq fragments

# %% 
# Individual H5, sample A, B
print(
    f"wget https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_5_A-atac_fragments.tsv.gz -O {folder_data_preproc}/atac_fragments_H5A.tsv.gz.raw"
)
print(
    f"wget https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_5_B-atac_fragments.tsv.gz -O {folder_data_preproc}/atac_fragments_H5B.tsv.gz.raw"
)
# Individual H6, sample A
print(
    f"wget https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_6_A-atac_fragments.tsv.gz -O {folder_data_preproc}/atac_fragments_H6A.tsv.gz.raw"
)
# Individual H7, sample A, B, C
print(
    f"wget https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_7_A-atac_fragments.tsv.gz -O {folder_data_preproc}/atac_fragments_H7A.tsv.gz.raw"
)
print(
    f"wget https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_7_B-atac_fragments.tsv.gz -O {folder_data_preproc}/atac_fragments_H7B.tsv.gz.raw"
)
print(
    f"wget https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_7_C-atac_fragments.tsv.gz -O {folder_data_preproc}/atac_fragments_H7C.tsv.gz.raw"
)


# %% [markdown]
# The original data is not block compressed, so we need to recompress the data using tabix.

# %%
print(
    f"zcat {folder_data_preproc}/atac_fragments_H5A.tsv.gz.raw > {folder_data_preproc}/atac_fragments_H5A.tsv"
)
print(
    f"zcat {folder_data_preproc}/atac_fragments_H5B.tsv.gz.raw > {folder_data_preproc}/atac_fragments_H5B.tsv"
)
print(
    f"zcat {folder_data_preproc}/atac_fragments_H6A.tsv.gz.raw > {folder_data_preproc}/atac_fragments_H6A.tsv"
)
print(
    f"zcat {folder_data_preproc}/atac_fragments_H7A.tsv.gz.raw > {folder_data_preproc}/atac_fragments_H7A.tsv"
)
print(
    f"zcat {folder_data_preproc}/atac_fragments_H7B.tsv.gz.raw > {folder_data_preproc}/atac_fragments_H7B.tsv"
)
print(
    f"zcat {folder_data_preproc}/atac_fragments_H7C.tsv.gz.raw > {folder_data_preproc}/atac_fragments_H7C.tsv"
)

# %%
import pysam 
samples = ["H5A", "H5B", "H6A", "H7A", "H7B", "H7C"]
for sample in samples:
    gz_raw = folder_data_preproc / f"atac_fragments_{sample}.tsv.gz.raw"
    tsv_file = folder_data_preproc / f"atac_fragments_{sample}.tsv"
    gz_file = folder_data_preproc / f"atac_fragments_{sample}.tsv.gz"
    if not gz_file.exists():
        pysam.tabix_compress(str(tsv_file), str(gz_file), force=True)
        pysam.tabix_index(str(gz_file), seq_col=0, start_col=1, end_col=2)
        print(f"{sample} file compression done")
    else:
        print(f"{gz_file} already exists, skipping compression.")

    # Cleanup
    if tsv_file.exists():
        tsv_file.unlink()
    if gz_raw.exists():
        gz_raw.unlink()

# %%
# List of sample names
samples = ["H5A", "H5B", "H6A", "H7A", "H7B", "H7C"]
fragments_files = [folder_data_preproc / f"atac_fragments_{sample}.tsv.gz" for sample in samples]

combined_data = pd.DataFrame()
for sample, fragments_file in zip(samples, fragments_files):
    fragments_data = pd.read_csv(fragments_file, sep="\t", compression="gzip", header=None, names=['chrom', 'start', 'end', 'sequence_id', 'count'])
    fragments_data["fragment_length"] = fragments_data["end"] - fragments_data["start"]
    fragments_data["batch"] = sample
    combined_data = pd.concat([combined_data, fragments_data], ignore_index=True)
print(combined_data.head())

# %%
# Check for duplicates in 'sequence_id'
duplicates = combined_data[combined_data.duplicated('sequence_id', keep=False)]
if duplicates.empty:
    print("No duplicate sequence IDs found across samples.")
else:
    print("Duplicate sequence IDs found across samples:")
    print(duplicates)


# %%
print(f"Saving raw fragments data with {combined_data["sequence_id"].nunique()} cells and {combined_data.shape[0]} fragments")
pickle.dump(combined_data, (folder_data_preproc / "atac_pandas.pkl").open("wb"))

# Load data into a dictionary of DataFrames
columns = ['chrom', 'start', 'end', 'sequence_id', 'count']
fragments_data = {sample: pd.read_csv(fragments_file, sep="\t", compression="gzip", header=None, names=columns) for sample, fragments_file in zip(samples, fragments_files)}
for sample in samples:
    fragments_data[sample]["fragment_length"] = fragments_data[sample]["end"] - fragments_data[sample]["start"]
print(fragments_data["H5A"].head())

# %%
pickle.dump(fragments_data, (folder_data_preproc / "atac_pandas.pkl").open("wb"))


# %% [markdown]
# ## Downloading snRNA-seq
# %% 
# Individual H5, sample A, B
print(
    f"wget https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_5_A-filtered_feature_bc_matrix.h5 -O {folder_data_preproc}/rna_counts_H5A.h5"
)
print(
    f"wget https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_5_B-filtered_feature_bc_matrix.h5 -O {folder_data_preproc}/rna_counts_H5B.h5"
)
# Individual H6, sample A
print(
    f"wget https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_6_A-filtered_feature_bc_matrix.h5 -O {folder_data_preproc}/rna_counts_H6A.h5"
)
# Individual H7, sample A, B, C
print(
    f"wget https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_7_A-filtered_feature_bc_matrix.h5 -O {folder_data_preproc}/rna_counts_H7A.h5"
)
print(
    f"wget https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_7_B-filtered_feature_bc_matrix.h5 -O {folder_data_preproc}/rna_counts_H7B.h5"
)
print(
    f"wget https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_7_C-filtered_feature_bc_matrix.h5 -O {folder_data_preproc}/rna_counts_H7C.h5"
)

# %%
adata_H5A = sc.read_10x_h5(
    folder_data_preproc / "rna_counts_H5A.h5",
    backup_url="https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_5_A-filtered_feature_bc_matrix.h5"
)
adata_H5A.var_names_make_unique()
adata_H5B = sc.read_10x_h5(
    folder_data_preproc / "rna_counts_H5B.h5",
    backup_url="https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_5_B-filtered_feature_bc_matrix.h5"
)
adata_H5B.var_names_make_unique()
adata_H6A = sc.read_10x_h5(
    folder_data_preproc / "rna_counts_H6A.h5",
    backup_url="https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_6_A-filtered_feature_bc_matrix.h5"
)
adata_H6A.var_names_make_unique()
adata_H7A = sc.read_10x_h5(
    folder_data_preproc / "rna_counts_H7A.h5",
    backup_url="https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_7_A-filtered_feature_bc_matrix.h5"
)
adata_H7A.var_names_make_unique()
adata_H7B = sc.read_10x_h5(
    folder_data_preproc / "rna_counts_H7B.h5",
    backup_url="https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_7_B-filtered_feature_bc_matrix.h5"
)
adata_H7B.var_names_make_unique()
adata_H7C = sc.read_10x_h5(
    folder_data_preproc / "rna_counts_H7C.h5",
    backup_url="https://www.ebi.ac.uk/biostudies/files/E-MTAB-13070/H_7_C-filtered_feature_bc_matrix.h5"
)
adata_H7C.var_names_make_unique()

# %%
adata = sc.concat(
    [adata_H5A, adata_H5B, adata_H6A, adata_H7A, adata_H7B, adata_H7C],
    join="outer",
    merge="same"
)
adata.obs_names_make_unique()
adata.obs['batch'] = ['H5A'] * adata_H5A.n_obs + ['H5B'] * adata_H5B.n_obs + ['H6A'] * adata_H6A.n_obs + ['H7A'] * adata_H7A.n_obs + ['H7B'] * adata_H7B.n_obs + ['H7C'] * adata_H7C.n_obs
adata.obs['sample'] = ['H5'] * adata_H5A.n_obs + ['H5'] * adata_H5B.n_obs + ['H6'] * adata_H6A.n_obs + ['H7'] * adata_H7A.n_obs + ['H7'] * adata_H7B.n_obs + ['H7'] * adata_H7C.n_obs

# Store the current index (gene symbols) into a new column
adata.var["symbol"] = adata.var.index  
# Set Ensembl Gene IDs as the new index
adata.var = adata.var.set_index("gene_ids")  

print(adata.obs["batch"].value_counts())
print(adata.obs["sample"].value_counts())

# %%
print(f"Saving raw adata with {len(adata.obs)} cells and {len(adata.var)} genes")
pickle.dump(adata, (folder_data_preproc / "adata_raw.pkl").open("wb"))

# %% [markdown]
# ## Create Transcriptome
# %%
transcriptome = eyck.modalities.Transcriptome(folder_dataset / "transcriptome")

# %% 
pickle_path = folder_data_preproc / "transcripts.pkl"

if pickle_path.exists():
    print("Loading transcripts from pickle...")
    with open(pickle_path, "rb") as f:
        transcripts = pickle.load(f)
else:
    print("Pickle file not found. Generating transcripts and saving to pickle...")
    print("Gene IDs to query:", adata.var.index.unique())  
    
    # Retrieve transcripts
    transcripts = chd.biomart.get_transcripts(
        chd.biomart.Dataset.from_genome(genome),
        gene_ids=adata.var.index.unique(),
        filter_protein_coding=False
    )
    
    # Debug: Check if transcripts are being retrieved
    if transcripts.empty:
        print("No transcripts retrieved.")
    else:
        print("Transcripts retrieved successfully, saving to pickle.")
        
    # Save the transcripts only if they are not empty
    if not transcripts.empty:
        with open(pickle_path, "wb") as f:
            pickle.dump(transcripts, f)

# Print the number of transcripts retrieved
print(f"Number of transcripts retrieved: {len(transcripts)}")
