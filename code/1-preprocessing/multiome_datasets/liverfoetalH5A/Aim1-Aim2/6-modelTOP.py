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

chd.set_default_device("cpu")

pp.setup_ipython()

# %%
dataset_name = "liverfoetalH"
genome = "hg38"
organism = "hs"
folder_dataset = chd.get_output() / "datasets" / dataset_name
folder_plots = chd.get_output() / "datasets" / dataset_name / "plotsTOP100"
folder_plots.mkdir(exist_ok=True, parents=True)

# %%
folds = chd.data.folds.Folds(
    folder_dataset / "folds"
)
transcriptome = chd.data.Transcriptome(folder_dataset / "transcriptome")
clustering = chd.data.Clustering.from_labels(
    transcriptome.obs["celltype"], path=folder_dataset / "clusterings" / "cluster",
    overwrite=True
)

# %%
fragments = chd.data.Fragments(folder_dataset / "fragments" / "100k100k")
model_folder = (
    chd.get_output() / "diff" / "liverfoetalH" / "100k100k" / "cluster"
)

# %%
eyck.m.t.plot_umap(transcriptome, ["RUNX1", "celltype"], datashader = True).display()


# %% [markdown]
# Training

# # %%
# import chromatinhd.models.diff.model.binary

# models = chd.models.diff.model.binary.Models.create(
#     fragments = fragments,
#     clustering=clustering,
#     folds=folds,
#     model_params = dict(
#         encoder="shared",
#         encoder_params=dict(
#             delta_regularization=True,
#             delta_p_scale=1.0,
#             # bias_regularization=True,
#             # bias_p_scale=0.5,
#             binwidths=(5000, 1000, 500, 100, 50, 25),
#         ),
#     ),
#     train_params = dict(
#         early_stopping=False,
#         n_epochs = 20, # <----- originally I used 40
#     ),
#     path=model_folder,
#     reset = True,
# )

# # %%
# chd.set_default_device("cuda")
# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# models.train_models()
# chd.set_default_device("cpu")

# %%
models.models["0"].trace.plot()

# %%
train_loss = pd.DataFrame(models.models["0"].trace.train_steps)
train_loss_filtered = train_loss[train_loss["epoch"] >= 1]
train_loss_mean = train_loss_filtered.groupby('epoch')['loss'].mean()

# %%
plt.figure(figsize=(10, 6))
plt.plot(train_loss_mean.index, train_loss_mean.values, marker='o', linestyle='-', color='#1e62c7ff', label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# %%
validation_loss = pd.DataFrame(models.models["0"].trace.validation_steps)
validation_loss_filtered = validation_loss[validation_loss["epoch"] >= 1]
validation_loss_mean = validation_loss_filtered.groupby('epoch')['loss'].mean()

# %%
plt.figure(figsize=(10, 6))
plt.plot(validation_loss_mean.index, validation_loss_mean.values, marker='o', linestyle='-', color='#1e62c7ff', label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Inference

# %%
fragments = chd.data.Fragments(folder_dataset / "fragments" / "100k100k")
model_folder = (
    chd.get_output() / "diff" / "liverfoetalH" / "100k100k" / "cluster"
)

# %%
models = chd.models.diff.model.binary.Models(model_folder)

# %%
regionpositional = chd.models.diff.interpret.RegionPositional(
    model_folder / "scoring" / "regionpositional",
    reset=True,
)

# %%
regionpositional.score(
    models,
    fragments=fragments,
    clustering=clustering,
    device="cpu",
)
regionpositional

# %%
regionpositional.fragments = fragments
regionpositional.regions = fragments.regions
regionpositional.clustering = clustering


# %% [markdown]
# ## Run genes of interest

# %%
def analyze_gene(gene_marker, cells_of_interest, transcriptome, regionpositional, chd, fragments, clustering, dataset_name):
    gene_id = transcriptome.gene_id(gene_marker)
    windows = regionpositional.select_windows(
        gene_id,
        prob_cutoff=1.5,
        differential_prob_cutoff=4.,
        keep_tss=True,
        padding=1000,
    )
    breaking = chd.grid.Breaking(windows, resolution=4000, gap=0.03)

    motifscan_name = "hocomocov12_1e-4"
    genome_folder = pathlib.Path("/srv/data/genomes/GRCh38")
    motifscan_genome = chd.flow.Flow.from_path(genome_folder / "motifscans" / motifscan_name)
    motifscan = chd.data.motifscan.MotifscanView.from_motifscan(
        path=chd.get_output() / "datasets" / dataset_name / "motifscans" / "100k100k" / motifscan_name,
        parent=motifscan_genome,
        regions=fragments.regions,
    )

    motifs_oi = pd.concat([
        pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["RUNX1"])], "group": "RUNX1"}),
        pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["SPI1"])], "group": "SPI1"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["SUH"])], "group": "RBPJ"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["HEY1", "HES1"])], "group": "HEY1/HES1"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["NR1H3"])], "group": "NR1H3"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["RXRA"])], "group": "RXRA"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["PPARG"])], "group": "PPARG"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["SMAD4"])], "group": "SMAD4"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["RREB1"])], "group": "RREB1"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["TCF7L1", "TCF3", "TCF7L2"])], "group": "TCF7L1/TCF3"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["MEIS3"])], "group": "MEIS3"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["KLF7"])], "group": "KLF7"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["SPIC"])], "group": "SPIC"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["MEF2A"])], "group": "MEF2A"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["FOS"])], "group": "FOS"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["CTCF"])], "group": "CTCF"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["RELB", "NFKB"])], "group": "RELB/NFKB"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["LHX2"])], "group": "LHX2"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["FLI1"])], "group": "FLI1"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["GATA1"])], "group": "GATA1"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["GATA2"])], "group": "GATA2"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["MAF"])], "group": "MAF"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["MAFB"])], "group": "MAFB"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["KLF2"])], "group": "KLF2"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["KLF4"])], "group": "KLF4"}),
        
    ]).set_index("motif")

    motif_groups = pd.DataFrame({"group": motifs_oi["group"].unique()}).set_index("group")
    motif_groups["label"] = motif_groups.index
    motif_groups["color"] = [mpl.colors.to_hex(mpl.colormaps["tab10"](i)) for i in range(len(motif_groups))]

    cluster_info = clustering.cluster_info.loc[cells_of_interest]
    slices = regionpositional.calculate_slices(-0.5, step=25)
    differential_slices = regionpositional.calculate_differential_slices(
        slices, fc_cutoff=1.5, a="Monocyte_I", b="Macrophage_III"
    )

    fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.01, padding_width=0.05))
    region = fragments.regions.coordinates.loc[gene_id]

    panel_genes = chd.plot.genome.genes.GenesExpanding.from_region(
        region,
        breaking=breaking,
        genome="GRCh38",
        xticks=[-100000, 0, 100000],
        gene_overlap_padding=10000 if windows["length"].sum() > 20000 else 10000,
        show_others=False,
        only_canonical=True
    )
    fig.main.add_under(panel_genes, padding=0.)

    panel_differential = chd.models.diff.plot.DifferentialBroken.from_regionpositional(
        gene_id,
        regionpositional,
        cluster_info=cluster_info,
        breaking=breaking,
        ylintresh=5,
        ymax=20,
        label_accessibility=False,
        norm_atac_diff=mpl.colors.Normalize(np.log(1 / 8), np.log(8.0), clip=True),
        label_cluster="front",
        relative_to="Hematopoietic Stem Cell"
    )
    fig.main.add_under(panel_differential)

    # panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
    #     transcriptome,
    #     clustering,
    #     gene_id,
    #     cluster_info=cluster_info,
    #     show_n_cells=False,
    #     layer= "magic",
    # )
    # fig.main.add_right(panel_expression, panel_differential)

    panel_motifs = chd.data.motifscan.plot.GroupedMotifsBroken(
        motifscan,
        gene_id,
        motifs_oi=motifs_oi,
        breaking=breaking,
        group_info=motif_groups,
        show_triangle=False,
        slices_oi=differential_slices.get_slice_scores(
            clustering=clustering, regions=fragments.regions
        ),
    )
    fig.main.add_under(panel_motifs)

    panel_genes = chd.plot.genome.genes.GenesBroken.from_region(
        region,
        breaking=breaking,
        genome="GRCh38",
        show_others=False,
        only_canonical=True,
    )
    fig.main.add_under(panel_genes, padding=0.0, padding_up=0.2)


    fig_path = folder_plots / f"{gene_marker}_CHD.png"  # Specify the desired path and file name
    fig.savefig(str(fig_path), dpi=300, bbox_inches='tight')  # Save with high resolution and tight bounding box


# %% [markdown]
# ### Select gene list

# %%
def gene_id(adata, symbol):
    return adata.var.reset_index().set_index("symbol").loc[symbol, "gene_ids"]


def symbol(adata, gene):
    return adata.var.reset_index().set_index("gene_ids").loc[gene, "symbol"]

# %%
# Most dispersed genes
DSG = transcriptome.var["dispersions_norm"].nlargest(32).index.tolist()
gene_list = symbol(transcriptome, DSG).tolist()
gene_list.remove("AL589693.1")
gene_list.remove("STAB2")

# %%
# Gene markers used for annotation
gene_list = [
            'CD34', 'SPINK2', 'MLLT3', 'RNF220', 
            'AZU1', 'MPO', 'SFMBT2','FAM178B',  'WNT5B', 'TFRC', 'HBA1',
            'C17orf99','SLC4A1', 'MED12L', 'LTBP1', 
            'ITGA2B', 'ITGB3', 'CD247', 'TOX2', 'NCAM1',
            'NCR1', 'BACE2', 'KIT', 'HDC', 'RETN', 'PLAUR', 
            'MYO1F', 'CD163', 'CTSB', 'HLA-DPB1', 'HMOX1', 
            'IRF8', 'CLEC4C', 'PTPRS', 'IL3RA', 'IL1R1',
            'UHRF1', 'RRM2', 'IL7R', 'PAX5', 'ARPP21', 'FCRL1',
            'CPS1', 'GPC3', 'ASS1', 'APOB', 'LDB2', 'STAB2',
            'NRG3', 'NCL','CALN1','IL2RB'
            ]
# don't work with ensembl: 'CLIP2','PVT1', 

# %%
# Thesis: cell annotation validation
gene_list = [
            "CD34", "CD14", "GATA1", "CD19", "KLRB1", "CLEC4C", "ALB", "STAB2"
            ]

# %%
# Thesis: Blood cell maturation story
gene_list = [
            "FLI1", "GATA1", "GATA2"
]

# %%
# Thesis: Macrophage cell development story
gene_list = [
    "SPI1", "CSF1R"
    ]

# %%
# Thesis: KC developmental trajectory
gene_list = [
    "PROM1", "CD34", "FLT3", #HSC
    "BCL2", "MPO", "PRTN3", #CMP
    "FCN1", "VCAN", "S100A9", #monocyte
    "CSF1R", "MAF", "CLEC4F" #KC/macrophage
    ]

# %%
# Thesis: Functional Macrophage/Kupffer Cell Genes
gene_list = [
    "HLA-DQA1", "HLA-DQB1", "HLA-DRB1", "HLA-DPB1", "HLA-DRA", "HLA-DMB", "HLA-DPA1",  "HLA-DMA", "CD86", # antigen presentation and co-stimulation 
    "CTSB", "LRP1", "AXL", "RAB7A", "GAS6", "MERTK",  # phagocytosis and lysosomal activity
    "SIRPA", "HMOX1", "FTL" # erythroid remodeling/phagocytosis
]

# %%
# Thesis: Pro-inflammatory genes
gene_list = [
    "IL1B", "NLRP3", "CCL3", "TNF", "CYBB",  "CXCL8", "P2RX7", 
    "CASP1",  "NLRC4", "NFKB1", 
]

# %%
# Thesis: Scavenger Receptors
gene_list = [
    'CLEC7A', 'AXL', 'CD163', 'MRC1', 'MSR1', 'MERTK', 'CD68', 'MARCO', 'CD36', 'CALR', 'TIMD4', 'TYRO3', 'APOL2', 'SCARA5', 'FCGR1A'
    ]

# %%
# Thesis: Conserved KC markers
gene_list = [
            "GFRA2", "CD163", "HMOX1", "SLC1A3", "MARCO", "ADRB1", "TIMD4", "SUCNR1", "SLC40A1",  "SLC16A9", "FOLR2", "VCAM1", "TMEM26",  "CD5L"  
            ]
# not present in dataset: VSIG4

# %%
# Thesis: adhesion molecules
gene_list = [
    "AHNAK", "JAML", "ITGAX", "ICAM1", "MARCKS", "VCAN", "ADGRE5", "ITGAM", "CD44", "ITGB2", "ITGAL", # from GO analysis
    "SIGLEC1", "ITGB5", "ADGRE1",  "SIGLEC11", "SDC3", "ITGA9", "ITGAD", "ADGRG6", "BCAM", "VCAM1", "CDH5" # DEGs from liver cell atlas involved in cell adhesion
]


# %% [markdown]
# ### Select cells of interest

# %%
# macrophages + monocytes
cells_of_interest = ["Monocyte_I", "Monocyte_II", "Macrophage_I", "Macrophage_II",  "Macrophage_III"]

# %%
# macrophage trajectory
cells_of_interest = ["Hematopoietic Stem Cell", "Common Myeloid Progenitor", "Monocyte_I", "Monocyte_II", "Macrophage_I", "Macrophage_II", "Macrophage_III"]

# %%
# Blood trajectory
cells_of_interest = ["Hematopoietic Stem Cell", "Megakaryocyte Erythroid Progenitor", "Early Erythroid", "Mid Erythroid", "Late Erythroid", "Early Megakaryocyte", "Late Megakaryocyte"]

# %%
# macrophages + EBI macrophages
cells_of_interest = ["Macrophage_I", "Macrophage_II", "EBI Macrophages"]

# %%
# Myeloid cells
cells_of_interest = ["Hematopoietic Stem Cell", "Common Myeloid Progenitor", "Granulocyte", "Monocyte_I", "Monocyte_II", "Macrophage_I", "Macrophage_II", "Macrophage_III", "EBI Macrophages",
                     "Megakaryocyte Erythroid Progenitor", "Early Erythroid", "Mid Erythroid", "Late Erythroid", "Early Megakaryocyte", "Late Megakaryocyte"]

# %%
# Lymphoid cells
cells_of_interest = ["Hematopoietic Stem Cell", "Lymphoid Multipotent Progenitor", "Pre-B Cell", "Pro-B Cell", "Immature B Cell", "Natural Killer Cell", "ILC", "plamacytoid Dendritic Cell"]

# %%
# B cells
cells_of_interest = ["Hematopoietic Stem Cell", "Lymphoid Multipotent Progenitor", "Pre-B Cell", "Pro-B Cell", "Immature B Cell"]

# %%
# All cells
cells_of_interest = ["Hematopoietic Stem Cell", "Common Myeloid Progenitor", "Megakaryocyte Erythroid Progenitor", "Early Erythroid", "Mid Erythroid", "Late Erythroid",
                     "Early Megakaryocyte", "Late Megakaryocyte",
                     "Granulocyte", "Monocyte_I", "Monocyte_II", "Macrophage_I", "Macrophage_II",  "Macrophage_III", "EBI Macrophages",
                     "plasmacytoid Dendritic Cell", "Lymphoid Multipotent Progenitor", "Pro-B Cell", "Pre-B Cell", "Immature B Cell", "Natural Killer Cell", "ILC", 
                     "Hepatocytes", "LSECs",
                     "Low Expression"
                     ]

# %% [markdown]
# ### Run CHD on genes of interest on cells of interest

# %%
for gene in gene_list:
    if gene in transcriptome.var.symbol.tolist():
        print(gene)
        analyze_gene(gene, cells_of_interest, transcriptome, regionpositional, 
                     chd, fragments, clustering, dataset_name)
    else:
        print(f'Gene {gene} is not in current transcriptome selection')

# %% [markdown]
# ### Plot genes of interest

# %%
valid_genes = []

for gene in gene_list:
    print(gene)
    try:
        eyck.m.t.plot_umap(transcriptome, [gene], datashader=False).display()
        valid_genes.append(gene)
    except Exception as e:
        print(f"Error processing {gene}: {e}")

if valid_genes:
    print("Plotting all valid genes in one figure...")
    eyck.m.t.plot_umap(transcriptome, valid_genes, datashader=False).display()
else:
    print("No valid genes to plot.")


# %%
import pickle
folder_root = chd.get_output()
folder_data = folder_root / "data"
folder_data_preproc = folder_data_preproc = folder_data / dataset_name
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))

# %%
adata2 = adata[adata.obs["celltype"].isin(cells_of_interest)]

# %%
grouped_palette = {
    'Hematopoietic Stem Cell': '#67000d',
    'Common Myeloid Progenitor': '#a50f15',
    'Megakaryocyte Erythroid Progenitor': '#cb181d',
    'Early Erythroid': '#de2d26',
    'Mid Erythroid': '#fb6a4a',
    'Late Erythroid': '#fcae91',
    'Early Megakaryocyte': '#dd3497',
    'Late Megakaryocyte': '#f768a1',
    'Granulocyte': '#1a9850',
    'Monocyte_I': '#cc4c02',
    'Monocyte_II': '#993404',
    'Macrophage_I': '#fe9929',
    'Macrophage_II': '#fdae6b',
    'Macrophage_III': '#fdd49e',
    'plasmacytoid Dendritic Cell': '#a6d854',
    'EBI Macrophages': '#4daf4a',
    'Lymphoid Multipotent Progenitor': '#08519c',
    'Pro-B Cell': '#3182bd',
    'Pre-B Cell': '#6baed6',
    'Immature B Cell': '#9ecae1',
    'Natural Killer Cell': '#1c9099',
    'ILC': '#005b5b',
    'LSECs': '#969696',
    'Hepatocytes': '#d9d9d9',
    'Low Expression': '#bdbdbd'
}
palettes_1 = {"celltype": pd.Series(grouped_palette)}


# %%
eyck.m.t.plot_umap(adata, 
                   [
                    "celltype"],
                   norms=(0,"q.99"),
                   legend=None,
                   panel_size= 1.5,
                   ncol=4,
                   layer="magic",
                   datashader=False,
                   palettes=palettes_1
                   ).display()


# %%
sc.pl.dotplot(
            adata2, 
            var_names=gene_list,
            gene_symbols="symbol",
            groupby="celltype",
            layer="magic",
            )
