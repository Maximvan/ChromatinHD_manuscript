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
pp.paths.results().mkdir(parents=True, exist_ok=True)

# %%
dataset_name = "liverfoetalH"
genome = "hg38"
organism = "hs"
folder_dataset = chd.get_output() / "datasets" / dataset_name
folder_plots = chd.get_output() / "datasets" / dataset_name / "plots"
folder_plots.mkdir(exist_ok=True, parents=True)

# %%
folds = chd.data.folds.Folds(
    folder_dataset / "folds_all" # folds_all
)
transcriptome = chd.data.Transcriptome(folder_dataset / "transcriptome_all") # transcriptome_all
clustering = chd.data.Clustering.from_labels(
    transcriptome.obs["celltype"], path=folder_dataset / "clusterings_all" / "cluster" # clusterings_all
)
fragments = chd.data.Fragments(folder_dataset / "fragments_all" / "10k10k") # fragments_all
model_folder = (
    chd.get_output() / "diff" / "liverfoetalH_all" / "10k10k" / "cluster" # liverfoetalH_all
)

# %%
eyck.m.t.plot_umap(transcriptome, ["AZU1", "MPO", "celltype"], datashader = True).display()


# %% [markdown]
# ## Training

# %%
import chromatinhd.models.diff.model.binary

models = chd.models.diff.model.binary.Models.create(
    fragments = fragments,
    clustering=clustering,
    folds=folds,
    model_params = dict(
        encoder="shared",
        encoder_params=dict(
            delta_regularization=True,
            delta_p_scale=1.0,
            # bias_regularization=True,
            # bias_p_scale=0.5,
            binwidths=(5000, 1000, 500, 100, 50, 25),
        ),
    ),
    train_params = dict(
        early_stopping=False,
        n_epochs = 10, # <----- originally I used 40
    ),
    path=model_folder,
    reset = True,
)

# %%
chd.set_default_device("cuda:1")
models.train_models()
chd.set_default_device("cpu")

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
models = chd.models.diff.model.binary.Models(model_folder)

# %%
regionpositional = chd.models.diff.interpret.RegionPositional(
    model_folder / "scoring" / "regionpositional",
    reset=False,
)

# %%
regionpositional.score(
    models,
    fragments=fragments,
    clustering=clustering,
    device="cpu",
)
regionpositional


# %% [markdown]
# ## Run genes of interest

# %%
def analyze_gene(gene_marker, cells_of_interest, transcriptome, regionpositional, chd, fragments, clustering, dataset_name):
    gene_id = transcriptome.gene_id(gene_marker)
    windows = regionpositional.select_windows(
        gene_id,
        prob_cutoff=0.5,
        differential_prob_cutoff=3.,
        keep_tss=True,
        padding=1000,
    )
    breaking = chd.grid.Breaking(windows, resolution=4000, gap=0.03)

    motifscan_name = "hocomocov12_1e-4"
    genome_folder = pathlib.Path("/srv/data/genomes/GRCh38")
    motifscan_genome = chd.flow.Flow.from_path(genome_folder / "motifscans" / motifscan_name)
    motifscan = chd.data.motifscan.MotifscanView.from_motifscan(
        path=chd.get_output() / "datasets" / dataset_name / "motifscans" / "10k10k" / motifscan_name,
        parent=motifscan_genome,
        regions=fragments.regions,
    )

    motifs_oi = pd.concat([
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
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["LHX2"])], "group": "LHX2"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["FLI1"])], "group": "FLI1"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["GATA1"])], "group": "GATA1"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["GATA2"])], "group": "GATA2"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["MAF"])], "group": "MAF"}),
        pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["MAFB"])], "group": "MAFB"}),
        # pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["KLF2"])], "group": "KLF2"}),
        pd.DataFrame({"motif": motifscan.motifs.index[motifscan.motifs.tf.isin(["KLF4"])], "group": "KLF4"}),
        
    ]).set_index("motif")

    motif_groups = pd.DataFrame({"group": motifs_oi["group"].unique()}).set_index("group")
    motif_groups["label"] = motif_groups.index
    motif_groups["color"] = [mpl.colors.to_hex(mpl.colormaps["tab10"](i)) for i in range(len(motif_groups))]

    cluster_info = clustering.cluster_info.loc[cells_of_interest]
    slices = regionpositional.calculate_slices(-0.5, step=25)
    differential_slices = regionpositional.calculate_differential_slices(
        slices, fc_cutoff=1.5, a="Macrophage_II", b="Monocyte_I"
    )

    fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.01, padding_width=0.05))
    region = fragments.regions.coordinates.loc[gene_id]

    panel_genes = chd.plot.genome.genes.GenesExpanding.from_region(
        region,
        breaking=breaking,
        genome="GRCh38",
        xticks=[-10000, 0, 10000],
        gene_overlap_padding=100000 if windows["length"].sum() > 20000 else 10000,
        show_others=True,
        only_canonical=False,
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
    )
    fig.main.add_under(panel_differential)

    panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
        transcriptome,
        clustering,
        gene_id,
        cluster_info=cluster_info,
        show_n_cells=False,
        layer= "magic",
    )
    fig.main.add_right(panel_expression, panel_differential)

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
        only_canonical=False,
    )
    fig.main.add_under(panel_genes, padding=0.0, padding_up=0.2)


    fig_path = folder_plots / f"{gene_marker}_CHD.png"  # Specify the desired path and file name
    fig.savefig(str(fig_path), dpi=300, bbox_inches='tight')  # Save with high resolution and tight bounding box


# %%
# gene_id = transcriptome.gene_id("Slc40a1")
# gene_id = transcriptome.gene_id("CDH5")
#gene_id = transcriptome.gene_id("ID3")
# gene_id = transcriptome.gene_id("Lhx2")
# gene_id = transcriptome.gene_id("Bmp10")
# gene_id = transcriptome.gene_id("Lyve1")
# ["CD163", "CD68", "CTSB", "HLA-DPB1", "HMOX1", "MARCO", "CLEC4F", 
# "CD14", "TIMD4", "LYVE1", "VSIG4", "MERTK", "CD11b", "TLR4", "SPP1", 
# "CD5L", "SIGLEC1", "MSR1", "TREM2", "ITGAX", "STAB1", "FCGR1A", "FCGR3A", 
# "APOE", "C1QC", "C1QA", "C1QB", "MS4A4A", "GPNMB"]
# not in top 5000: CD68, CD14, TIMD4, LYVE1, VSIG4, MERTK, CD11B, 
# TREM2, FCGR1A, C1QC, C1QB

# %% [markdown]
# ### Select gene list

# %%
def gene_id(adata, symbol):
    return adata.var.reset_index().set_index("symbol").loc[symbol, "gene_ids"]


def symbol(adata, gene):
    return adata.var.reset_index().set_index("gene").loc[gene, "symbol"]

# %%
# Most dispersed genes
DSG = transcriptome.var["dispersions_norm"].nlargest(32).index.tolist()
gene_list = symbol(transcriptome, DSG).tolist()
gene_list.remove("AL589693.1")
gene_list.remove("STAB2")


# %%
# Non differential genes

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
# Top 20 KC markers (from Liver Atlas)
gene_list = [
            "CDH5", "TIMD4", "SLC16A9", "NDST3",
            "ITGAD",  "VCAM1", "SELENBP1",
            "BCAM", "CDH5", "CETP", "RND3",
            "FEZ1", "FABP3", "SDC3", "CXCL12",
            "ITLN1", "CTD-2337J16.1", "LYVE1"
            ]

# %%
# Top KC markers (conserved across species) (https://www.cell.com/immunity/fulltext/S1074-7613(22)00395-8)
gene_list = [
            "CD5L", "SLC1A3", "CD163", "FOLR2",
            "TIMD4", "MARCO", "GFRA2", "ADRB1", "TMEM26",
            "SLC40A1", "HMOX1", "SLC16A9", "VCAM1", "SUCNR1"
            ]

# %%
# Adhesion molecules
gene_list = [
            "ITGAL", "TLR4", "CDH1", "CDH5"
]


# %%
# FK1 (https://pmc.ncbi.nlm.nih.gov/articles/PMC7617064/#F3)
gene_list = [
            "RAB32", "HMOX1", "RNASE1", "ABCG2", "RPS2", "RPS17", "CORO1A", "RPS20", "STAB1"
]

# %%
# FK2 (https://pmc.ncbi.nlm.nih.gov/articles/PMC7617064/#F3)
gene_list = [
            "CREM", "RPS4Y1", "SRGN", "C5AR1", "VMO1", "PPP1CB", "FOSL2", "SH3BP5", "ELL2", "TIMP1"
]

# %%
# FK3 (https://pmc.ncbi.nlm.nih.gov/articles/PMC7617064/#F3)
gene_list = [
            "HLA-DRB1", "CD74", "HLA-DRA", "HLA-DPA1", "CST3", "FGL2", "HLA-DMA", "CTSS", "HLA-F", "CXCL12"
]

# %%
# Thesis: B cell maturation story
gene_list = [
            "FLT3", "IL7R", "EBF1", "PAX5", "RAG1", "RAG2", "BACH2", "SPIB", "FCRL1"
            ]

# %%
# Thesis: Blood cell maturation story
gene_list = [
            "FLI1", "GATA1", "GATA2", "TFRC", "GYPA"
]

# %%
# Thesis: Macrophage cell development story
gene_list = [
    "SPI1", "MAFB", "KLF4"
    ]

# %%
# Thesis: Macrophage cell maturation story
gene_list = [
    "RUNX1", "SPI1", "CSF1R"
    ]

# %%
# Thesis: classical macrophages vs EBI macrophages
gene_list = [
    'PLCG2', 'ANK1', 'SPTA1', 'WNT5B', 'HSPA5', 'ZBTB16', 'TFR2', 'RIPOR3', 'TAF1D', 'DNMT1',
    'FOSB', 'CD83', 'NAMPT', 'AHNAK', 'HLA-DQA1', 'RRP12', 'KLF6', 'CD74', 'RAB11FIP1',
    'SAMHD1', 'MYO1F', 'SAT1', 'ABR', 'CIITA', 'AOAH', 'FOS', 'STAB1', 'PLXDC2', 'CSF2RA'
    ]

# %%
# KC and MACRO markers (https://www.sciencedirect.com/science/article/pii/S0142961218307932?via%3Dihub)
gene_list = [
    "CD14", "CD68", "CD163",
    "CD200R1", "CD86", "CD83",
    "CD14", "FCGR2A", "CD68", "ITGAM",
    "CLEC4F", "ID3"
]

# %%
# liver-resident KC gene signature (https://www.sciencedirect.com/science/article/pii/S2589555921000549#appsec1)
# "CD5L"
gene_list = [
    "ARL4C", "CD163", "MERTK", "NR1H3", "SIGLEC1", "TIMD4", "VCAM1"
]

# %%
# Differential Immune Activation in Fetal Macrophage Populations (https://www.nature.com/articles/s41598-019-44181-8)
gene_list = [
    "IL1B", "CASP1", "ADGRE1", "ITGAM", "PTPRC", "FCGR1A", "CD68", "TNF", "HSP90AA1", "C1QA", "HSP90B1", "HSP90AB1", "TLR4"
]

# %%
# Niche signals and transcription factors involved in tissue-resident macrophage development (https://www.sciencedirect.com/science/article/pii/S0008874918300534)
gene_list = [
    "RUNX1", "SPI1", "CSF1R", "MAFB", "KLF2", "KLF4", "TIMD4", "CLEC4F", "SPIC", "NR1H3", "AIF1", "MERTK", "ADGRE1"
    ]

gene_list = [
    "RUNX1", "SPI1", "CSF1R"
]

# %%
# Fetal liver macrophages contribute to the hematopoietic stem cell niche by controlling granulopoiesis (Fetal liver macrophages contribute to the hematopoietic stem cell niche by controlling granulopoiesis)
gene_list = [
    "CLEC7A", "CCR2", "CX3CR1", "CSF1R", # macrophage precursor
    "ADGRE1", "SIGLEC1", "MSR1", "CD63", "MRC1", # macrophage core genes
    "TIMD4", "CLEC4F", "VCAM1" # KC genesS
]

# %% [markdown]
# ### Select cells of interest

# %%
# KC trajectory
cells_of_interest = ["HSCs", "CMP/GMP", "Monocyte_I", "Monocyte_II", "Monocyte_III", "Macrophage_I", "Macrophage_II"]

# %%
# Blood trajectory
cells_of_interest = ["HSCs", "MEP", "Early_Ery", "Mid_Ery", "Late_Ery", "Early_MK", "Late_MK"]

# %%
# KC + EBI
cells_of_interest = ["Macrophage_I", "Macrophage_II", "EBI Macrophages"]

# %%
# Myeloid cells
cells_of_interest = ["HSCs", "CMP/GMP", "Granulocyte", "Monocyte_I", "Monocyte_II", "KC_0", "KC_1", "KC_2", "KC_3",
                     "MEP", "Early_Ery", "Mid_Ery", "Late_Ery", "Early_MK", "Late_MK"]

# %%
# Lymphoid cells
cells_of_interest = ["HSCs", "LMPP", "Pre-Pro-B", "Pro-B", "Immature B", "NK cells", "ILC", "pDCS"]

# %%
# B cells
cells_of_interest = ["HSCs", "LMPP", "Pre-Pro-B", "Pro-B", "Immature B"]

# %%
# All cells
cells_of_interest = ["HSCs", "CMP/GMP", "Granulocyte", "Monocyte_I", "Monocyte_II", "KC_0", "KC_1", "KC_2", "KC_3",
                     "MEP", "Early_Ery", "Mid_Ery", "Late_Ery", "Early_MK", "Late_MK", "LMPP", "Pre-Pro-B", "Pro-B", "Immature B", "NK cells", "ILC", "pDCS",
                     "Hepatocytes", "Low_Expr", "LSECs", "Doublet_Ery_B"]

# %% [markdown]
# ### Run CHD on genes of interest on cells of interest

# %%
# gene_list = ["SIGLEC1", "MERTK", "CLEC4F", "CLEC7A", "CD163", "CD68", "MRC1", "HAVCR2"]
# gene_list = ["PRDX1", "PRDX2", "PRDX3", "PRDX4", "PRDX5", "PRDX6", "MPO", "LPO", "EPX", "PRDX1", "CYBA", "CYBB"]
gene_list = ['CSF1R']
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


# %%# %%
import pickle
folder_root = chd.get_output()
folder_data = folder_root / "data"
folder_data_preproc = folder_data_preproc = folder_data / dataset_name
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))
# %%
eyck.m.t.plot_umap(adata, gene_list, datashader=False).display()

# %% 
def gene_id(adata, symbol):
    return adata.var.reset_index().set_index("symbol").loc[symbol, "gene_ids"]
sc.pl.umap(adata, 
           color=(gene_id(adata, ["SPI1"])), 
           cmap="rocket",
           layer="magic")

# %%
adata2 = adata[adata.obs["celltype"].isin(cells_of_interest)]

gene_list = [
    "CD14", "CD68", "CD163",
    "CD200R1", "CD86", "CD83",
    "CD14", "FCGR2A", "CD68", "ITGAM",
    "CLEC4F", "ID3"
]

# %%
eyck.m.t.plot_umap(adata2, 
                   ["ANK1", "EMP2", "EPOR", "CSF2RA", "celltype"], 
                   norms=(0,"q.99"),
                   legend="under panel",
                   panel_size= 1.5,
                   #ncol=1,
                   layer="magic",
                   datashader=False).display()



















# %%
sc.tl.rank_genes_groups(
    adata2, "leiden", method="wilcoxon", key_added="wilcoxon", use_raw=False
)

# %%
# Extract the dictionary of gene names from recarray
top_genes = {cluster: adata2.uns["wilcoxon"]["names"][cluster] for cluster in adata2.uns["wilcoxon"]["names"].dtype.names}

# Convert to DataFrame
num_top_genes = 20  # Adjust the number of genes to retrieve
top_genes_df = pd.DataFrame({f"Cluster {cluster}": top_genes[cluster][:num_top_genes] for cluster in top_genes})

# %%
diffexp = (
    sc.get.rank_genes_groups_df(adata2, group=None, key="wilcoxon")
    .sort_values("scores", ascending=False)
    .groupby("group")
    .head(10)
)
diffexp["symbol"] = diffexp["names"].apply(lambda x: adata2.var.loc[x, "symbol"])

# %%
symbols_dict = {}
for i in range(27):
    symbols_dict[f"symbols{i}"] = diffexp[diffexp.group == str(i)]["symbol"].tolist()

def symbol(adata, gene):
    return adata.var.reset_index().set_index("gene").loc[gene, "symbol"]

# %%
sc.pl.dotplot(
            adata2, 
            var_names=symbol(adata2, np.unique(top_genes_df.iloc[:3].values.flatten())), 
            gene_symbols="symbol",
            groupby="celltype"
            )


# %%
