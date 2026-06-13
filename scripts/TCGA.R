library(TCGAbiolinks)
library(SummarizedExperiment)
library(dplyr)
library(purrr)
library(tibble)

# Get TCGA projects
projects_df <- getGDCprojects()
tcga_projects <- projects_df$project_id
tcga_projects <- tcga_projects[grepl("^TCGA-", tcga_projects)]

length(tcga_projects)
head(tcga_projects)

# Data query

get_project_data <- function(project,
                             sample_types = c("TP")) {
  
  message("Processing ", project)
  
  # mRNA: use STAR - Counts (GDC v32+)
  query_rna <- GDCquery(
    project       = project,
    data.category = "Transcriptome Profiling",
    data.type     = "Gene Expression Quantification",
    experimental.strategy = "RNA-Seq",
    workflow.type = "STAR - Counts"
  )
  GDCdownload(query_rna)
  se_rna <- GDCprepare(query_rna)
  
  # miRNA: keep as before (BCGSC miRNA Profiling still valid)
  query_mirna <- GDCquery(
    project       = project,
    data.category = "Transcriptome Profiling",
    data.type     = "miRNA Expression Quantification",
    workflow.type = "BCGSC miRNA Profiling"
  )
  GDCdownload(query_mirna)
  se_mirna <- GDCprepare(query_mirna)
  
  bar_rna   <- colnames(se_rna)
  bar_mirna <- colnames(se_mirna)
  
  bar_rna_tp   <- TCGAquery_SampleTypes(bar_rna,   typesample = sample_types)
  bar_mirna_tp <- TCGAquery_SampleTypes(bar_mirna, typesample = sample_types)
  
  se_rna   <- se_rna[,   bar_rna_tp]
  se_mirna <- se_mirna[, bar_mirna_tp]
  
  list(
    rna_counts    = assay(se_rna),
    mirna_counts  = assay(se_mirna),
    rna_coldata   = as.data.frame(colData(se_rna)),
    mirna_coldata = as.data.frame(colData(se_mirna))
  )
}

all_proj_data <- map(tcga_projects, get_project_data)
names(all_proj_data) <- tcga_projects

# Global Subtypes

subtypes_all <- PanCancerAtlas_subtypes()
head(subtypes_all)
# columns include: pan.samplesID, cancer.type, various subtype fields

# combine rna_coldata from all projects into one big table
rna_annot_all <- imap_dfr(
  all_proj_data,
  ~ .x$rna_coldata %>%
    rownames_to_column("barcode") %>%   # barcodes are rownames
    mutate(project = .y)
)

# join PanCancerAtlas subtypes
rna_annot_all <- rna_annot_all %>%
  left_join(subtypes_all, by = c("barcode" = "pan.samplesID"))

head(rna_annot_all)

# Tumor Purity
data("Tumor.purity", package = "TCGAbiolinks")
head(Tumor.purity)

rna_annot_all <- rna_annot_all %>%
  left_join(Tumor.purity, by = c("barcode" = "Sample.ID")) %>%
  mutate(tumor_fraction_CPE = 1 - CPE)
