###############################################################
#######       TCGA RECOUNCT3 - DATA DOWN-SAMPLING      ########
###############################################################

# Created: 11/03/2024
# Asta M. Rasmussen

##### packages #####
library(tidyverse)

###############################################
################## Load data ##################

# mount_path
mnt_path <- "~"
data_path <- paste0(mnt_path, "shared_data/TCGA_counts/")
out_path <- paste0(mnt_path, "data/downsampled_raw/")

############ Create different levels of down-sampling ############

##### Load data #####
library(parallel)

print("Loading data...")
mrna_rpkms <- readRDS(paste0(data_path, "TCGA_mrna_counts_match.rds")) # Normalized expression matrix of interest to use for down-sampling

# set parameters
n_genes <- dim(mrna_rpkms)[1]
n_samples <- dim(mrna_rpkms)[2]
n_cores <- 24

##### match levels used by MMN and check current sparsity levels for 10x technology #####
sparsity <- c(1000, 5000, 10000, 50000, 100000, 200000, 500000, 1000000, 5000000, 10000000) # number of sampled reads (library size

# Function to generate down-sampled datasets
fun_downsample <- function(lib_size, RPKMs, n_cores){
  set.seed(42)
  # generate down-sampled dataset
  samples_lst <- mclapply(1:n_samples, function(x) table(sample(1:n_genes, size=lib_size, replace=T, prob=RPKMs[,x])),
                          mc.cores=min(n_cores, getOption("mc.cores", 2L), n_samples))

  samples_mat <- matrix(0,nrow=dim(RPKMs)[1],ncol=dim(RPKMs)[2])
  for(i in 1:n_samples) samples_mat[as.numeric(names(samples_lst[[i]])),i] <- samples_lst[[i]]
  colnames(samples_mat) <- colnames(RPKMs)
  rownames(samples_mat) <- rownames(RPKMs)
  # Normalize and log transform
  #samples_mat_norm <- sweep(samples_mat, 2, colSums(samples_mat), FUN = '/')*median(colSums(samples_mat))
  #samples_mat_norm <- log2(samples_mat_norm+1)
  # Return down-sampled dataset
  return(samples_mat) # Returns raw counts now
}

## Generate down-sampled datasets ##
for (i in 1:length(sparsity)){
  print(paste0("Start generating down-sampled datasets with sparsity ", sparsity[i]))
  downsampled_data <- fun_downsample(sparsity[i], mrna_rpkms, n_cores)
  print("Saving down-sampled datasets...")
  saveRDS(downsampled_data, paste0(out_path, "TCGA_mrna_downsampled_", sparsity[i],".rds"))
  print(paste0("Down-sampled dataset with ", sparsity[i], " reads generated."))
}

print("Done!")

