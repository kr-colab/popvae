<img src="img/popvae_logo.svg" width="100%">

Dimensionality reduction for population genetic data with a variational autoencoder (VAE)

# Overview
popVAE fits a variational autoencoder to a set up input genotypes. A VAE is essentially a pair of neural networks which first encodes an input as a probability distribution in a low-dimensional latent space and then attempts to recreate the input given a location in latent space. Here's a figure describing the basic setup:  

<img src="img/vae_network.svg" width="70%">

By passing genotypes to a trained encoder we can visualize the relative differentiation among samples (more similar genotypes should be close to each other in latent space). This is similar to PCA, t-SNE, or UMAP, but with a user-defined number of dimensions and using a completely nonlinear framework. By passing latent space coordinates to the decoder we can also create new artificial genotypes characteristic of a given sample or population. 

A manuscript describing popVAE's methods and testing it on several empirical datasets can be found at: <TBD>

# Install
The `setup.py` script should take care of all dependencies for you. Clone this repo then install with 

```
cd popvae
python setup.py install
```

# Run
popVAE requires input genotypes in .vcf, .vcf.gz, or .zarr formats. This repo includes a test dataset of around 1,000 genome-wide SNPs from migratory Painted Buntings (from this paper: http://cjbattey.com/papers/pabu_amnat_final.pdf). Fit a model to this data with: 
  
  ```popvae.py --infile data/pabu/pabu_test_genotypes.vcf --out out/pabu_test --seed 42```

This model should fit in less than a minute on a regular laptop CPU. For running on larger datasets we strongly recommend using a CUDA-enabled GPU (typically 5 - 100x faster).

# Output
At default settings popvae will output 4 files:    
`pabu_test_latent_coords.txt` --  coordinates for all samples in latent space (.  
`pabu_test_history.txt` -- training and validation loss by epoch.  
`pabu_test_history.pdf` -- a plot of training and validation loss by epoch.  
`pabu_test_training_preds.txt` -- estimated latent coordinates output during model training, stored every `--prediction_freq` epochs.   

# Parameters
Many hyperparameters and filtering options can be adjusted at the command line.
Run `popvae.py --h` to see all parameters. 

Default settings work well on most datasets, but validation loss can usually be improved by tuning hyperparameters. We've seen most effects from changing three settings: network size, early stopping patience, and the proportion of samples used for model training versus validation. 

`--search_network_sizes` runs short optimizations for a range of network sizes and selects the network with lowest validation loss. Alternately, `--depth` and `--width` set the number of layers and the number of hidden units per layer in the network. If you're running low on GPU memory, reducing `--width` will help. 

`--patience` sets the number of epochs the optimizer will run after the last improvement in validation loss -- we've found that increasing this value (to, say, 300) sometimes helps with small datasets. 

`--train_prop` sets the proportion of samples used for model training, with the rest used for validation. 

# Plotting
Plot popVAE's latent_coords output just like a genotype PCA. For the test data a simple scatter plot can be produced in R with:  
``` 
library(ggplot2);library(data.table)
setwd("~/popvae/")
theme_set(theme_classic())

#load data
pd <- fread("out/pabu_test_latent_coords.txt",header=T)
names(pd)[1:2] <- c("LD1","LD2")
sd <- fread("data/pabu/pabu_test_sample_data.csv") #this has sample metadata for interpreting plots

#merge tables
pd <- merge(pd,sd,by="sampleID")

#plot VAE
ggplot(data=pd,aes(x=LD1,y=LD2,col=Longitude))+
  geom_point()
```
It should look something like this:  
![pabu_example_plot](https://github.com/cjbattey/popvae/blob/master/data/pabu/pabu_test_plot.png)

Note there are two main groups of samples corresponding to eastern and western sampling localities, as well as cline within the western group. These are allopatric (the big gap) and parapatric (the cline in western samples) breeding populations with different migratory strategies (see http://cjbattey.com/papers/pabu_amnat_final.pdf to compare these results with PCA and STRUCTURE). 

# Generating Artificial Genotypes
We're still working on the best way to allow users to generate artificial genotypes from trained models, since this tends to be a more interactive task than just fitting the model and visualizing the latent space. For now we have included a working example of fitting a VAE, generating artificial genotypes, and analyzing them with PCA and Admixture clustering at `scripts/popvae_decoder_HGDP_tests.py`. Stay tuned for updates. 



