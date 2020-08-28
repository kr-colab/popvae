import argparse
import os
import random
import re
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import allel
import h5py
import keras
import tensorflow
import zarr
from keras import backend as K
from keras import layers
from keras.layers.core import Lambda
from keras.models import Model, Sequential
from matplotlib import pyplot as plt
from streaming_data import DataGenerator
from tqdm import tqdm


def parse_arguments():
    #TODO maybe think about making this a tsv input? That's a lot of parameters
    parser=argparse.ArgumentParser()
    parser.add_argument("--infile", 
                        help="path to input genotypes in vcf (.vcf | .vcf.gz), \
                            zarr, or .popvae.hdf5 format. Zarr files should be as produced \
                            by scikit-allel's `vcf_to_zarr( )` function. `.popvae.hdf5`\
                            files store filtered genotypes from previous runs (i.e. \
                            from --save_allele_counts).")
    parser.add_argument("--out",default="vae", 
                        help="path for saving output")
    parser.add_argument("--patience",default=50,type=int,
                        help="training patience. default=50")
    parser.add_argument("--max_epochs",default=500,type=int,
                        help="max training epochs. default=500")
    parser.add_argument("--batch_size",default=32,type=int,
                        help="batch size. default=32")
    parser.add_argument("--save_allele_counts",default=False,action="store_true",
                        help="save allele counts and and sample IDs to \
                        out+'.popvae.hdf5'.")
    parser.add_argument("--save_weights",default=False,action="store_true",
                        help="save model weights to out+weights.hdf5.")
    parser.add_argument("--seed",default=None,type=int,help="random seed. \
                                                            default: None")
    parser.add_argument("--train_prop",default=0.9,type=float,
                        help="proportion of samples to use for training \
                            (vs validation). default: 0.9")
    parser.add_argument("--search_network_sizes",default=True,action="store_true",
                        help='run grid search over network sizes and use the network with \
                            minimum validation loss. default: True. ')
    parser.add_argument("--width_range",default="32,64,128,256,512",type=str,
                        help='range of hidden layer widths to test when `--search_network_sizes` is called.\
                            Should be a comma-delimited list with no spaces. Default: 32,64,128,256,512')
    parser.add_argument("--depth_range",default="3,6,10,20",type=str,
                        help='range of network depths to test when `--search_network_sizes` is called.\
                            Should be a comma-delimited list with no spaces. Default: 3,6,10,20')
    parser.add_argument("--depth",default=6,type=int,
                        help='number of hidden layers. default=6.')
    parser.add_argument("--width",default=128,type=int,
                        help='nodes per hidden layer. default=128')
    parser.add_argument("--gpu_number",default='0',type=str,
                        help='gpu number to use for training (try `gpustat` to get GPU numbers).\
                            Use ` --gpu_number "" ` to run on CPU, and  \
                            ` --parallel --gpu_number 0,1,2,3` to split batches across 4 GPUs.\
                            default: 0')
    parser.add_argument("--prediction_freq",default=5,type=int,
                        help="print predictions during training every \
                            --prediction_freq epochs. default: 10")
    parser.add_argument("--max_SNPs",default=None,type=int,
                        help="If not None, randomly select --max_SNPs variants \
                            to run. default: None")
    parser.add_argument("--latent_dim",default=2,type=int,
                        help="N latent dimensions to fit. default: 2")
    parser.add_argument("--PCA",default=False,action="store_true",
                        help="Run PCA on the derived allele count matrix in scikit-allel.")
    parser.add_argument("--n_pc_axes",default=20,type=int,
                        help="Number of PC axes to save in output. default: 20")
    parser.add_argument("--PCA_scaler",default="Patterson",type=str,
                        help="How should allele counts be scaled prior to running the PCA?. \
                            Options: 'None' (mean-center the data but do not scale sites), \
                            'Patterson' (mean-center then apply the scaling described in Eq 3 of Patterson et al. 2006, Plos Gen)\
                            default: Patterson. See documentation of allel.pca for further information.")
    parser.add_argument("--plot",default=False,action="store_true",
                        help="generate an interactive scatterplot of the latent space. requires --metadata. Run python scripts/plotvae.py --h for customizations")
    parser.add_argument("--metadata",default=None,
                        help="path to tab-delimited metadata file with column 'sampleID'.")
    args=parser.parse_args()
    

    args.depth_range=np.array([int(x) for x in re.split(",", args.depth_range)])
    args.width_range=np.array([int(x) for x in re.split(",", args.width_range)])

    if args.plot:
        if args.metadata==None:
            print("ERROR: `--plot` argument requires `--metadata`")
            exit()

    return args

def load_genotypes(infile, max_SNPs):
    """Loads genotypes and performs filtering/subsetting using any of the available filetypes
 
    Args:
        infile (str): Filename for input data
        max_SNPs (int): Maximum number of snps to subset to

    Returns:
        dc: Derived counts
        samples: sample array
    """
    print("\nLoading Genotypes")
    if infile.endswith('.zarr'):
        gen, samples = load_zarr(infile)
        ac, ac_all = get_allele_counts(gen, infile)
        missingness, dc, dc_all = drop_non_biallelic_sites(ac, ac_all, gen)
        dc, dc_all, missingness, ninds = drop_singletons(missingness, dc_all, dc)
        dc = subset_to_max_snps(max_SNPs, dc)
        dc = impute_dc(dc, dc_all, missingness, ninds)

    elif infile.endswith('.vcf') or infile.endswith('.vcf.gz'):
        gen, samples = load_vcf(infile)
        ac, ac_all = get_allele_counts(gen, infile)
        missingness, dc, dc_all = drop_non_biallelic_sites(ac, ac_all, gen)
        dc, dc_all, missingness, ninds = drop_singletons(missingness, dc_all, dc)
        dc = subset_to_max_snps(max_SNPs, dc)
        dc = impute_dc(dc, dc_all, missingness, ninds)

    elif infile.endswith('.popvae.hdf5'):
        dc, samples = load_hdf5(infile)

    return dc, samples

def load_zarr(infile):
    """Loads zarr filetype

    Args:
        infile (str): Filename

    Returns:
        genotype array
        samples
    """
    callset = zarr.open_group(infile, mode='r')
    gt = callset['calldata/GT']
    gen = allel.GenotypeArray(gt[:])
    samples = callset['samples'][:]
    
    return gen, samples

def load_vcf(infile):
    """Loads vcf file

    Args:
        infile (str): Filename

    Returns:
        genotype array: allel Genotype array object
        samples
    """
    vcf=allel.read_vcf(infile,log=sys.stderr)
    gen=allel.GenotypeArray(vcf['calldata/GT'])
    samples=vcf['samples']

    return gen, samples

def load_hdf5(infile):
    """Loads hdf5 file

    Args:
        infile (str): Filename

    Returns:
        dc: derived counts
        samples
    """
    h5=h5py.File(infile,'r')
    dc=np.array(h5['derived_counts'])
    samples=np.array(h5['samples'])
    h5.close()

    return dc, samples

def get_allele_counts(gen, infile):
    """Gets allele counts per-snp and per-snp-per-individual from genotypes

    Args:
        gen (genotype array): allel genotype array object
        infile (str): input file

    Returns:
        ac: allele count array per snp
        ac_all: allele count array per snp per individual
    """
    #snp filters
    if not infile.endswith('.popvae.hdf5'):
        print("counting alleles")
        ac_all=gen.count_alleles() #count of alleles per snp
        ac=gen.to_allele_counts() #count of alleles per snp per individual

    return ac, ac_all

def drop_non_biallelic_sites(ac, ac_all, gen):
    """Gets derived alleles by removing non-biallelic sites from all counts

    Args:
        ac (ndarray): individual snps
        ac_all (ndarray): individual snps per individual
        gen (genotype array): allel genotype array

    Returns:
       missingness: Biallelic sites that are missing from dataset
        dc: Derived counts with just biallelic sites
        dc_all: Derived counts from all individs
    """
    print("Dropping non-biallelic sites")
    biallel=ac_all.is_biallelic()
    dc_all=ac_all[biallel,1] #derived alleles per snp
    dc=np.array(ac[biallel,:,1],dtype="int_") #derived alleles per individual
    missingness=gen[biallel,:,:].is_missing()

    return missingness, dc, dc_all

def drop_singletons(missingness, dc_all, dc):
    """Drops singletons from derived counts on per-individual basis

    Args:
        missingness (ndarray): Array containing missing values
        dc_all (ndarray): Derived counts on per snp per individual basis
        dc (ndarray): Derived counts

    Returns:
        dc: derived counts, without singletons
        dc_all: derived counts all, without singletons
        missingness: without singletons
        ninds: indices of everything not singleton
    """
    print("Dropping singletons")
    ninds=np.array([np.sum(x) for x in ~missingness])
    singletons=np.array([x<=2 for x in dc_all])
    dc_all=dc_all[~singletons]
    dc=dc[~singletons,:]
    ninds=ninds[~singletons]
    missingness=missingness[~singletons,:]

    return dc, dc_all, missingness, ninds

def impute_dc(dc, dc_all, missingness, ninds):
    """Generate missing derived counts using all counts and binomial distribution

    Args:
        dc (ndarray): Derived counts
        dc_all (ndarray): Derived counts per snp per person
        missingness (ndarray): Missing counts
        ninds (ndarray): Indices of non-singleton locations

    Returns:
        dc: derived counts, filled with imputed data
    """
    print("Filling missing data with rbinom(2,derived_allele_frequency)")
    af=np.array([dc_all[x]/(ninds[x]*2) for x in range(dc_all.shape[0])])
    for i in tqdm(range(np.shape(dc)[1])):
        indmiss=missingness[:,i]
        dc[indmiss,i]=np.random.binomial(2,af[indmiss])

    dc=np.transpose(dc)
    dc=dc*0.5 #0=homozygous reference, 0.5=heterozygous, 1=homozygous alternate

    return dc

def save_hdf5(infile, prune_LD, dc, samples):
    """Saved h5 file for re-analysis

    Args:
        infile (str): Filename of input file
        prune_LD (bool): Whether to prune LD or not
        dc (ndarray): Derived counts, filtered, imputed 
        samples (ndarray): Samples
    """
    #save hdf5 for reanalysis
    print("saving derived counts for reanalysis")
    if prune_LD:
        outfile=h5py.File(infile+".LDpruned.popvae.hdf5", "w")
    else:
        outfile=h5py.File(infile+".popvae.hdf5", "w")
    outfile.create_dataset("derived_counts", data=dc)
    outfile.create_dataset("samples", data=samples,dtype=h5py.string_dtype()) #requires h5py >= 2.10.0
    outfile.close()

def subset_to_max_snps(max_SNPs, dc):
    """Subsets SNPs to maximum desired count

    Args:
        max_SNPs (int): Max snps to sample down to
        dc (ndarray): Derived counts

    Returns:
        dc: derived counts, filtered to only be length of max_snps
    """
    if not max_SNPs==None:
        print("subsetting to "+str(max_SNPs)+" SNPs")
        dc=dc[:,np.random.choice(range(dc.shape[1]),max_SNPs,replace=False)]
    
    return dc

def split_training_data(dc, samples, train_prop):
    """Splits data into training/testing data using sklearn train/test/split 
        or shuffles if train_prop is 1

    Args:
        dc (ndarray): Derived counts, filtered/imputed
        samples (ndarray): Samples
        train_prop (float): Percent of training proportion from whole sample

    Returns:
        trainsamples/testsamples: Training, testing sample lists
        traingen/testgen: Training, testing genotype lists
    """
    print("Running train/test splits")
    ninds=dc.shape[0]
    if train_prop==1:
        train_inds = np.arange(ninds)
        np.random.shuffle(train_inds)
        test_inds = train_inds
        traingen=dc[train,:]
        testgen=dc[test,:]
        trainsamples=samples[train]
        testsamples=samples[test]
    else:
        train_inds, test_inds = train_test_split(np.arange(ninds), train_size=train_prop)
        traingen=dc[train_inds,:]
        testgen=dc[test_inds,:]
        trainsamples=samples[train_inds]
        testsamples=samples[test_inds]

    print('Validation Samples:'+ testsamples +'\n')
    print('Running on '+str(dc.shape[1])+" SNPs")

    return trainsamples, testsamples, traingen, testgen

def saveLDpos(encoder, predgen, samples, batch_size, epoch, frequency, out):
    """Runs predictions from encoder and saves latent space to csv
    Logging function for Keras

    Args:
        encoder (keras model): Trained encoder model
        predgen (ndarray): Genotypes to predict latent space from ??????
        samples (ndarray): Samples
        batch_size (int): Batch size
        epoch (int): Number of epochs to run predictions for
        frequency (int): How often to save predictions
        out (str): Output file
    """
    if(epoch%frequency==0):
        pred=encoder.predict(predgen,batch_size=batch_size)[0]
        pred=pd.DataFrame(pred)
        pred['sampleID']=samples
        pred['epoch']=epoch
        if(epoch==0):
            pred.to_csv(out+"_training_preds.txt",sep='\t',index=False,mode='w',header=True)
        else:
            pred.to_csv(out+"_training_preds.txt",sep='\t',index=False,mode='a',header=False)
    
def sampling(args, latent_dim):
    """A Lambda function \n
    Too much math in here for me \n 
    Enjoy this haiku\n
    """
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                            mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

def create_vae(traingen, width, depth, latent_dim):
    """Builds the whole VAE end to end network

    Args:
        traingen (ndarray): Training genotypes
        width (int): Width of net (number of nodes per layer)
        depth (int): Depth of net (number of layers)
        latent_dim (int): Dimensionality of latent space

    Returns:
        vae: VAE model
        encoder: Encoder portion of vae model
        input_seq: Keras Input layer of shape (traingen.shape[1],)

    The reason encoder is functionalized is because it needs to be called for a few things, 
    but decoder is only used for end-to-end vae prediction, so I'm leaving it here.
    """
    #Encoder
    input_seq = keras.Input(shape=(traingen.shape[1],))
    x=layers.Dense(width,activation="elu")(input_seq)
    for i in range(depth-1):
        x=layers.Dense(width,activation="elu")(x)
    z_mean=layers.Dense(latent_dim)(x)
    z_log_var=layers.Dense(latent_dim)(x)
    z = layers.Lambda(sampling,output_shape=(latent_dim,), name='z', arguments={'latent_dim':latent_dim})([z_mean, z_log_var])
    encoder=Model(input_seq,[z_mean,z_log_var,z],name='encoder')

    #decoder
    decoder_input=layers.Input(shape=(latent_dim,),name='z_sampling')
    x=layers.Dense(width,activation="linear")(decoder_input)#was elu
    for i in range(depth-1):
        x=layers.Dense(width,activation="elu")(x)
    output=layers.Dense(traingen.shape[1],activation="sigmoid")(x) #hard sigmoid seems natural here but appears to lead to more left-skewed decoder outputs.
    decoder=Model(decoder_input,output,name='decoder')

    #end-to-end vae
    output_seq = decoder(encoder(input_seq)[2])
    vae = Model(input_seq, output_seq, name='vae')

    #get loss as xent_loss+kl_loss
    reconstruction_loss = keras.losses.binary_crossentropy(input_seq,output_seq)
    reconstruction_loss *= traingen.shape[1]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    #kl_loss *= 5 #beta from higgins et al 2017, https://openreview.net/pdf?id=Sy2fzU9gl. Deprecated but loss term weighting is a thing to work on.
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    return vae, encoder, input_seq

def train_model(vae, encoder, dc, samples, traingen, testgen, user_args):
    """Trains given VAE model

    Args:
        vae (keras model): Built keras model with custom loss function
        encoder (keras model): Encoder section of vae
        dc (ndarray): Derived counts
        samples (ndarray): All samples
        traingen (ndarray): Training genotypes
        testgen (ndarray): Testing genotypes
        user_args (dict): Dictionary of user arguments to script

    Returns:
        history: History object from best model
        vae: fitted VAE model
        vaetime: float time of train
    
    """
    vae.compile(optimizer='adam')

    checkpointer=keras.callbacks.ModelCheckpoint(
                    filepath=user_args.out+"_weights.hdf5",
                    verbose=2,
                    save_best_only=True,
                    monitor="val_loss",
                    period=1)

    earlystop=keras.callbacks.EarlyStopping(monitor="val_loss",
                                            min_delta=0,
                                            patience=user_args.patience)

    reducelr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.5,
                                                patience=int(user_args.patience/4), # This is being divided by 4 twice? Is that supposed to be the case?
                                                verbose=0,
                                                mode='auto',
                                                min_delta=0,
                                                cooldown=0,
                                                min_lr=0)

    print_predictions=keras.callbacks.LambdaCallback(
            on_epoch_end= lambda epoch, 
            logs: saveLDpos(encoder=encoder,
                            predgen=dc,
                            samples=samples,
                            batch_size=user_args.batch_size,
                            epoch=epoch,
                            frequency=user_args.prediction_freq,
                            out=user_args.out))

    t1=time.time()
    history=vae.fit(x=traingen,
                    y=None,
                    shuffle=True,
                    verbose=2,
                    epochs=user_args.max_epochs,
                    callbacks=[checkpointer,earlystop,reducelr,print_predictions],
                    validation_data=(testgen,None),
                    batch_size=user_args.batch_size)
    t2=time.time()
    vaetime=t2-t1
    print("VAE run time: "+str(vaetime)+" seconds")

    return history, vae, vaetime

def append_min_losses(history, width, depth, param_losses):
    """Gets minimum loss from history and returns to param_losses dataframe

    Args:
        history (Keras History): Keras history object for the model just trained
        width (int): Width of model (nodes per layer)
        depth (int): Depth of model (layers)
        param_losses (pd.df): Losses dataframe to append new minimum loss to

    Returns:
        param_losses: Losses from each parameterization
    """
    minloss=np.min(history.history['val_loss'])
    row=np.transpose(pd.DataFrame([width,depth,minloss]))
    row.columns=['width','depth','val_loss']
    param_losses=param_losses.append(row,ignore_index=True)
    K.clear_session() #maybe solves the gpu memory issue(???)

    return param_losses

def get_best_losses(param_losses, user_args):
    """Gets best parameters from loss values and prints

    Args:
        param_losses (dict): Dictionary of losses calculated during training
        user_args (dict): Dictionary of user arguments to script

    Returns:
        width: Width of best network
        depth: Depth of best network
    """
    #save tests and get min val_loss parameter set
    print(param_losses)
    param_losses.to_csv(user_args.out+"_param_grid.csv",index=False,header=True)
    bestparams=param_losses[param_losses['val_loss']==np.min(param_losses['val_loss'])]
    width=int(bestparams['width'])
    depth=int(bestparams['depth'])
    print('Best parameters:\nWidth = '+str(width)+'\nDepth = '+str(depth))

    return width, depth

def final_model_run(width, depth, dc, samples, traingen, testgen, train_samples, test_samples, user_args):
    """What if you're too good for grid search? Then you only run this.
    Same workflow as grid search, just leave out all the reparameterizations

    Args:
        width (int): Width of model (nodes per layer)
        depth (int): Depth of model (layers)
        dc (ndarray): Derived counts
        samples (ndarray): All samples
        traingen (ndarray): Training genotypes
        testgen (ndarray): Testing genotypes
        train_samples (ndarray): Training samples
        test_samples (ndarray): Testing samples
        user_args (dict): Dictionary with user arguments to script

    Returns:
        history: History object from best model
        vae: fitted VAE model
        vaetime: Time to train final model
    """
                                                                
    vae, encoder, input_seq = create_vae(traingen, width, depth, user_args.latent_dim)
   
    t1=time.time()

    history, vae, vaetime = train_model(vae, encoder, dc, samples, traingen, testgen, user_args)
    
    t2 = time.time()
    vatime = t2-t1

    return history, vae, vaetime

def grid_search(dc, samples, traingen, testgen, train_samples, test_samples, user_args):
    """Grid search through network parameterizations
    - Iterates through next set of parameters
    - Creates net
    - Trains net
    - Gets best losses and spits out optimal width and depth to use for optimal encoder

    Args:
        dc (ndarray): Derived counts
        traingen (ndarray): Training genotypes
        testgen (ndarray): Testing genotypes
        train_samples (ndarray): Training samples
        test_samples (ndarray): Testing samples
        latent_dim (int): Number of latent dimensions
        user_args (dict): Dictionary with user arguments to script

    Returns:
        best_width: Best width
        best_depth: Best depth
    """

    #grid search on network sizes. Getting OOM errors on 256 networks when run in succession -- GPU memory not clearing on new compile? unclear.
    #Wonder if switching to something pre-written would help? Talos, for example?
    print('Running grid search on network sizes')
    user_args.patience = user_args.patience / 4

    #get parameter combinations (will need to rework this for >2 params)
    # OOB gridsearch will make that ^ very easy -Logan
    paramsets=[[x,y] for x in user_args.width_range for y in user_args.depth_range]

    #output dataframe
    param_losses=pd.DataFrame()
    param_losses['width']=None
    param_losses['depth']=None
    param_losses['val_loss']=None

    #params=paramsets[0]
    for params in tqdm(paramsets):
        width=params[0]
        depth=params[1]
        print('width='+str(width)+'\ndepth='+str(depth))
                                                                    
        vae, encoder, input_seq = create_vae(traingen, width, depth, user_args.latent_dim)
    
        t1=time.time()

        history, vae, vaetime = train_model(vae, encoder, dc, samples, traingen, testgen, user_args)
        param_losses = append_min_losses(history, width, depth, param_losses)

    best_width, best_depth = get_best_losses(param_losses, user_args)

    return best_width, best_depth

def save_training_history(history, out):
    """Saves history to text file

    Args:
        history (keras history): Trained model history
        out (str): Filename for output file prefix
    """
    #save training history
    h=pd.DataFrame(history.history)
    h.to_csv(out+"_history.txt",sep="\t")

def predict_latent_coords(dc, samples, traingen, width, depth, user_args):
    """Get latent space mean and var from trained encoder, write coordinates to csv

    Args:
        dc (ndarray): Derived count
        samples (ndarray): Samples
        traingen (ndarray): Training genotypes
        width (int): Width of net
        depth (int): Depth of net
        user_args (dict): Dictionary with user arguments to script
    """

    vae, encoder, input_seq = create_vae(traingen, width, depth, user_args.latent_dim)

    #predict latent space coords for all samples from weights minimizing val loss
    vae.load_weights(user_args.out+"_weights.hdf5")
    pred=encoder.predict(dc,batch_size=user_args.batch_size) #returns [mean,sd,sample] for individual distributions in latent space
    p=pd.DataFrame()
    if user_args.latent_dim==2:
        p['mean1']=pred[0][:,0]
        p['mean2']=pred[0][:,1]
        p['sd1']=pred[1][:,0]
        p['sd2']=pred[1][:,1]
        pred=p
    else:
        pred=pd.DataFrame(pred[0])
        pred.columns=['LD'+str(x+1) for x in range(len(pred.columns))]
    pred['sampleID']=samples
    pred.to_csv(user_args.out+'_latent_coords.txt',sep='\t',index=False)

def run_PCA(dc, PCA_scaler, n_pc_axes):
    """Runs PCA on derived count data to compare against VAE performance, saves to csv

    Args:
        dc (ndarray): Derived counts
        PCA_scaler (str): Scaler to use for PCA
        n_pc_axes (int): Number of components to return from PCA

    Returns:
        float: PCAtime, time took to complete PCA process
    """
    pcdata=np.transpose(dc)
    t1=time.time()
    print("running PCA")
    pca=allel.pca(pcdata,scaler=PCA_scaler,n_components=n_pc_axes)
    pca=pd.DataFrame(pca[0])
    colnames=['PC'+str(x+1) for x in range(n_pc_axes)]
    pca.columns=colnames
    pca['sampleID']=samples
    pca.to_csv(out+"_pca.txt",index=False,sep="\t")
    t2=time.time()
    pcatime=t2-t1
    print("PCA run time: "+str(pcatime)+" seconds")

    return pcatime

def plot_history(history):
    """Plots training history of best model, saves to file

    Args:
        history (keras history): Keras History object, gotten from best VAE model trianing
    """
    #training history
    #plt.switch_backend('agg')
    fig = plt.figure(figsize=(3,1.5),dpi=200)
    plt.rcParams.update({'font.size': 7})
    ax1=fig.add_axes([0,0,1,1])
    ax1.plot(history.history['val_loss'][3:],"--",color="black",lw=0.5,label="Validation Loss")
    ax1.plot(history.history['loss'][3:],"-",color="black",lw=0.5,label="Training Loss")
    ax1.set_xlabel("Epoch")
    #ax1.set_yscale('log')
    ax1.legend()
    fig.savefig(out+"_history.pdf",bbox_inches='tight')

def write_times(vaetime, pcatime, out):
    """Writes runtimes to text

    Args:
        vaetime (float): Time for vae to run
        pcatime (float): Time for pca to run
        out (str): Prefix for outfile
    """
    if PCA:
        timeout=np.array([vaetime,pcatime])
        np.savetxt(X=timeout,fname=out+"_runtimes.txt")

def run_plotter(out, metadata):
    """Subprocess call to run plotter on latent space coordinates

    Args:
        out (str): Filename to prefix onto written plot files
        metadata (str): Metadata supplied by user
    """
    subprocess.run("python scripts/plotvae.py --latent_coords "+out+'_latent_coords.txt'+' --metadata '+metadata,shell=True)

def main():

    user_args = parse_arguments()

    #OS Side settings and random seeds
    os.environ["CUDA_VISIBLE_DEVICES"]=user_args.gpu_number

    if not user_args.seed==None:
        os.environ['PYTHONHASHSEED']=str(user_args.seed)
        random.seed(user_args.seed)
        np.random.seed(user_args.seed)
        tensorflow.set_random_seed(user_args.seed)

    #Get data
    dc, samples = load_genotypes(user_args.infile, user_args.max_SNPs)
    train_samples, test_samples, traingen, testgen = split_training_data(dc, samples, user_args.train_prop)

    if user_args.search_network_sizes: #Grid search
        # Should reduce the parameterization of this function, but it's handy for now
        print("Grid Search")
        best_width, best_depth = grid_search(dc, samples, traingen, testgen, train_samples, test_samples, user_args)
    else:
        best_width = user_args.width
        best_depth = user_args.depth   

    print("Final Model")
    history, vae, vaetime = final_model_run(best_width, best_depth, dc, samples, traingen, testgen, train_samples, test_samples, user_args)
    save_training_history(history, user_args.out)
    predict_latent_coords(dc, samples, traingen, best_width, best_depth, user_args)

    if user_args.PCA:
        run_PCA(dc, pcdata, PCA_scaler, n_pc_axes)
        plot_PCA(vaetime, pcatime, out)

    if user_args.plot:
        run_plotter(out, metadata)

    if not user_args.save_weights:
        subprocess.check_output(['rm',user_args.out+"_weights.hdf5"])

    if user_args.save_allele_counts and not user_args.infile.endswith('.popvae.hdf5'):
        save_hdf5(user_args.infile, True, dc, samples) 
        #I have no idea where prune_LD is coming from, so setting it to default True for now

if __name__ == "__main__":
    main()

# ###debugging parameters
# os.chdir("/Users/cj/popvae/")
# infile="data/pabu/pabu_testgenotypes.vcf"
# sample_data="data/pabu/pabu_test_sample_data.txt"
# save_allele_counts=True
# patience=100
# batch_size=32
# max_epochs=300
# seed=12345
# save_weights=False
# train_prop=0.9
# gpu_number='0'
# prediction_freq=2
# out="out/test"
# latent_dim=2
# max_SNPs=10000
# PCA=True
# depth=6
# width=128
# parallel=False
# prune_iter=1
# prune_size=500
# PCA_scaler="Patterson"
