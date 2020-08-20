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
    parser.add_argument("--search_network_sizes",default=False,action="store_true",
                        help='run grid search over network sizes and use the network with \
                            minimum validation loss. default: False. ')
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

    user_args = {
        'infile': args.infile
        'save_allele_counts': args.save_allele_counts,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'seed': args.seed,
        'save_weights': args.save_weights,
        'train_prop': args.train_prop,
        'gpu_number': args.gpu_number,
        'out': args.out,
        'prediction_freq': args.prediction_freq,
        'max_SNPs': args.max_SNPs,
        'latent_dim': args.latent_dim,
        'PCA': args.PCA,
        'PCA_scaler': args.PCA_scaler,
        'depth': args.depth,
        'width': args.width,
        'n_pc_axes': args.n_pc_axes,
        'search_network_sizes': args.search_network_sizes,
        'plot': args.plot,
        'metadata': args.metadata
        'depth_range': args.depth_range
        'width_range': args.width_range
    }

    user_args['depth_range']=np.array([int(x) for x in re.split(",", user_args['depth_range'])])
    user_args['width_range']=np.array([int(x) for x in re.split(",", user_args['width_range'])])

    if args.plot:
        if args.metadata==None:
            print("ERROR: `--plot` argument requires `--metadata`")
            exit()

    return user_args

def load_genotypes(infile, max_SNPs):
    """Loads genotypes and performs filtering/subsetting using any of the available filetypes
 
    Args:
        infile (str): Filename for input data
        max_SNPs (int): Maximum number of snps to subset to

    Returns:
        ndarray: dc, Derived counts
        ndarray: samples, sample array
    """
    print("\nLoading Genotypes")
    if infile.endswith('.zarr'):
        gen, samples = load_zarr(infile)
        ac, ac_all = get_allele_counts(gen)
        missingness, dc, dc_all = drop_non_biallelic_sites(ac, ac_all)
        dc, missingness = drop_singletons(missingness, dc_all)
        dc = subset_to_max_snps(max_SNPs, dc)
        dc = impute_dc(dc, dc_all, missingness)

    elif infile.endswith('.vcf') or infile.endswith('.vcf.gz'):
        gen, samples = load_vcf(infile)
        ac, ac_all = get_allele_counts(gen)
        missingness, dc, dc_all = drop_non_biallelic_sites(ac, ac_all)
        dc, missingness = drop_singletons(missingness, dc_all)
        dc = subset_to_max_snps(max_SNPs, dc)
        dc = impute_dc(dc, dc_all, missingness)

    elif infile.endswith('.popvae.hdf5'):
        dc, samples = load_hdf5(infile)

    return dc, samples

def load_zarr(infile):
    """Loads zarr filetype

    Args:
        infile (str): Filename

    Returns:
        gen: genotype array
        ndarray: samples
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
        ndarray: samples
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
        ndarray: derived counts
        ndarray: samples
    """
    h5=h5py.File(infile,'r')
    dc=np.array(h5['derived_counts'])
    samples=np.array(h5['samples'])
    h5.close()

    return dc, samples

def get_allele_counts(gen):
    """Gets allele counts per-snp and per-snp-per-individual from genotypes

    Args:
        gen (genotype array): allel genotype array object

    Returns:
        ndarray: ac, allele count array per snp
        ndarray: ac_all, allele count array per snp per individual
    """
    #snp filters
    if not infile.endswith('.popvae.hdf5'):
        print("counting alleles")
        ac_all=gen.count_alleles() #count of alleles per snp
        ac=gen.to_allele_counts() #count of alleles per snp per individual

    return ac, ac_all

def drop_non_biallelic_sites(ac, ac_all):
    """Gets derived alleles by removing non-biallelic sites from all counts

    Args:
        ac (ndarray): individual snps
        ac_all (ndarray): individual snps per individual

    Returns:
        ndarray: Biallelic sites that are missing from dataset
        ndarray: Derived counts with just biallelic sites
        ndarray: Derived counts from all individs
    """
    print("Dropping non-biallelic sites")
    biallel=ac_all.is_biallelic()
    dc_all=ac_all[biallel,1] #derived alleles per snp
    dc=np.array(ac[biallel,:,1],dtype="int_") #derived alleles per individual
    missingness=gen[biallel,:,:].is_missing()

    return missingness, dc, dc_all

def drop_singletons(missingness, dc_all):
    """Drops singletons from derived counts on per-individual basis

    Args:
        missingness (ndarray): Array containing missing values
        dc_all (ndarray): Derived counts on per snp per individual basis

    Returns:
        ndarray: derived counts, without singletons
        ndarray: missingness, without singletons
    """
    print("dropping singletons")
    ninds=np.array([np.sum(x) for x in ~missingness])
    singletons=np.array([x<=2 for x in dc_all])
    dc_all=dc_all[~singletons]
    dc=dc[~singletons,:]
    ninds=ninds[~singletons]
    missingness=missingness[~singletons,:]

    return dc, missingness

def impute_dc(dc, dc_all, missingness):
    """Generate missing derived counts using all counts and binomial distribution

    Args:
        dc (ndarray): Derived counts
        dc_all (ndarray): Derived counts per snp per person
        missingness (ndarray): Missing counts

    Returns:
        ndarray: derived counts, filled with imputed data
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
        ndarray: derived counts, filtered to only be length of max_snps
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
        ndarray: Training, testing sample lists
        ndarray: Training, testing genotype lists
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

def sampling(args):
    """A Lambda function \n
       Too much statistics for me \n 
       Enjoy this haiku\n
    """
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

def saveLDpos(encoder, predgen, samples, batch_size, epoch, frequency):
    """Runs predictions from encoder and saves latent space to csv
    Logging function for Keras

    Args:
        encoder (keras model): Trained encoder model
        predgen (ndarray): Genotypes to predict latent space from ??????
        samples (ndarray): Samples
        batch_size (int): Batch size
        epoch (int): Number of epochs to run predictions for
        frequency (int): How often to save predictions
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

def create_encoder(traingen, width, depth, latent_dim):
    """Builds encoder network for first half of VAE

    Args:
        traingen (ndarray): training dataset
        width (int): Width of network
        depth (int): Depth of network
        latent_dim (int): Latent dimensions to use

    Returns:
        keras model: built encoder model
    """
    input_seq = keras.Input(shape=(traingen.shape[1],))
    x=layers.Dense(width,activation="elu")(input_seq)
    for i in range(depth-1):
        x=layers.Dense(width,activation="elu")(x)
    z_mean=layers.Dense(latent_dim)(x)
    z_log_var=layers.Dense(latent_dim)(x)
    z = layers.Lambda(sampling(), output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder=Model(input_seq,[z_mean,z_log_var,z],name='encoder')

    return encoder

def create_vae(traingen, width, depth, latent_dim, sampling, encoder):
    """Builds the whole VAE end to end network

    Args:
        traingen (ndarray): Training genotypes
        width (int): Width of net (number of nodes per layer)
        depth (int): Depth of net (number of layers)
        latent_dim (int): Dimensionality of latent space
        sampling (func): Function call for  function sampling()
        encoder (keras model): Built encoder model to use for first half of VAE

    Returns:
        keras model: VAE model
        keras layer: Keras Input layer of shape (traingen.shape[1],)
        keras layer: Keras output layer of shape (traingen.shape[1],) after decoder generates data

    The reason encoder is functionalized is because it needs to be called for a few things, 
    but decoder is only used for end-to-end vae prediction, so I'm leaving it here.
    """
    #Decoder
    decoder_input=layers.Input(shape=(latent_dim,),name='z_sampling')
    x=layers.Dense(width,activation="linear")(decoder_input)#was elu
    for i in range(depth-1):
        x=layers.Dense(width,activation="elu")(x)
    output=layers.Dense(traingen.shape[1],activation="sigmoid")(x) #hard sigmoid seems natural here but appears to lead to more left-skewed decoder outputs.
    decoder=Model(decoder_input,output,name='decoder')

    #end-to-end vae
    output_seq = decoder(encoder(input_seq)[2])
    vae = Model(input_seq, output_seq, name='vae')

    return vae, input_seq, output_seq

def add_loss_function(vae, input_seq, output_seq):
    """Adds combined loss from KL and Reconstruction loss to VAE model

    Args:
        vae (keras model): Built VAE end-to-end model
        input_seq (keras layer): Input array to the network (ndarray?)
        output_seq (keras layer): Output array from generative net

    Returns:
        keras model: VAE model with custom loss function added
    """
    #get loss as xent_loss+kl_loss
    reconstruction_loss = keras.losses.binary_crossentropy(input_seq,output_seq)
    reconstruction_loss *= traingen.shape[1]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    #kl_loss *= 5 #beta from higgins et al 2017, https://openreview.net/pdf?id=Sy2fzU9gl. Deprecated but loss term weighting is a thing to work on.
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    return vae

def train_model(vae, out, grid_patience, print_predictions):
    """Trains given VAE model

    Args:
        vae (keras model): Built keras model with custom loss function
        out (str): Output filename
        grid_patience (int): Patience to use for early stopping of val_loss
        print_predictions (func): Function for callback

    Returns:
        keras history: History object from best model
        keras model: fitted VAE model
    """
    vae.compile(optimizer='adam')

    checkpointer=keras.callbacks.ModelCheckpoint(
                    filepath=out+"_weights.hdf5",
                    verbose=2,
                    save_best_only=True,
                    monitor="val_loss",
                    period=1)

    earlystop=keras.callbacks.EarlyStopping(monitor="val_loss",
                                            min_delta=0,
                                            patience=grid_patience)

    reducelr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                factor=0.5,
                                                patience=int(grid_patience/4), # This is being divided by 4 twice? Is that supposed to be the case?
                                                verbose=0,
                                                mode='auto',
                                                min_delta=0,
                                                cooldown=0,
                                                min_lr=0)

    t1=time.time()
    history=vae.fit(x=traingen,
                    y=None,
                    shuffle=True,
                    verbose=2,
                    epochs=int(max_epochs/4),
                    callbacks=[checkpointer,earlystop,reducelr,print_predictions],
                    validation_data=(testgen,None),
                    batch_size=batch_size)
    t2=time.time()
    vaetime=t2-t1
    print("VAE run time: "+str(vaetime)+" seconds")

    return history, vae

def create_print_pred_callback(encoder, dc, samples, batch_size, prediction_freq, saveLDpos):
    """Creates callback function using custom parameterization

    Args:
        encoder (keras network): Built keras network for encoder portion of VAE
        dc (ndarray): Derived counts
        samples (ndarray): Samples
        batch_size (int): Batch size for training
        prediction_freq (int): How often to print predictions
        saveLDpos (func): Lambda function for logging

    Returns:
        func: Keras callback function
    """
    print_predictions=keras.callbacks.LambdaCallback(
            on_epoch_end= epoch,
            logs:saveLDpos(encoder=encoder,
                            predgen=dc,
                            samples=samples,
                            batch_size=batch_size,
                            epoch=epoch,
                            frequency=prediction_freq))

    return print_predictions

def append_min_losses(history, width, depth, param_losses):
    """Gets minimum loss from history and returns to param_losses dataframe

    Args:
        history (Keras History): Keras history object for the model just trained
        width (int): Width of model (nodes per layer)
        depth (int): Depth of model (layers)
        param_losses (pd.df): Losses dataframe to append new minimum loss to

    Returns:
        pd.df: Losses from each parameterization
    """
    minloss=np.min(history.history['val_loss'])
    row=np.transpose(pd.DataFrame([width,depth,minloss]))
    row.columns=['width','depth','val_loss']
    param_losses=param_losses.append(row,ignore_index=True)
    K.clear_session() #maybe solves the gpu memory issue(???)

    return param_losses

def get_best_losses(param_losses):
    """Gets best parameters from loss values and prints

    Args:
        param_losses (dict): Dictionary of losses calculated during training

    Returns:
        int: Width of best network
        int: Depth of best network
    """
    #save tests and get min val_loss parameter set
    print(param_losses)
    param_losses.to_csv(out+"_param_grid.csv",index=False,header=True)
    bestparams=param_losses[param_losses['val_loss']==np.min(param_losses['val_loss'])]
    width=int(bestparams['width'])
    depth=int(bestparams['depth'])
    print('Best parameters:\nWidth = '+str(width)+'\nDepth = '+str(depth))

    return width, depth

def final_model_run(train_gen, test_gen, train_samples, test_samples, user_args):
    """What if you're too good for grid search? Then you run this.
    Same workflow as grid search, just leave out all the reparameterizations

    Args:
        depth_range (str): Ranges of depths to test
        width_range (str): Range of widths to test
        patience (int): Patience parameter for early stopping
        train_gen (ndarray): Training genotypes
        test_gen (ndarray): Testing genotypes
        train_samples (ndarray): Training samples
        test_samples (ndarray): Testing samples
        latent_dim (int): Number of latent dimensions
        sampling (func): Lambda function for sampling
        out (str): Output file prefix
        print_prediction_callback (func): Lambda function for keras callback

    Returns:
        keras history: History object from best model
        keras model: fitted VAE model
    """
    encoder = create_encoder(train_gen, user_args['width'], user_args['depth'], user_args['latent_dim'])
    print_prediction_callback = create_print_pred_callback(encoder, dc, samples,
                                                            user_args['batch_size'], user_args['prediction_freq'],
                                                            saveLDpos(encoder, predgen,
                                                                samples, user_args['batch_size'], 
                                                                epoch, user_args['pred_frequency']))

    vae, input_seq, output_seq, encoder = create_vae(train_gen, width, depth, user_args['latent_dim'], sampling())
    vae = add_loss_function(vae, input_seq, output_seq)

    history, vae = train_model(vae, out, grid_patience, print_prediction_callback)
    param_losses = append_min_losses(history, width, depth, param_losses)
    
    t1=time.time()

    history, vae = train_model(vae, out, patience, print_prediction_callback)
    t2 = time.time()
    print("VAE run time: "+str(vaetime)+" seconds")

    return history, vae

def grid_search(train_gen, test_gen, train_samples, test_samples, user_args):
    """Grid search through network parameterizations
    - Iterates through next set of parameters
    - Creates net
    - Trains net
    - Gets best losses and spits out optimal width and depth to use for optimal encoder

    Args:
        depth_range (str): Ranges of depths to test
        width_range (str): Range of widths to test
        patience (int): Patience parameter for early stopping
        train_gen (ndarray): Training genotypes
        test_gen (ndarray): Testing genotypes
        train_samples (ndarray): Training samples
        test_samples (ndarray): Testing samples
        latent_dim (int): Number of latent dimensions
        out (str): Output file prefix

    Returns:
        int: Best width
        int: Best depth

    TODO: Implement OOB gridsearch here, should be pretty straightfoward to swap this over to an API
    TODO: Optimize more than 2 hyperparameters, reliant on above for easy implementation

    """

    #grid search on network sizes. Getting OOM errors on 256 networks when run in succession -- GPU memory not clearing on new compile? unclear.
    #Wonder if switching to something pre-written would help? Talos, for example?
    print('Running grid search on network sizes')
    grid_patience = user_args['patience'] / 4

    #get parameter combinations (will need to rework this for >2 params)
    # OOB gridsearch will make that ^ very easy -Logan
    paramsets=[[x,y] for x in user_args['width_range'] for y in user_args['depth_range']]

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

        encoder = create_encoder(train_gen, user_args['width'], user_args['depth'], user_args['latent_dim'])
        print_prediction_callback = create_print_pred_callback(encoder, dc, samples,
                                                                user_args['batch_size'], user_args['prediction_freq'],
                                                                saveLDpos(encoder, predgen,
                                                                    samples, user_args['batch_size'], 
                                                                    epoch, user_args['pred_frequency']))

        vae, input_seq, output_seq, encoder = create_vae(train_gen, width, depth, user_args['latent_dim'], sampling())
        vae = add_loss_function(vae, input_seq, output_seq)

        history, vae = train_model(vae, out, grid_patience, print_prediction_callback)
        param_losses = append_min_losses(history, width, depth, param_losses)

    best_width, best_depth = get_best_losses(param_losses)

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

def predict_latent_coords(dc, batch_size, latent_dim, samples, traingen, width, depth):
    """Get latent space mean and var from trained encoder, write coordinates to csv

    Args:
        dc (ndarray): Derived count
        batch_size (int): Batch size for predictions
        latent_dim (int): Latent dimensions to predict for
        samples (ndarray): Samples
        traingen (ndarray): Training genotypes
        width (int): Width of net
        depth (int): Depth of net
        sampling (func): Lambda func
    """

    encoder = create_encoder(traingen, width, depth, latent_dim)
    vae = create_vae(traingen, width, depth, latent_dim, sampling(), encoder)

    #predict latent space coords for all samples from weights minimizing val loss
    vae.load_weights(out+"_weights.hdf5")
    pred=encoder.predict(dc,batch_size=batch_size) #returns [mean,sd,sample] for individual distributions in latent space
    p=pd.DataFrame()
    if latent_dim==2:
        p['mean1']=pred[0][:,0]
        p['mean2']=pred[0][:,1]
        p['sd1']=pred[1][:,0]
        p['sd2']=pred[1][:,1]
        pred=p
    else:
        pred=pd.DataFrame(pred[0])
        pred.columns=['LD'+str(x+1) for x in range(len(pred.columns))]
    pred['sampleID']=samples
    pred.to_csv(out+'_latent_coords.txt',sep='\t',index=False)

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
    #OS Side settings and random seeds
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_number

    if not seed==None:
        os.environ['PYTHONHASHSEED']=str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tensorflow.set_random_seed(seed)

    user_args = parse_arguments()

    #Get data
    dc, samples = load_genotypes(user_args['infile'], user_args['max_SNPs'])
    trainsamples, testsamples, traingen, testgen = split_training_data(dc, samples, user_args['train_prop'])

    if search_network_sizes: #Grid search
        # Should reduce the parameterization of this function, but it's handy for now
        best_width, best_depth = grid_search(train_gen, test_gen, train_samples, test_samples, user_args)                                           )
    else:
        best_width = user_args['width']
        best_depth = user_args['depth']     

    history, vae = final_model_run(train_gen, test_gen, train_samples, test_samples, user_args)
    save_training_history(history, user_args['out'])
    predict_latent_coords(dc, user_args['batch_size'], user_args['latent_dim'], 
                            samples, traingen, best_width, best_depth)

    if user_args['PCA']:
        run_PCA(dc, pcdata, PCA_scaler, n_pc_axes)
        plot_PCA(vaetime, pcatime, out)

    if user_args['plot']:
        run_plotter(out, metadata)

    if not user_args['save_weights']:
        subprocess.check_output(['rm',out+"_weights.hdf5"])

    if user_arg['save_allele_counts'] and not user_args['infile'].endswith('.popvae.hdf5'):
        save_hdf5(user_args['infile'], True, dc, samples) 
        #I have no idea where prune_LD is coming from, so setting it to default True for now



if __name__ == "__main__":
    main()

# ###debugging parameters
# os.chdir("/Users/cj/popvae/")
# infile="data/pabu/pabu_test_genotypes.vcf"
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
