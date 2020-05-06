import keras, numpy as np, os, allel, pandas as pd, time
import zarr, subprocess, h5py, re, sys, os, argparse
from matplotlib import pyplot as plt
from tqdm import tqdm
from keras.models import Sequential
from keras import layers
from keras.layers.core import Lambda
from keras import backend as K
from keras.models import Model
import tensorflow

os.chdir("/home/cbattey2/popvae/")
infile="data/hgdp/hgdp_chr1_1e5snps_seed42.popvae.hdf5"
sample_data="data/hgdp/hgdp_sample_data.txt"
save_allele_counts=True
patience=20
batch_size=32
max_epochs=300
seed=54321
save_weights=False
train_prop=0.9
gpu_number='0'
prediction_freq=2
out="out/decoder_test"
latent_dim=2
max_SNPs=None
PCA=True
nlayers=6
width=128
parallel=False
prune_iter=1
prune_size=500
PCA_scaler="Patterson"

if not parallel:
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_number

if not seed==None:
    np.random.seed(seed)
    tensorflow.set_random_seed(seed)

h5=h5py.File(infile,'r')
dc=np.array(h5['derived_counts'])
samples=np.array(h5['samples'])
h5.close()

if not max_SNPs==None:
    print("subsetting to "+str(max_SNPs)+" SNPs")
    dc=dc[:,np.random.choice(range(dc.shape[1]),max_SNPs,replace=False)]

print("running train/test splits")
ninds=dc.shape[0]
if train_prop==1:
    train=np.random.choice(range(ninds),int(train_prop*ninds),replace=False)
    test=train
    traingen=dc[train,:]
    testgen=dc[test,:]
    trainsamples=samples[train]
    testsamples=samples[test]
else:
    train=np.random.choice(range(ninds),int(train_prop*ninds),replace=False)
    test=np.array([x for x in range(ninds) if x not in train])
    traingen=dc[train,:]
    testgen=dc[test,:]
    trainsamples=samples[train]
    testsamples=samples[test]

#################### load network #####################
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

#encoder
input_seq = keras.Input(shape=(traingen.shape[1],))
x=layers.Dense(width,activation="elu")(input_seq)
for i in range(nlayers-1):
    x=layers.Dense(width,activation="elu")(x)
z_mean=layers.Dense(latent_dim)(x)
z_log_var=layers.Dense(latent_dim)(x)
z = layers.Lambda(sampling,output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder=Model(input_seq,[z_mean,z_log_var,z],name='encoder')

#decoder
decoder_input=layers.Input(shape=(latent_dim,),name='z_sampling')
x=layers.Dense(width,activation="linear")(decoder_input)#was elu
for i in range(nlayers-1):
    x=layers.Dense(width,activation="elu")(x)
output=layers.Dense(traingen.shape[1],activation="sigmoid")(x) #hard sigmoid seems natural here but appears to lead to more left-skewed decoder outputs.
decoder=Model(decoder_input,output,name='decoder')

#end-to-end vae
output_seq = decoder(encoder(input_seq)[2])
vae = Model(input_seq, output_seq, name='vae')

#add loss function
reconstruction_loss = keras.losses.binary_crossentropy(input_seq,output_seq)
reconstruction_loss *= traingen.shape[1]
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

vae.compile(optimizer='adam')

#callbacks
checkpointer=keras.callbacks.ModelCheckpoint(
              filepath=out+"_weights.hdf5",
              verbose=1,
              save_best_only=True,
              monitor="val_loss",
              period=1)

earlystop=keras.callbacks.EarlyStopping(monitor="val_loss",
                                        min_delta=0,
                                        patience=patience)

reducelr=keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.5,
                                           patience=int(patience/4),
                                           verbose=1,
                                           mode='auto',
                                           min_delta=0,
                                           cooldown=0,
                                           min_lr=0)

#training
t1=time.time()
history=vae.fit(x=traingen,
                y=None,
                shuffle=True,
                epochs=max_epochs,
                callbacks=[checkpointer,earlystop,reducelr],
                validation_data=(testgen,None),
                batch_size=batch_size)
t2=time.time()
vaetime=t2-t1
print("VAE run time: "+str(vaetime)+" seconds")

#load best weights
vae.load_weights(out+"_weights.hdf5")

#plot VAE latent space
sample_data=pd.read_csv("data/hgdp/hgdp_sample_data.txt",sep="\t")
pred=encoder.predict(dc)[0]
pred=pd.DataFrame(pred)
pred.columns=['LD1','LD2']
pred['sampleID']=samples
pred=pred.merge(sample_data,on="sampleID")
plt.scatter(pred['LD1'],pred['LD2'],c=pd.factorize(pred['region'])[0])

##################### decoder tests ######################
#get generated genotypes for all real samples by sampling from the latent space
pgen=decoder.predict(encoder.predict(dc)[2]) #encoder.predict() returns [mean,sd,sample] for normal distributions describing sample locations in latent space, so [0] is fixed but [2] is stochastic given a set of weights.

#binning with binomial draws
def binomialBinGenotypes(pgen):
    out=np.copy(pgen)
    for i in range(out.shape[0]):
        out[i,:]=np.random.binomial(2,out[i,:])
    return out

bingen=binomialBinGenotypes(pgen)

#comparing PCA of real vs generated genotypes
realpca=allel.pca(np.transpose(dc)*2,scaler="Patterson",n_components=2) #*2 here to rescale genotypes to 0/1/2 to match binomial(2,...) used to bin genotypes.
#genpca=allel.pca(np.transpose(bingen),scaler=None,n_components=2)[0] #run a separate PCA
genpca=realpca[1].transform(np.transpose(bingen)) #project generated coordinates into the "real" PC space
sampledata=pd.read_csv("data/hgdp/hgdp_sample_data.txt",sep="\t")
df=pd.DataFrame(np.hstack((realpca[0],genpca)))
df.columns=['realPC1','realPC2','genPC1','genPC2']
df['sampleID']=samples
df=df.merge(sampledata,on='sampleID')
df.to_csv('pca_decoder_test.csv',sep=",",index=False)

fig,[ax1,ax2]=plt.subplots(nrows=1,ncols=2)
fig.set_figwidth(6)
fig.set_figheight(2.75)
ax1.scatter(df['realPC1'],df['realPC2'],c=pd.factorize(df['region'])[0])
ax1.set_title("real")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax2.scatter(df['genPC1'],df['genPC2'],c=pd.factorize(df['region'])[0])
ax2.set_title("generated")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
fig.tight_layout()
fig.savefig('fig/PCA_decoder_comp_mpl.pdf',bbox_inches='tight')


#convert generated genotype matrix to ped format for running Admixture
#for (???) reasons the ped won't run in admixture directly, but does run after conversion to binary via `plink --make-bed --file <<outpath.ped>>`
def genotypes_to_ped(bingen,sample_IDs,outpath=None,):
    genotypes=np.empty((bingen.shape[0],bingen.shape[1]*2))
    for i in tqdm(range(bingen.shape[0])):
        dat=np.empty((bingen.shape[1],2))
        for j in range(bingen.shape[1]):
            if bingen[i,j]==0:
                dat[j]=[1,1]
            elif bingen[i,j]==1:
                dat[j]=[1,2]
            elif bingen[i,j]==2:
                dat[j]=[2,2]
        dat=np.concatenate(dat)
        genotypes[i,:]=dat
    startfields=pd.DataFrame(np.transpose([np.arange(0,bingen.shape[0],1), #family ID
                                           sample_IDs,                     #individual ID
                                           np.arange(0,bingen.shape[0],1),     #paternal ID
                                           np.arange(0,bingen.shape[0],1),     #maternal ID
                                           np.arange(0,bingen.shape[0],1),     #sex
                                           np.arange(0,bingen.shape[0],1)]))   #phenotype
    pedout=pd.concat([startfields,pd.DataFrame(genotypes)],axis=1)
    mapout=pd.DataFrame(np.transpose([np.repeat(1,bingen.shape[1]), #chromosome (dummy vars here since admixture requires this file (!?) but doesn't use position information)
                                      np.arange(0,bingen.shape[1],1), #snp identifier
                                      np.repeat(0,bingen.shape[1]), #genetic distance
                                      np.arange(0,bingen.shape[1],1)])) #base pair position
    if not outpath==None:
        print("saving .ped and .map files to "+outpath)
        pedout.to_csv(outpath+".ped",sep=" ",index=False,header=False)
        mapout.to_csv(outpath+".map",sep=" ",index=False,header=False)
    return pedout,mapout

pedout,mapout=genotypes_to_ped(bingen,outpath="/home/cbattey2/popvae/admixture/hgdp_genotypes_1e5snps_generated",sample_IDs=df['sampleID'])
pedout,mapout=genotypes_to_ped(dc*2,outpath="/home/cbattey2/popvae/admixture/hgdp_genotypes_1e5snps",sample_IDs=df['sampleID'])

#convert to bed files
print("converting to .bed")
subprocess.check_output('cd /home/cbattey2/popvae/admixture/; \
                         \
                         plink --noweb --make-bed \
                         --file hgdp_genotypes_1e5snps_generated \
                         --out hgdp_genotypes_1e5snps_generated; \
                         \
                         plink --noweb --make-bed \
                         --file hgdp_genotypes_1e5snps \
                         --out hgdp_genotypes_1e5snps;',shell=True)

#run Admixture
print("running Admixture")
subprocess.check_output('cd /home/cbattey2/popvae/admixture/; \
                         admixture hgdp_genotypes_1e5snps_generated.bed -s 54321 7 -j10; \
                         admixture hgdp_genotypes_1e5snps.bed -s 54321 7 -j10;',
                        shell=True,stderr=True)



#generate a set of genotypes from latent positions characteristic of the population X
pops=np.unique(sample_data['population'])
for pop in pops:
    pred=encoder.predict(dc)[0]
    pred=pd.DataFrame(pred)
    pred.columns=['LD1','LD2']
    pred['sampleID']=samples
    pred=pred.merge(sample_data,on="sampleID")

    sanLD1=np.mean(pred[pred['population']==pop]['LD1'])
    sanLD1sd=np.std(pred[pred['population']==pop]['LD1'])
    sanLD2=np.mean(pred[pred['population']==pop]['LD2'])
    sanLD2sd=np.std(pred[pred['population']==pop]['LD2'])

    newLD1=np.random.normal(sanLD1,sanLD1sd,10)
    newLD2=np.random.normal(sanLD2,sanLD2sd,10)
    newLD=np.array([newLD1,newLD2])

    pgen=decoder.predict(np.transpose(newLD))
    bingen=binomialBinGenotypes(pgen)

    genpca=realpca[1].transform(np.transpose(bingen)) #project generated coordinates into the "real" PC space
    genpca=pd.DataFrame(genpca)

    fig,ax=plt.subplots(1,1)
    ax.scatter(df['realPC1'],df['realPC2'],c=pd.factorize(df['region'])[0])
    ax.scatter(genpca[0],genpca[1],c='red')
    fig.tight_layout()
    fig.set_figheight(2.5)
    fig.set_figwidth(3)
    fig.savefig("fig/decoder_population_PCAs/"+pop+".pdf")









#
