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

os.chdir("/Users/cj/popvae/")
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
out="out/test"
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

#plot VAE latent space
sample_data=pd.read_csv("/Users/cj/popvae/data/hgdp/hgdp_sample_data.txt",sep="\t")
pred=encoder.predict(dc)[0]
pred=pd.DataFrame(pred)
pred.columns=['LD1','LD2']
pred['sampleID']=samples
pred=pred.merge(sample_data,on="sampleID")
plt.scatter(pred['LD1'],pred['LD2'],c=pd.factorize(pred['region'])[0])

##################### decoder tests ######################
#get generated genotypes for all real samples
vae.load_weights(out+"_weights.hdf5")
pgen=decoder.predict(encoder.predict(dc)[0])
pgen.shape
dc.shape

def binGenotypes(pgen,a,b):
    out=np.copy(pgen)
    out[out<a]=0
    out[(out>=a) & (out<=b)]=0.5 #required syntax on these multiple booleans sucks.
    out[out>b]=1
    return out



def gridSearchCutoffs(pgen,dc,return_all_loss=False,arange=[0,0.5],brange=[0.5,1],step=0.05):
    loss=[];params=[]
    asearch=np.arange(arange[0],arange[1],step)
    bsearch=np.arange(brange[0],brange[1],step)
    for a in tqdm(asearch):
        for b in bsearch:
            #print(str(a)+" "+str(b))
            bingen=binGenotypes(pgen,a,b)
            loss.append(sum(sum(abs(bingen-dc))))
            params.append((a,b))
    opt_params=params[np.argmin(loss)]
    if return_all_loss:
        return opt_params,loss,params
    else:
        return opt_params

opt_params=gridSearchCutoffs(pgen,dc,step=0.05)

bingen=binGenotypes(pgen,opt_params[0],opt_params[1])

#comparing PCA of real vs generated genotypes
realpca=allel.pca(np.transpose(dc),scaler=None,n_components=2)
genpca=allel.pca(np.transpose(bingen),scaler=None,n_components=2)[0] #run a separate PCA something causing an issue with the Patterson scaler (invariant sites in generated genotypes?)
genpca=realpca[1].transform(np.transpose(bingen)) #project generated coordinates into the "real" PC space
sampledata=pd.read_csv("data/hgdp/hgdp_sample_data.txt",sep="\t")
df=pd.DataFrame(np.hstack((realpca[0],genpca)))
df.columns=['realPC1','realPC2','genPC1','genPC2']
df['sampleID']=samples
df=df.merge(sampledata,on='sampleID')
df.to_csv('/Users/cj/Desktop/pca_decoder_test.csv',sep=",",index=False)

#compare real and generated genotypes in PC space
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
fig.savefig('/Users/cj/popvae/fig/PCA_decoder_comp.pdf',bbox_inches='tight')

#convert generated genotype matrix to ped format for running Admixture
#for (???) reasons the ped won't run in admixture directly, but does run after conversion to binary via `plink --make-bed --file <<outpath.ped>>`
def genotypes_to_ped(bingen,sample_IDs,outpath=None,):
    genotypes=np.empty((bingen.shape[0],bingen.shape[1]*2))
    for i in tqdm(range(bingen.shape[0])):
        dat=np.empty((bingen.shape[1],2))
        for j in range(bingen.shape[1]):
            if bingen[i,j]==0:
                dat[j]=[1,1]
            elif bingen[i,j]==0.5:
                dat[j]=[1,2]
            elif bingen[i,j]==1:
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

pedout,mapout=genotypes_to_ped(bingen,outpath="/Users/cj/popvae/admixture/hgdp_genotypes_1e5snps_generated",sample_IDs=df['sampleID'])
pedout,mapout=genotypes_to_ped(dc,outpath="/Users/cj/popvae/admixture/hgdp_genotypes_1e5snps",sample_IDs=df['sampleID'])

#convert to bed files
print("converting to .bed")
subprocess.check_output('cd /Users/cj/popvae/admixture/; \
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
subprocess.check_output('cd /Users/cj/popvae/admixture/; \
                         admixture hgdp_genotypes_1e5snps_generated.bed -s 54321 7 -j10; \
                         admixture hgdp_genotypes_1e5snps.bed -s 54321 7 -j10;',
                        shell=True,stderr=True)










#
