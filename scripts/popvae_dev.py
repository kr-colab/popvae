#estimating sample locations from genotype matrices
import allel, re, os, keras, matplotlib, sys
import numpy as np, pandas as pd, tensorflow as tf
from scipy import spatial
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
import zarr
import numcodecs
from sklearn import preprocessing
import time, gnuplotlib as gp

basedir="/Users/cj/popvae"
np.random.seed(42)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.chdir(basedir)

patience=20
nsnps=10000
prediction_freq=2
max_epochs=20

#replace missing sites with binomial(2,mean_allele_frequency)
def replace_md(genotypes):
    print("filling missing data")
    dc=genotypes.count_alleles()[:,1]
    ac=genotypes.to_allele_counts()[:,:,1]
    missingness=genotypes.is_missing()
    ninds=np.array([np.sum(x) for x in ~missingness])
    af=np.array([dc[x]/(2*ninds[x]) for x in range(len(ninds))])
    for i in tqdm(range(np.shape(ac)[0])):
        for j in range(np.shape(ac)[1]):
            if(missingness[i,j]):
                ac[i,j]=np.random.binomial(2,af[i]
    return ac

##### read in VCF
vcf=allel.read_vcf("data/hgdp/hgdp_wgs.20190516.full.chr1.0-5e6.vcf.gz",log=sys.stderr)
genotypes=allel.GenotypeArray(vcf['calldata/GT'])
print(genotypes.shape)
samples=vcf['samples']
#positions=vcf['variants/POS']
#pos=np.array([x/np.max(positions) for x in positions])

##### read in zarr
# callset = zarr.open_group("data/ag1000g/ag1000g2L_1e6_to_5e6.zarr", mode='r')
# gt = callset['calldata/GT']
# genotypes = allel.GenotypeArray(gt[:])
# samples = callset['samples'][:]

derived_counts=genotypes.count_alleles()[:,1]
ac_filter=[x >= 2 for x in derived_counts] #drop SNPs with minor allele < min_mac
genotypes=genotypes[ac_filter,:,:]
if not nsnps==None:
    print("subsetting to "+str(nsnps)+" SNPs")
    genotypes=genotypes[np.random.choice(range(genotypes.shape[0]),nsnps,replace=False),:,:]

ac=replace_md(genotypes)


#load and sort sample data
sample_data=pd.read_csv("data/hgdp/hgdp_sample_data.txt",sep="\t")
sample_data['sampleID2']=sample_data['sampleID']
sample_data.set_index('sampleID',inplace=True)
sample_data=sample_data.reindex(np.array(samples)) #sort loc table so samples are in same order as vcf samples
if not all([sample_data['sampleID2'][x]==samples[x] for x in range(len(samples))]): #check that all sample names are present
    print("sample ordering failed! Check that sample IDs match the VCF.")
    sys.exit()
locs=np.array(sample_data[["longitude","latitude"]])

#normalize coordinates and genotypes
meanlong=np.nanmean(locs[:,0])
sdlong=np.nanstd(locs[:,0])
meanlat=np.nanmean(locs[:,1])
sdlat=np.nanstd(locs[:,1])
locs=np.array([[(x[0]-meanlong)/sdlong,(x[1]-meanlat)/sdlat] for x in locs])

ac_norm=preprocessing.normalize(ac,'l2')

#define networks
from keras.models import Sequential
from keras import layers
from keras.layers.core import Lambda
from keras import backend as K
from keras.models import Model
import keras

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 2),
                              mean=0., stddev=1.)
    return z_mean + K.exp(z_log_var) * epsilon

class CustomVariationalLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x) #unnecessary but maybe later
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean( #TODO figure out this constant...
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x


input_seq = keras.Input(shape=(ac_norm.shape[0],))
x=layers.Dense(64,activation="elu")(input_seq)
x=layers.Dense(64,activation="elu")(x)
x=layers.Dense(64,activation="elu")(x)
x=layers.Dense(64,activation="elu")(x)
x=layers.Dense(64,activation="elu")(x)
x=layers.Dense(64,activation="elu")(x)
z_mean=layers.Dense(2)(x)
z_log_var=layers.Dense(2)(x)
z = layers.Lambda(sampling)([z_mean, z_log_var])
decoder_input=layers.Input(K.int_shape(z)[1:])
x=layers.Dense(64,activation="elu")(decoder_input)
x=layers.Dense(64,activation="elu")(x)
x=layers.Dense(64,activation="elu")(x)
x=layers.Dense(64,activation="elu")(x)
x=layers.Dense(64,activation="elu")(x)
x=layers.Dense(64,activation="elu")(x)
x=layers.Dense(ac_norm.shape[0],activation="elu")(x)
decoder=Model(decoder_input,x)
z_decoded=decoder(z)
y=CustomVariationalLayer()([input_seq,z_decoded])
vae=Model(input_seq,y)
vae.compile(optimizer='Adam',loss=None)
encoder=Model(input_seq,z_mean)

#callbacks
checkpointer=keras.callbacks.ModelCheckpoint(
              filepath="vae_weights.hdf5",
              verbose=1,
              save_best_only=True,
              monitor="loss",
              period=1)
earlystop=keras.callbacks.EarlyStopping(monitor="loss",
                                        min_delta=0,
                                        patience=patience)
reducelr=keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                           factor=0.5,
                                           patience=int(patience/4),
                                           verbose=1,
                                           mode='auto',
                                           min_delta=0,
                                           cooldown=0,
                                           min_lr=0)
def saveLDpos(encoder,data,batch_size,epoch,frequency):
    if(epoch%frequency==0):
        pred=encoder.predict(data,batch_size=256)
        pred=pd.DataFrame(pred)
        pred['sampleID']=samples
        pred.to_csv(os.path.join(basedir,"training_preds/"+str(epoch)+".txt"),sep='\t',index=False)

print_predictions=keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch,logs: saveLDpos(encoder,np.transpose(ac_norm),256,epoch,prediction_freq))

history=vae.fit(x=np.transpose(ac_norm),
        y=None,
        shuffle=True,
        epochs=max_epochs,
        callbacks=[checkpointer,earlystop,reducelr,print_predictions],
        batch_size=256)

h=pd.DataFrame(history.history)
h.to_csv("vae_history.txt",sep="\t")

gp.plot(np.array(history.history['loss'][3:]),
                unset='grid',
                terminal='dumb 60 20',
                title='Validation Loss by Epoch')

vae.load_weights("vae_weights.hdf5")
pred=encoder.predict(np.transpose(ac),batch_size=32)
pred=pd.DataFrame(pred)
pred['sampleID']=samples
pred.to_csv('vaetest.txt',sep='\t',index=False)

#history plot
from matplotlib import pyplot as plt
plt.plot(history.history['loss'])
plt.show()

#VAE plots
pred=pd.DataFrame(pred)
pred.columns=['x','y','sampleID']
pred['sampleID']=list(sample_data['sampleID2'])
# pred['m_s']=list(sample_data['m_s'])
#pred['population']=list(sample_data['population'])
#pred['longitude']=list(sample_data['longitude'])
#pred['latitude']=list(sample_data['latitude'])
import seaborn as sns
sns.set(rc={'figure.figsize':(8,6)})
p=sns.scatterplot(x="x", y="y", data=pred,markers='.',palette="RdYlBu")
lgd=p.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#PCA
pca=allel.pca(ac,scaler=None,n_components=2)
pca=pd.DataFrame(pca[0])
pca.columns=['PC1','PC2']
pca['sampleID']=samples
pca.to_csv("vaetest_pca.txt",index=False)
# pca['sampleID']=list(sample_data['sampleID2'])
# pca['m_s']=list(sample_data['m_s'])
# pca['population']=list(sample_data['population'])
# pca['longitude']=list(sample_data['longitude'])
# pca['latitude']=list(sample_data['latitude'])
# import seaborn as sns
# sns.set(rc={'figure.figsize':(8,6)})
# p=sns.scatterplot(x="PC1", y="PC2", data=pca,hue='population',
# markers='.',palette="RdYlBu")
# p.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
