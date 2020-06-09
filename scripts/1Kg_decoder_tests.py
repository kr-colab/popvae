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
from scipy import spatial

os.chdir("/Users/cj/popvae/")
infile="data/1kg/YRI_CEU_CHB.chr22.highcoverageCCDG.vcf.gz" #phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes
sample_data="data/1kg/sample_metadata.txt"
save_allele_counts=True
patience=50
batch_size=32
max_epochs=300
seed=12345
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

def filter_genotypes(gen,pos,refs=None,alts=None):
    print("genotype matrix: "+str(gen.shape))
    if not np.all(refs==None):
        #drop sites with>1 alt allele or non-ACGT ref/alt
        oneAlt=np.array([np.logical_and(x[2]=="",x[1]=="") for x in alts])
        a=np.array([x[0] in ["A","C","G","T"] for x in refs])
        b=np.array([x[0] in ["A","C","G","T"] for x in alts])
        drop=np.logical_or(~a,~b)
        drop=np.logical_or(~oneAlt,drop)
        pos=pos[~drop]
        gen=gen[~drop,:,:]
        print("dropped "+str(np.sum(drop))+" non-biallelic sites")
        print(gen.shape)

    ac_all=gen.count_alleles() #count of alleles per snp
    ac=gen.to_allele_counts() #count of alleles per snp per individual

    biallel=ac_all.is_biallelic()
    dc_all=ac_all[biallel,1] #derived alleles per snp
    dc=np.array(ac[biallel,:,1],dtype="int_") #derived alleles per individual
    ac=ac[biallel,:,:]
    ac_all=ac_all[biallel,:]
    pos=pos[biallel]
    missingness=gen[biallel,:,:].is_missing()
    print("dropped "+str(np.sum(~biallel))+" invariant sites")
    print(dc.shape)

    ninds=np.array([np.sum(x) for x in ~missingness])
    singletons=np.array([x<=2 for x in dc_all])
    dc_all=dc_all[~singletons]
    dc=dc[~singletons,:]
    ac=ac[~singletons,:,:]
    ac_all=ac_all[~singletons,:]
    ninds=ninds[~singletons]
    missingness=missingness[~singletons,:]
    pos=pos[~singletons]
    print("dropped "+str(np.sum(singletons))+" singletons")
    print(dc.shape)

    print("filling missing data with rbinom(2,derived_allele_frequency)")
    af=np.array([dc_all[x]/(ninds[x]*2) for x in range(dc_all.shape[0])]) #get allele frequencies for missing data imputation
    for i in tqdm(range(np.shape(dc)[1])):
        indmiss=missingness[:,i]
        dc[indmiss,i]=np.random.binomial(2,af[indmiss])

    print("genotype matrix shape: "+str(dc.shape))

    return(dc_all,dc,ac_all,ac,pos)


#get accessibility mask
mask=[]
with open("data/1kg/chr22.strictMask.fasta") as f:
    next(f)
    for line in f:
        a=line
        a=re.sub("\n","",a)
        for position in a:
            mask.append(position)
mask=np.array(mask)
keep=np.argwhere(mask=="P")

print("reading VCF")
infile="data/1kg/YRI_CEU_CHB.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
#infile="data/1kg/YRI_CEU_CHB.chr22.highcoverageCCDG.vcf.gz" #even more SNPs in the high coverage resequencing data, both before and after masking
vcf=allel.read_vcf(infile,log=sys.stderr)
gen=allel.GenotypeArray(vcf['calldata/GT'])
samples=vcf['samples']
pos=vcf['variants/POS']
refs=vcf['variants/REF']
alts=vcf['variants/ALT']
m1=np.isin(pos,keep)
gen=gen[m1,:,:]
pos=pos[m1]
refs=refs[m1]
alts=alts[m1]
dc_all,dc,ac_all,ac,pos=filter_genotypes(gen,pos,refs,alts)

dc=np.transpose(dc)
dc=dc*0.5 #0=homozygous reference, 0.5=heterozygous, 1=homozygous alternate

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

def saveLDpos(encoder,predgen,samples,batch_size,epoch,frequency):
    if(epoch%frequency==0):
        pred=encoder.predict(predgen,batch_size=batch_size)[0]
        pred=pd.DataFrame(pred)
        pred['sampleID']=samples
        pred['epoch']=epoch
        if(epoch==0):
            pred.to_csv(out+"_training_preds.txt",sep='\t',index=False,mode='w',header=True)
        else:
            pred.to_csv(out+"_training_preds.txt",sep='\t',index=False,mode='a',header=False)

print_predictions=keras.callbacks.LambdaCallback(
         on_epoch_end=lambda epoch,
         logs:saveLDpos(encoder=encoder,
                        predgen=dc,
                        samples=samples,
                        batch_size=batch_size,
                        epoch=epoch,
                        frequency=prediction_freq))

#training
history=vae.fit(x=traingen,
                y=None,
                shuffle=True,
                epochs=max_epochs,
                callbacks=[checkpointer,earlystop,reducelr,print_predictions],
                validation_data=(testgen,None),
                batch_size=batch_size)

#load best weights
vae.load_weights(out+"_weights.hdf5")
#vae.load_weights("/Users/cj/popvae/out/1kg/1kg_3pop_weights.hdf5")

######################### plot VAE latent space ##############################
sample_data=pd.read_csv("data/1kg/sample_metadata.txt",sep="\t")
pred=encoder.predict(dc)[0]
pred=pd.DataFrame(pred)
pred.columns=['LD1','LD2']
pred['sampleID']=samples
pred=pred.merge(sample_data,on="sampleID")
plt.scatter(pred['LD1'],pred['LD2'],c=pd.factorize(pred['pop'])[0])

##################### get decoder outputs for sample latent means #################
pgen=decoder.predict(encoder.predict(dc)[0]) #encoder.predict() returns [mean,sd,sample] for normal distributions describing sample locations in latent space, so [0] is fixed but [2] is stochastic given a set of weights.

#binning with binomial draws
def binomialBinGenotypes(pgen):
    out=np.copy(pgen)
    for i in range(out.shape[0]):
        out[i,:]=np.random.binomial(2,out[i,:])
    return out

bingen=binomialBinGenotypes(pgen)

############################ run OOA simulation and prep allele counts #####################
import msprime as msp
import stdpopsim
print("simulating")
species = stdpopsim.get_species("HomSap")
contig = species.get_contig("chr22",genetic_map="HapMapII_GRCh37")
model = species.get_demographic_model('OutOfAfrica_3G09') #similar results with OutOfAfrica_3G09 and OutOfAfricaArchaicAdmixture_5R19
simsamples = model.get_samples(100, 100, 100)
engine = stdpopsim.get_engine('msprime')
sim = engine.simulate(model,contig,simsamples,seed=12345)
sim_gen=allel.HaplotypeArray(sim.genotype_matrix()).to_genotypes(ploidy=2)
sim_pos=np.array([s.position for s in sim.sites()],dtype="int32")

m2=np.isin(sim_pos,keep)
sim_gen=sim_gen[m2,:,:]
sim_pos=sim_pos[m2]

sim_dc_all,sim_dc,sim_ac_all,sim_ac,sim_pos=filter_genotypes(sim_gen,sim_pos)


##################### comparing PCA of real vs generated genotypes #######################
realpca=allel.pca(np.transpose(dc)*2,scaler=None,n_components=2) #*2 here to rescale real genotypes back to 0/1/2 to match binomial(2,...) used to bin genotypes.
genpca=allel.pca(np.transpose(bingen),scaler=None,n_components=2)
simpca=allel.pca(sim_dc,scaler=None,n_components=6)
sampledata=pd.read_csv("data/1kg/sample_metadata.txt",sep="\t")
df=pd.DataFrame(np.hstack((realpca[0],genpca[0])))
df.columns=['realPC1','realPC2','genPC1','genPC2']
df['sampleID']=samples
df=df.merge(sampledata,on='sampleID')
df.to_csv('out/1kg/1kg_decoder_PCA.csv',sep=",",index=False)

simdf=pd.DataFrame(simpca[0])
simdf.columns=['PC1','PC2','PC3','PC4','PC5','PC6']
simdf['pop']=np.concatenate([np.repeat("YRI",50),np.repeat("CEU",50),np.repeat("CHB",50)])
simdf.to_csv('out/1kg/1kg_sim_PCA.txt',sep="\t",index=False)

fig,[ax1,ax2,ax3]=plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True)
fig.set_figwidth(6.5)
fig.set_figheight(3)
ax1.scatter(df['realPC1'],df['realPC2'],c=pd.factorize(df['pop'])[0])
ax1.set_title("real")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax2.scatter(df['genPC1'],df['genPC2'],c=pd.factorize(df['pop'])[0])
ax2.set_title("generated")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
ax3.scatter(simdf['PC1'],simdf['PC2'],c=pd.factorize(simdf['pop'])[0])
ax3.set_title("simulated")
ax3.set_xlabel("PC1")
ax3.set_ylabel("PC2")
fig.tight_layout()
#fig.savefig('fig/PCA_decoder_comp_mpl.pdf',bbox_inches='tight')

###################### pi ####################### (need to fix this to account for masking)
# pi=allel.sequence_diversity(pos,ac_all)
# pi2=allel.sequence_diversity(sim_pos,sim_ac_all)
# gen_ac_all=np.apply_along_axis(sum,0,bingen)
# gen_ac_all=np.array(gen_ac_all,dtype="i")
# tmp=np.array([300-x for x in gen_ac_all],dtype="i")
# gen_ac_all=allel.AlleleCountsArray(np.transpose(np.vstack((tmp,gen_ac_all))))
# pi3=allel.sequence_diversity(pos,gen_ac_all)
#
# print("real pi:"+str(pi))
# print("simulated pi:"+str(pi2))
# print("VAE decoder pi:"+str(pi3))

########################### site frequency spectrum #############################
realYRI=dc*2
realYRI=realYRI[pred['pop']=="YRI"]
YRI_ac_all=np.apply_along_axis(sum,0,realYRI)
YRI_ac_all=np.array(YRI_ac_all,dtype="i")
tmp=np.array([100-x for x in YRI_ac_all],dtype="i")
YRI_ac_all=allel.AlleleCountsArray(np.transpose(np.vstack((tmp,YRI_ac_all))))

simYRI=np.transpose(sim_dc)
simYRI=simYRI[simdf['pop']=="YRI"]
simYRI_ac_all=np.apply_along_axis(sum,0,simYRI)
simYRI_ac_all=np.array(simYRI_ac_all,dtype="i")
tmp=np.array([100-x for x in simYRI_ac_all],dtype="i")
simYRI_ac_all=allel.AlleleCountsArray(np.transpose(np.vstack((tmp,simYRI_ac_all))))

genYRI=bingen
genYRI=genYRI[pred['pop']=="YRI"]
genYRI_ac_all=np.apply_along_axis(sum,0,genYRI)
genYRI_ac_all=np.array(genYRI_ac_all,dtype="i")
tmp=np.array([100-x for x in genYRI_ac_all],dtype="i")
genYRI_ac_all=allel.AlleleCountsArray(np.transpose(np.vstack((tmp,genYRI_ac_all))))

realsfs=allel.sfs(YRI_ac_all[:,1])
gensfs=allel.sfs(genYRI_ac_all[:,1])
simsfs=allel.sfs(simYRI_ac_all[:,1])
sfs=pd.DataFrame()
sfs['real']=realsfs
sfs['VAE']=gensfs
sfs['simulation']=simsfs
sfs['bin']=np.arange(0,len(realsfs))
sfs.to_csv("out/1kg/1kg_sfs.csv",index=False)


################### LD decay ####################
#get LD and pairwise distance for a subset of 1000 SNPs
np.random.seed(12345)
mask=np.logical_and(pos>3e7,pos<3.1e7)
dc2=dc[:,mask]
dc2=dc2[pred['pop']=="YRI",:]
bingen2=bingen[:,mask]
bingen2=bingen2[pred['pop']=='YRI',:]
pos2=pos[mask]

#calculate pairwise LD matrices
LDr=allel.rogers_huff_r(np.transpose(dc2))
LDg=allel.rogers_huff_r(np.transpose(bingen2))
LDr=spatial.distance.squareform(LDr)
LDg=spatial.distance.squareform(LDg)

#get bp distances
dists=[x-y for x in pos2 for y in pos2]
LDr2=np.concatenate(LDr)
LDr2=np.array(LDr2,dtype="float64")
LDr2=LDr2**2

LDg2=np.concatenate(LDg)
LDg2=np.array(LDg2,dtype="float64")
LDg2=LDg2**2

#simulation LD
sim_mask=np.logical_and(sim_pos>3e7,sim_pos<3.1e7)
sim_dc2=np.transpose(sim_dc)[:,sim_mask]
sim_dc2=sim_dc2[simdf['pop']=="YRI",:]
sim_pos2=sim_pos[sim_mask]
sim_LDr=allel.rogers_huff_r(np.transpose(sim_dc2))
sim_LDr=spatial.distance.squareform(sim_LDr)

#get bp distances
sim_dists=[x-y for x in sim_pos2 for y in sim_pos2]
sim_LDr2=np.concatenate(sim_LDr)
sim_LDr2=np.array(sim_LDr2,dtype="float64")
sim_LDr2=sim_LDr2**2

#3-panel plot
fig,[ax1,ax2,ax3]=plt.subplots(1,3,sharex=True)
ax1.scatter(np.abs(dists),LDr2,s=5)
ax1.set_title("real")
ax1.set_ylabel(r"$R^2$")
ax2.scatter(np.abs(dists),LDg2,s=5)
ax2.set_title("VAE generated")
ax2.set_xlabel("Distance (bp)")
ax3.scatter(np.abs(sim_dists),sim_LDr2,s=5)
ax3.set_title("coalescent simulation")
fig.set_figwidth(6.5)
fig.set_figheight(3)
fig.tight_layout()

#export data for prettier R versions
out_real=pd.DataFrame()
out_real['dist']=np.abs(dists)
out_real['LD']=LDr2
out_real.to_csv("/Users/cj/popvae/out/1kg/1kg_LD_decay_chr22:2e7-2.2e7_real.csv",index=False)

out_gen=pd.DataFrame()
out_gen['dist']=np.abs(dists)
out_gen['LD']=LDg2
out_gen.to_csv("/Users/cj/popvae/out/1kg/1kg_LD_decay_chr22:2e7-2.2e7_gen.csv",index=False)

out_sim=pd.DataFrame()
out_sim['dist']=np.abs(sim_dists)
out_sim['LD']=sim_LDr2
out_sim.to_csv("/Users/cj/popvae/out/1kg/1kg_LD_decay_chr22:2e7-2.2e7_sim.csv",index=False)
