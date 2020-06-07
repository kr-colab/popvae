#input VCFs prepped from 1000genomes phase 3 with:

#cd data/1kg
#bcftools view -S YRI_CEU_CHB_sample_IDs.txt -o YRI_CEU_CHB.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz ALL.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz
#bgzip YRI_CEU_CHB.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf

#where YRI_CEU_CHB_sample_IDs.txt is a list of 50 individual IDs per population sampled from the integrated_call_samples_v3.20130502.ALL.panel

import allel, numpy as np, pandas as pd, re, sys, os, stdpopsim
np.random.seed(12345)
os.chdir("/Users/cj/popvae/")
#os.chdir("/home/cbattey2/popvae/") #on sesame

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
    ac_all=ac_all[~singletons,:]
    ac=ac[~singletons,:,:]
    ninds=ninds[~singletons]
    missingness=missingness[~singletons,:]
    pos=pos[~singletons]
    print("dropped "+str(np.sum(singletons))+" singletons")
    print(dc.shape)

    return(dc_all,dc,ac_all,ac,pos)

print("reading VCF")
infile="data/1kg/YRI_CEU_CHB.chr22.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz"
#infile="data/1kg/YRI_CEU_CHB.chr22.highcoverageCCDG.vcf.gz" #even more SNPs in the high coverage resequencing data, both before and after masking
vcf=allel.read_vcf(infile,log=sys.stderr)
gen=allel.GenotypeArray(vcf['calldata/GT'])
samples=vcf['samples']
pos=vcf['variants/POS']
refs=vcf['variants/REF']
alts=vcf['variants/ALT']
dc_all,dc,ac_all,ac,pos=filter_genotypes(gen,pos,refs,alts)

print("simulating")
species = stdpopsim.get_species("HomSap")
contig = species.get_contig("chr22",genetic_map="HapMapII_GRCh37")
model = species.get_demographic_model('OutOfAfricaArchaicAdmixture_5R19') #similar results with OutOfAfrica_3G09 and OutOfAfricaArchaicAdmixture_5R19
simsamples = model.get_samples(100, 100, 100)
engine = stdpopsim.get_engine('msprime')
sim = engine.simulate(model,contig,simsamples,seed=12345)
sim_gen=allel.HaplotypeArray(sim.genotype_matrix()).to_genotypes(ploidy=2)
sim_pos=np.array([s.position for s in sim.sites()])
sim_dc_all,sim_dc,sim_ac_all,sim_ac,sim_pos=filter_genotypes(sim_gen,sim_pos)

#mask inaccessistble sites
sim_dc=sim_dc[sim_pos>1.6e7,:]
sim_dc_all=sim_dc_all[sim_pos>1.6e7]
sim_pos=sim_pos[sim_pos>1.6e7]
dc=dc[pos>1.6e7,:]
dc_all=dc_all[pos>1.6e7]
pos=pos[pos>1.6e7]

#pi
print("simulation pi from tskit: "+str(sim.diversity()))
print("simulation pi from SNPs passing filters: "+str(allel.sequence_diversity(sim_pos,sim_ac_all)))
print("empirical pi from SNPs passing filters: "+str(allel.sequence_diversity(pos,ac_all)))

#segregating sites
print("simulation SNPs passing filters: "+str(sim_dc.shape[0]))
print("empirical SNPs passing filters: "+str(dc.shape[0]))

ac_all.shape
sim_ac_all.shape




#
#
# from matplotlib import pyplot as plt
# plt.hist(pos,bins=100)[2] #weird distribution of coverage
