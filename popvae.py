import keras, numpy as np, os, allel, pandas as pd, time, random
import zarr, subprocess, h5py, re, sys, os, argparse
from matplotlib import pyplot as plt
from tqdm import tqdm
from keras import layers
from keras import backend as K
from keras.models import Model
import tensorflow

parser = argparse.ArgumentParser()
parser.add_argument(
    "--infile",
    help="path to input genotypes in vcf (.vcf | .vcf.gz), \
                          zarr, or .popvae.hdf5 format. Zarr files should be as produced \
                          by scikit-allel's `vcf_to_zarr( )` function. `.popvae.hdf5`\
                          files store filtered genotypes from previous runs (i.e. \
                          from --save_allele_counts).",
)
parser.add_argument("--out", default="vae", help="path for saving output")
parser.add_argument(
    "--patience", default=50, type=int, help="training patience. default=50"
)
parser.add_argument(
    "--max_epochs", default=500, type=int, help="max training epochs. default=500"
)
parser.add_argument("--batch_size", default=32, type=int, help="batch size. default=32")
parser.add_argument(
    "--save_allele_counts",
    default=False,
    action="store_true",
    help="save allele counts and and sample IDs to \
                    out+'.popvae.hdf5'.",
)
parser.add_argument(
    "--save_weights",
    default=False,
    action="store_true",
    help="save model weights to out+weights.hdf5.",
)
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="random seed. \
                                                         default: None",
)
parser.add_argument(
    "--train_prop",
    default=0.9,
    type=float,
    help="proportion of samples to use for training \
                          (vs validation). default: 0.9",
)
parser.add_argument(
    "--search_network_sizes",
    default=False,
    action="store_true",
    help="run grid search over network sizes and use the network with \
                          minimum validation loss. default: False. ",
)
parser.add_argument(
    "--width_range",
    default="32,64,128,256,512",
    type=str,
    help="range of hidden layer widths to test when `--search_network_sizes` is called.\
                          Should be a comma-delimited list with no spaces. Default: 32,64,128,256,512",
)
parser.add_argument(
    "--depth_range",
    default="3,6,10,20",
    type=str,
    help="range of network depths to test when `--search_network_sizes` is called.\
                          Should be a comma-delimited list with no spaces. Default: 4,6,8,10",
)
parser.add_argument(
    "--depth", default=6, type=int, help="number of hidden layers. default=6."
)
parser.add_argument(
    "--width", default=128, type=int, help="nodes per hidden layer. default=128"
)
parser.add_argument(
    "--gpu_number",
    default="0",
    type=str,
    help='gpu number to use for training (try `gpustat` to get GPU numbers).\
                          Use ` --gpu_number "" ` to run on CPU, and  \
                          ` --parallel --gpu_number 0,1,2,3` to split batches across 4 GPUs.\
                          default: 0',
)
parser.add_argument(
    "--prediction_freq",
    default=5,
    type=int,
    help="print predictions during training every \
                          --prediction_freq epochs. default: 10",
)
parser.add_argument(
    "--max_SNPs",
    default=None,
    type=int,
    help="If not None, randomly select --max_SNPs variants \
                          to run. default: None",
)
parser.add_argument(
    "--latent_dim", default=2, type=int, help="N latent dimensions to fit. default: 2"
)
parser.add_argument(
    "--PCA",
    default=False,
    action="store_true",
    help="Run PCA on the derived allele count matrix in scikit-allel.",
)
parser.add_argument(
    "--n_pc_axes",
    default=20,
    type=int,
    help="Number of PC axes to save in output. default: 20",
)
parser.add_argument(
    "--PCA_scaler",
    default="Patterson",
    type=str,
    help="How should allele counts be scaled prior to running the PCA?. \
                          Options: 'None' (mean-center the data but do not scale sites), \
                          'Patterson' (mean-center then apply the scaling described in Eq 3 of Patterson et al. 2006, Plos Gen)\
                          default: Patterson. See documentation of allel.pca for further information.",
)
parser.add_argument(
    "--plot",
    default=False,
    action="store_true",
    help="generate an interactive scatterplot of the latent space. requires --metadata. Run python scripts/plotvae.py --h for customizations",
)
parser.add_argument(
    "--metadata",
    default=None,
    help="path to tab-delimited metadata file with column 'sampleID'.",
)
args = parser.parse_args()

infile = args.infile
save_allele_counts = args.save_allele_counts
patience = args.patience
batch_size = args.batch_size
max_epochs = args.max_epochs
seed = args.seed
save_weights = args.save_weights
train_prop = args.train_prop
gpu_number = args.gpu_number
out = args.out
prediction_freq = args.prediction_freq
max_SNPs = args.max_SNPs
latent_dim = args.latent_dim
PCA = args.PCA
PCA_scaler = args.PCA_scaler
depth = args.depth
width = args.width
n_pc_axes = args.n_pc_axes
search_network_sizes = args.search_network_sizes
plot = args.plot
metadata = args.metadata

depth_range = args.depth_range
depth_range = np.array([int(x) for x in re.split(",", depth_range)])
width_range = args.width_range
width_range = np.array([int(x) for x in re.split(",", width_range)])

if args.plot:
    if args.metadata == None:
        print("ERROR: `--plot` argument requires `--metadata`")
        exit()

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

if not seed == None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)

print("\nloading genotypes")
if infile.endswith(".zarr"):
    callset = zarr.open_group(infile, mode="r")
    gt = callset["calldata/GT"]
    gen = allel.GenotypeArray(gt[:])
    samples = callset["samples"][:]
elif infile.endswith(".vcf") or infile.endswith(".vcf.gz"):
    vcf = allel.read_vcf(infile, log=sys.stderr)
    gen = allel.GenotypeArray(vcf["calldata/GT"])
    samples = vcf["samples"]
elif infile.endswith(".popvae.hdf5"):
    h5 = h5py.File(infile, "r")
    dc = np.array(h5["derived_counts"])
    samples = np.array(h5["samples"])
    h5.close()

# snp filters
if not infile.endswith(".popvae.hdf5"):
    print("counting alleles")
    ac_all = gen.count_alleles()  # count of alleles per snp
    ac = gen.to_allele_counts()  # count of alleles per snp per individual

    print("dropping non-biallelic sites")
    biallel = ac_all.is_biallelic()
    dc_all = ac_all[biallel, 1]  # derived alleles per snp
    dc = np.array(ac[biallel, :, 1], dtype="int_")  # derived alleles per individual
    missingness = gen[biallel, :, :].is_missing()

    print("dropping singletons")
    ninds = np.array([np.sum(x) for x in ~missingness])
    singletons = np.array([x <= 2 for x in dc_all])
    dc_all = dc_all[~singletons]
    dc = dc[~singletons, :]
    ninds = ninds[~singletons]
    missingness = missingness[~singletons, :]

    print("filling missing data with rbinom(2,derived_allele_frequency)")
    af = np.array([dc_all[x] / (ninds[x] * 2) for x in range(dc_all.shape[0])])
    for i in tqdm(range(np.shape(dc)[1])):
        indmiss = missingness[:, i]
        dc[indmiss, i] = np.random.binomial(2, af[indmiss])

    dc = np.transpose(dc)
    dc = dc * 0.5  # 0=homozygous reference, 0.5=heterozygous, 1=homozygous alternate

    # save hdf5 for reanalysis
    if save_allele_counts and not infile.endswith(".popvae.hdf5"):
        print("saving derived counts for reanalysis")
        outfile = h5py.File(infile + ".popvae.hdf5", "w")
        outfile.create_dataset("derived_counts", data=dc)
        outfile.create_dataset(
            "samples", data=samples, dtype=h5py.string_dtype()
        )  # requires h5py >= 2.10.0
        outfile.close()

if not max_SNPs == None:
    print("subsetting to " + str(max_SNPs) + " SNPs")
    dc = dc[:, np.random.choice(range(dc.shape[1]), max_SNPs, replace=False)]

print("running train/test splits")
ninds = dc.shape[0]
if train_prop == 1:
    train = np.random.choice(range(ninds), int(train_prop * ninds), replace=False)
    test = train
    traingen = dc[train, :]
    testgen = dc[test, :]
    trainsamples = samples[train]
    testsamples = samples[test]
else:
    train = np.random.choice(range(ninds), int(train_prop * ninds), replace=False)
    test = np.array([x for x in range(ninds) if x not in train])
    traingen = dc[train, :]
    testgen = dc[test, :]
    trainsamples = samples[train]
    testsamples = samples[test]

print("validation samples:" + testsamples)
print("running on " + str(dc.shape[1]) + " SNPs")

# grid search on network sizes. Getting OOM errors on 256 networks when run in succession -- GPU memory not clearing on new compile? unclear.
if search_network_sizes:
    print("running grid search on network sizes")
    tmp_patience = patience / 4

    # get parameter combinations (will need to rework this for >2 params)
    paramsets = [[x, y] for x in width_range for y in depth_range]

    # output dataframe
    param_losses = pd.DataFrame()
    param_losses["width"] = None
    param_losses["depth"] = None
    param_losses["val_loss"] = None

    # params=paramsets[0]
    for params in tqdm(paramsets):
        tmpwidth = params[0]
        tmpdepth = params[1]
        print("width=" + str(tmpwidth) + "\ndepth=" + str(tmpdepth))

        # load model
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(
                shape=(K.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0
            )
            return z_mean + K.exp(z_log_var) * epsilon

        # encoder
        input_seq = keras.Input(shape=(traingen.shape[1],))
        x = layers.Dense(tmpwidth, activation="elu")(input_seq)
        for i in range(tmpdepth - 1):
            x = layers.Dense(tmpwidth, activation="elu")(x)
        z_mean = layers.Dense(latent_dim)(x)
        z_log_var = layers.Dense(latent_dim)(x)
        z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")(
            [z_mean, z_log_var]
        )
        encoder = Model(input_seq, [z_mean, z_log_var, z], name="encoder")

        # decoder
        decoder_input = layers.Input(shape=(latent_dim,), name="z_sampling")
        x = layers.Dense(tmpwidth, activation="linear")(decoder_input)  # was elu
        for i in range(tmpdepth - 1):
            x = layers.Dense(tmpwidth, activation="elu")(x)
        output = layers.Dense(traingen.shape[1], activation="sigmoid")(
            x
        )  # hard sigmoid seems natural here but appears to lead to more left-skewed decoder outputs.
        decoder = Model(decoder_input, output, name="decoder")

        # end-to-end vae
        output_seq = decoder(encoder(input_seq)[2])
        vae = Model(input_seq, output_seq, name="vae")

        # get loss as xent_loss+kl_loss
        reconstruction_loss = keras.losses.binary_crossentropy(input_seq, output_seq)
        reconstruction_loss *= traingen.shape[1]
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        # kl_loss *= 5 #beta from higgins et al 2017, https://openreview.net/pdf?id=Sy2fzU9gl. Deprecated but loss term weighting is a thing to work on.
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)

        vae.compile(optimizer="adam")

        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=out + "_weights.hdf5",
            verbose=0,
            save_best_only=True,
            monitor="val_loss",
            period=1,
        )

        earlystop = keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=tmp_patience
        )

        reducelr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=int(tmp_patience / 4),
            verbose=0,
            mode="auto",
            min_delta=0,
            cooldown=0,
            min_lr=0,
        )
        history = vae.fit(
            x=traingen,
            y=None,
            shuffle=True,
            verbose=0,
            epochs=int(max_epochs / 4),
            callbacks=[checkpointer, earlystop, reducelr],
            validation_data=(testgen, None),
            batch_size=batch_size,
        )
        minloss = np.min(history.history["val_loss"])
        row = np.transpose(pd.DataFrame([tmpwidth, tmpdepth, minloss]))
        row.columns = ["width", "depth", "val_loss"]
        param_losses = param_losses.append(row, ignore_index=True)
        K.clear_session()  # maybe solves the gpu memory issue(???)

    # save tests and get min val_loss parameter set
    print(param_losses)
    param_losses.to_csv(out + "_param_grid.csv", index=False, header=True)
    bestparams = param_losses[
        param_losses["val_loss"] == np.min(param_losses["val_loss"])
    ]
    width = int(bestparams["width"])
    depth = int(bestparams["depth"])
    print("best parameters:\nwidth = " + str(width) + "\ndepth = " + str(depth))


# load model
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], latent_dim), mean=0.0, stddev=1.0, seed=seed
    )
    return z_mean + K.exp(z_log_var) * epsilon


# encoder
input_seq = keras.Input(shape=(traingen.shape[1],))
x = layers.Dense(width, activation="elu")(input_seq)
for i in range(depth - 1):
    x = layers.Dense(width, activation="elu")(x)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)
z = layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
encoder = Model(input_seq, [z_mean, z_log_var, z], name="encoder")

# decoder
decoder_input = layers.Input(shape=(latent_dim,), name="z_sampling")
x = layers.Dense(width, activation="linear")(decoder_input)  # was elu
for i in range(depth - 1):
    x = layers.Dense(width, activation="elu")(x)
output = layers.Dense(traingen.shape[1], activation="sigmoid")(
    x
)  # hard sigmoid seems natural here but appears to lead to more left-skewed decoder outputs.
decoder = Model(decoder_input, output, name="decoder")

# end-to-end vae
output_seq = decoder(encoder(input_seq)[2])
vae = Model(input_seq, output_seq, name="vae")

# get loss as xent_loss+kl_loss
reconstruction_loss = keras.losses.binary_crossentropy(input_seq, output_seq)
reconstruction_loss *= traingen.shape[1]
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
# kl_loss *= 5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

vae.compile(optimizer="adam")

# callbacks
checkpointer = keras.callbacks.ModelCheckpoint(
    filepath=out + "_weights.hdf5",
    verbose=1,
    save_best_only=True,
    monitor="val_loss",
    period=1,
)

earlystop = keras.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=0, patience=patience
)

reducelr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=int(patience / 4),
    verbose=1,
    mode="auto",
    min_delta=0,
    cooldown=0,
    min_lr=0,
)


def saveLDpos(encoder, predgen, samples, batch_size, epoch, frequency):
    if epoch % frequency == 0:
        pred = encoder.predict(predgen, batch_size=batch_size)[0]
        pred = pd.DataFrame(pred)
        pred["sampleID"] = samples
        pred["epoch"] = epoch
        if epoch == 0:
            pred.to_csv(
                out + "_training_preds.txt",
                sep="\t",
                index=False,
                mode="w",
                header=True,
            )
        else:
            pred.to_csv(
                out + "_training_preds.txt",
                sep="\t",
                index=False,
                mode="a",
                header=False,
            )


print_predictions = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: saveLDpos(
        encoder=encoder,
        predgen=dc,
        samples=samples,
        batch_size=batch_size,
        epoch=epoch,
        frequency=prediction_freq,
    )
)

# training
t1 = time.time()
history = vae.fit(
    x=traingen,
    y=None,
    shuffle=True,
    epochs=max_epochs,
    callbacks=[checkpointer, earlystop, reducelr, print_predictions],
    validation_data=(testgen, None),
    batch_size=batch_size,
)
t2 = time.time()
vaetime = t2 - t1
print("VAE run time: " + str(vaetime) + " seconds")

# save training history
h = pd.DataFrame(history.history)
h.to_csv(out + "_history.txt", sep="\t")

# predict latent space coords for all samples from weights minimizing val loss
vae.load_weights(out + "_weights.hdf5")
pred = encoder.predict(
    dc, batch_size=batch_size
)  # returns [mean,sd,sample] for individual distributions in latent space
p = pd.DataFrame()
if latent_dim == 2:
    p["mean1"] = pred[0][:, 0]
    p["mean2"] = pred[0][:, 1]
    p["sd1"] = pred[1][:, 0]
    p["sd2"] = pred[1][:, 1]
    pred = p
else:
    pred = pd.DataFrame(pred[0])
    pred.columns = ["LD" + str(x + 1) for x in range(len(pred.columns))]
pred["sampleID"] = samples
pred.to_csv(out + "_latent_coords.txt", sep="\t", index=False)

if not save_weights:
    subprocess.check_output(["rm", out + "_weights.hdf5"])

if PCA:
    pcdata = np.transpose(dc)
    t1 = time.time()
    print("running PCA")
    pca = allel.pca(pcdata, scaler=PCA_scaler, n_components=n_pc_axes)
    pca = pd.DataFrame(pca[0])
    colnames = ["PC" + str(x + 1) for x in range(n_pc_axes)]
    pca.columns = colnames
    pca["sampleID"] = samples
    pca.to_csv(out + "_pca.txt", index=False, sep="\t")
    t2 = time.time()
    pcatime = t2 - t1
    print("PCA run time: " + str(pcatime) + " seconds")

######### plots #########
# training history
# plt.switch_backend('agg')
fig = plt.figure(figsize=(3, 1.5), dpi=200)
plt.rcParams.update({"font.size": 7})
ax1 = fig.add_axes([0, 0, 1, 1])
ax1.plot(
    history.history["val_loss"][3:],
    "--",
    color="black",
    lw=0.5,
    label="Validation Loss",
)
ax1.plot(history.history["loss"][3:], "-", color="black", lw=0.5, label="Training Loss")
ax1.set_xlabel("Epoch")
# ax1.set_yscale('log')
ax1.legend()
fig.savefig(out + "_history.pdf", bbox_inches="tight")

if PCA:
    timeout = np.array([vaetime, pcatime])
    np.savetxt(X=timeout, fname=out + "_runtimes.txt")

if plot:
    subprocess.run(
        "python scripts/plotvae.py --latent_coords "
        + out
        + "_latent_coords.txt"
        + " --metadata "
        + metadata,
        shell=True,
    )

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
