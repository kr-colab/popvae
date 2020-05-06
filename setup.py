from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='popvae',
      version='0.1',
      description='popVAE: a variational autoencoder for population genetic data',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/cjbattey/locator',
      author='CJ Battey, Peter Ralph, Andrew Kern',
      author_email='cbattey2@uoregon.edu, plr@uoregon.edu, adk@uoregon.edu',
      license='NPOSL-3.0',
      packages=find_packages(exclude=[]),
      install_requires=["numpy",
                        "h5py",
                        "scikit-allel",
                        "matplotlib",
                        "scipy",
                        "keras",
                        "tensorflow==1.15.2",
                        "tqdm",
                        "pandas",
                        "zarr"],
      scripts=["scripts/popvae.py"],
      zip_safe=False,
      setup_requires=[]
)
