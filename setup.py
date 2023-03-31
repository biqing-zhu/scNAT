import setuptools

with open("./README.md", "r", encoding='utf-8') as fh:
  long_description = fh.read()

setuptools.setup(
    name="scNAT",
    version="0.1.0",
    keywords=['scRNA-seq', 'scTCR-seq', 'data integration', 'deep learning', 'variational autoencoder', 'clone expansion'],
    author="Biqing Zhu",
    author_email="biqing.zhu@yale.edu",
    description="A deep learning method for integrating single cell RNA and the paired T cell receptor sequencing profiles.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT License',
    url="https://github.com/AprilYuge/ResPAN",
    install_requires = [
    'numpy',
    'pandas',
    'scipy',
    'tensorflow',
    'sklearn',
    'scanpy',
    'os',
    'Bio.SubsMat',
    'collections',
    'pickle',
    'matplotlib',
    'sys',
    'glob'.
    'seaborn',
    'umap',
    'warnings',
    'shutil',
    'multiprocessing'
    ],
    packages=['scNAT'],
    classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ],
)