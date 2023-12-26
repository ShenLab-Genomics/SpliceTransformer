# SpliceTransformer

The SpliceTransformer (SpTransformer) is a deep learning tool designed to predict tissue-specific splicing sites from pre-mRNA sequences.


# Installation
## 1. Retrive the repository

The program is developed in Python and the source code can be run directly.

However, the model weights file is too large to be uploaded to GitHub, we support two ways to retrieve the repository.

### 1.1. Download through Git-LFS
A copy of the well-trained model weights is managed by Git-LFS.

If you are using the Conda package manager, execute `conda install git-lfs` to install the tool.

After installing Git-LFS, use the following command:
```bash
git lfs install
git lfs clone https://github.com/ShenLab-Genomics/SpliceTransformer
```
to clone this repository.

### 1.2. (or) Download from Google Drive

Clone the repository from GitHub first, then download the model weights from Google Drive and put them into the `model/weights` folder.
```bash
git clone https://github.com/ShenLab-Genomics/SpliceTransformer.git
```
Download the [model weights](https://drive.google.com/file/d/1u7owrAgX7K1MUiP-6AWnC4T1Jrql2eig/view?usp=drive_link)

## 2. Install dependencies
Please see the **Software Dependencies** section.

# Requirement

## OS Requirements
This package is supported for Linux. The package has been tested on the following systems:
- CentOS 7
- Red Hat 4.8.5

And it should be compatible with other common Linux systems.

## Software Dependencies


- Python 3.8
- numpy
- pandas
- pytorch>=1.10.2
- gffutils>=0.11.0
- tqdm
- sinkhorn-transformer
- pyfaidx
- pyvcf3
- pyensembl

We suggest using `Anaconda` and `pypi` to install python packages. **bioconda** channel of Conda is recommended.

Example:
```
conda create -n spt-test python==3.8 numpy pandas gffutils tqdm pytorch pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch -c conda-forge -c bioconda

conda activate spt-test

pip install sinkhorn-transformer pyfaidx pyvcf3 pyensembl
```


## Dataset Requirements

The SpliceTransformer requires a genome assembly file and a genome annotation database file to locate genes and strands.

(1) Download the genome assembly file from ensemble. 
<https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz> 

The uncompressed file should be placed at `./data/data_package/` and renamed into `hg38.fa`

(2) Download the genome annotation file from gencode.
<https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz>
The file should be placed at `./data/data_package/` and renamed into `gencode.v38.annotation.gtf.gz`

The default configuration is for hg38. However, other versions of annotation can also be used (the fasta files should contain chromosome labels like `chr1` rather than `1`).


---

# Run software

## 1.Annotate variants

Run `sptransformer.py` to predict mutation effects.
```bash
python sptransformer.py --input data/example/input38.vcf --output data/example/output38.tsv --reference hg38
```

>At the first running of the script, a hint message will be printed to guide users to build .db file for the .gtf files.

## 2.Reproduce analysis results

The code snippets for analysis performed in the article is represented in `annotator_pytorch.py` and `annotator_pytorch_2.py`

```bash
python annotator_pytorch.py [task]
python annotator_pytorch_2.py [task]
```

The [task] should be replaced by task names recorded in the files `annotator_pytorch_2.py` or `annotator_pytorch.py`. The path of input files should be updated in the code.

**Warning**: Some tasks can not run directly because the source data are not included in the repository. Details about how to get the input files are described in corresponding section of paper.

## 3.Custom usage

The python class `Annotator` in `sptransformer.py` showed examples for usage of SpTransformer. The code is able to be modified for custom usage.


# License

This project is covered under the Apache 2.0 License.