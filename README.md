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
Download the [model weights](https://drive.google.com/file/d/1d8n4vHDSbXqpPc_JFEswLomSUDBgHvno/view?usp=drive_link)

## 2. Install dependencies
Please see the **Software Dependencies** section.

# Requirement

## OS Requirements
This package is supported for Linux. The package has been tested on the following systems:
- CentOS 7
- Red Hat 4.8.5

And it should be compatible with other common Linux systems.

## Software Dependencies


- Python 3.8 or higher
- numpy
- pandas
- pytorch>=1.10.2
- gffutils>=0.11.0
- tqdm
- sinkhorn-transformer
- pyfaidx
- pyvcf3

We suggest using `Anaconda` and `pypi` to install python packages. **bioconda** channel of Conda is recommended.

## Dataset Requirements

The SpliceTransformer requires a genome assembly file and a genome annotation database file to locate genes and strands.

(1) Download the genome assembly file from ensemble. 
<https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz> 

The uncompressed file should be placed at `./data/data_package/` and renamed into `hg38.fa`

(2) Download the genome annotation file from gencode.
<https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz>
The file should be placed at `./data/data_package/` and renamed into `hg38.annotation.gtf.gz`

The default configuration is for hg38. However, other versions of annotation can also be used.


---

# Run software

## 1.Annotate variants

Run `sptransformer.py` to predict mutation effects.
```bash
python sptransformer.py --input data/example/input38.vcf --output data/example/output38.tsv --reference hg38
```

>At the first running of the script, a hint message will be printed to guide users to build .db file for the .gtf files.

## 2.Reproduce analysis results

The code snippets for analysis performed in the article is represented in `tasks_annotate_mutations.py`.

```bash
python tasks_annotate_mutations.py [task]
```


The [task] should be replaced by task names recorded in the file. The path of input files should be updated in the code.

> The code snippets for analysis performed during earlier revision processes is represented in `old_annotator_pytorch.py` and `old_annotator_pytorch_2.py`

**Warning**: Some tasks can not run directly because the source data are not included in the repository. Details about how to get the input files are described in corresponding section of paper.

## 3.Custom usage

### Variants annotation
The python class `Annotator` in `sptransformer.py` showed examples for usage of SpTransformer. The code is able to be modified for custom usage.

Run this command to generate example outputs as `data/example/output38.csv`:
```bash
python sptransformer.py
```

Detailed options：
```bash
python sptransformer.py -I input_file -O output_file --reference hg38 --vcf True --raw_score False
```
input_file: .vcf or .csv containing variants, e.g. `data/example/input38.vcf`
output_file:  e.g. `data/example/output38.csv`
reference: hg38 or hg19
vcf: Set it to False if the input file is a .csv file. The input .csv file should contains at lease 4 columns:"CHROM POS REF ALT"
raw_score:
    - True: output raw Δtissue usage scores
    - False output tissue specificity prediction. 95% threshold (described in the article) was used.

**Note**: We have observed that the model's output may exhibit minor discrepancies across different GPU hardware. There is a small probability that the predicted results for individual mutations may be inconsistent across devices, but the model's statistical performance on large-scale datasets remains unaffected.

### Other application
The file `custom_usage.py` showed an example for getting raw outputs of SpTransformer model.

```python
python custom_usage.py
```


# License

This project is covered under the Apache 2.0 License.