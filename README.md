

# SpliceTransformer

The SpliceTransformer (SpTransformer) is a deep learning tool designed to predict tissue-specific splicing sites from pre-mRNA sequences.

# Update log

2024.10.24 Our paper has been published at Nature Communications! [Link](https://www.nature.com/articles/s41467-024-53088-6) 
**We are continuously maintaining and improving our tool and associated web services. If you have any suggestions or encounter any issues, please do not hesitate to contact us via email or Github issues.**

# Citation
If you use the code or the data for your research, please cite our paper as follows:
```
@article{You2024,
  author    = {You, Ningyuan and Liu, Chang and Gu, Yuxin and Wang, Rong and Jia, Hanying and Zhang, Tianyun and Jiang, Song and Shi, Jinsong and Chen, Ming and Guan, Min-Xin and Sun, Siqi and Pei, Shanshan and Liu, Zhihong and Shen, Ning},
  title     = {{SpliceTransformer predicts tissue-specific splicing linked to human diseases}},
  journal   = {Nature Communications},
  year      = {2024},
  volume    = {15},
  number    = {1},
  pages     = {9129},
  month     = {oct},
  doi       = {10.1038/s41467-024-53088-6},
  issn      = {2041-1723},
  url       = {https://doi.org/10.1038/s41467-024-53088-6}
}
```

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

## Hardware Requirements

We recommend run the software on a computer with a GPU. However, the lightweight example data can also be handled with CPU only.

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
The file should be placed at `./data/data_package/` and renamed into `hg38.annotation.gtf.gz`

The default configuration is for hg38. However, other versions of annotation can also be used (the fasta files should contain chromosome labels like `chr1` rather than `1`).


---

# Run software

## 1.Annotate variants

Run `sptransformer.py` to predict mutation effects. The example output should be a `.tsv` table file containing the prediction.
```bash
python sptransformer.py --input data/example/input38.vcf --output data/example/output38.tsv --reference hg38
```

>At the first running of the script, a hint message will be printed to guide users to build .db file for the .gtf files.

>Expected run time for the sample vcf file on a normal desktop computer is no more than 5 minutes.

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
input_file: A .vcf or .csv file containing variants, e.g. `data/example/input38.vcf`

output_file: The output file, e.g. `data/example/output38.csv`

reference: The reference genome, either `hg38` or `hg19`

vcf: Set this to False if the input file is a .csv file. The input .csv file should contain at least 4 columns: "CHROM POS REF ALT"

raw_score:
    - True: Output raw Δtissue usage scores
    - False: Output tissue specificity prediction. The 95% threshold (as described in the article) is used.

**Note**: We have observed that the model's output may exhibit minor discrepancies across different GPU hardware. There is a small probability that the predicted results for individual mutations may be inconsistent across devices, but the model's statistical performance on large-scale datasets remains unaffected.

### Other application
The file `custom_usage.py` showed an example for getting raw outputs of SpTransformer model.

```python
python custom_usage.py
```


# License

This project is covered under the Apache 2.0 License.
