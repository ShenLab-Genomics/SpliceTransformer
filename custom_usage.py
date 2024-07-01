import pandas as pd
import numpy as np
from pyfaidx import Fasta
import argparse
import vcf as pyvcf
from pyensembl import Genome
import tqdm
import os
from sptransformer import Annotator
import torch

if __name__ == '__main__':
    annotator = Annotator()
    gtf = annotator.gtf

    tis_names = ['Adipose Tissue', 'Blood', 'Blood Vessel', 'Brain', 'Colon', 'Heart', 'Kidney',
                 'Liver', 'Lung', 'Muscle', 'Nerve', 'Small Intestine', 'Skin', 'Spleen', 'Stomach']

    input_seq = 'N'*4000 + 'ACGTAGGGCG' + 'N'*4000  # just an example
    input_seq = annotator.model.one_hot_encode(input_seq)
    input_seq = torch.tensor(input_seq).to(annotator.model.device)
    print(input_seq.shape)
    # the function step() accepts encoded sequence, (Batch, 4, Length),
    # thus, the input_seq should have shape (1, 4, Length)
    input_seq = input_seq.unsqueeze(0).float().transpose(1, 2)
    output = annotator.model.step(input_seq)
    print(output.shape)
