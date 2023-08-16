from pkg_resources import resource_filename
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import torch
from torch.nn import functional as F
import gffutils
import argparse
import logging
import vcf as pyvcf
from model.model import SpTransformer
from model import SpliceAI as re_sp
import tqdm


IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
# One-hot encoding of the inputs: 0 is for padding, and 1, 2, 3, 4 correspond
# to A, C, G, T respectively.

OUT_MAP = np.asarray([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
# One-hot encoding of the outputs: 0 is for no splice, 1 is for acceptor,
# 2 is for donor and -1 is for padding.


class Annotator:

    def __init__(self, ref_fasta, annotations, models=None, use_spliceai=False):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print('torch device:{}'.format(self.device))

        try:
            self.ref_fasta = Fasta(ref_fasta, rebuild=False)
            self.ref_seq_len = {}
            for key in self.ref_fasta.keys():
                self.ref_seq_len[key] = len(self.ref_fasta[key])
        except IOError as e:
            logging.error('{}'.format(e))
            exit()

        self.use_spliceai = False
        if use_spliceai:
            from tensorflow.keras.models import load_model
            self.use_spliceai = True
            paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
            self.models = [load_model(resource_filename('spliceai', x))
                           for x in paths]
        else:
            self.models = [model.to(self.device).eval() for model in models]

    def one_hot_encode(self, seq):
        seq = seq.upper().replace('A', '1').replace('C', '2')
        seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
        seq = np.asarray(list(map(int, list(seq))))
        seq = IN_MAP[seq.astype('int8')]
        return seq

    def normalise_chrom(self, source, target='chr'):
        '''
        get consistent chromosome format. 
        e.g 'ChrX' -> 'X' or 'X' -> 'ChrX'
        '''
        source = str(source)
        if source[:3] == 'Chr':
            source = 'chr'+source[3:]

        def has_prefix(x):
            return x.startswith('chr')

        if has_prefix(source) and not has_prefix(target):
            return source.strip('chr')
        elif not has_prefix(source) and has_prefix(target):
            return 'chr'+source

        return source

    def step(self, seq):
        '''
        input:
            seq: (B,4,N)
        output:
            score: (B,13,N)
        '''
        output = []
        assert len(seq.size()) == 3
        with torch.no_grad():
            for i in range(len(self.models)):
                self.models[i].eval()
                out = self.models[i](seq, use_usage_head=True)
                out[:, :3, :] = F.softmax(out[:, :3, :], dim=1)
                out[:, 3:, :] = torch.sigmoid(out[:, 3:, :])
                output.append(out.cpu().detach().numpy())
            output = torch.tensor(np.mean(output, axis=0))
        return output

    def calc_one_seq(self, seq):
        '''
        input:
            seq: sequence vector with shape of (N,4)
            clip_left
            clip_right
        output:
            score: score vector with shape of (N-clip_left-clip_right, 3 + 11)
        '''
        if self.use_spliceai:
            return self.calc_one_seq_spliceai(seq)

        seq = torch.tensor(seq).to(self.device)
        seq = seq.unsqueeze(0).transpose(1, 2)
        # print(seq.shape)
        res = self.step(seq.float())
        res = res[0].transpose(0, 1).numpy()
        # print(res.shape)
        return res

    def calc_two_seqs(self, seq):
        if self.use_spliceai:
            return self.calc_one_seq_spliceai(seq)
        seq = torch.tensor(seq).to(self.device)
        seq = seq.transpose(1, 2)
        res = self.step(seq.float())
        res = res.transpose(1, 2).numpy()
        return res

    def calc_one_seq_spliceai(self, seq):
        seq = seq.reshape(1, -1, 4)
        res = [np.array(self.models[m](seq))
               for m in range(5)]
        res = np.mean(res, axis=0)
        res = res.reshape(-1, 3)
        return res

    def calc_one_snp(self, chrom, strand, center_pos, snp_pos, ref, alt, context_len=5000):
        CONTEXT_LEN = context_len
        chrom = self.normalise_chrom(
            chrom, list(self.ref_fasta.keys())[0])
        # check boundary
        chr_len = self.ref_seq_len[chrom]
        if (center_pos < CONTEXT_LEN) or (center_pos > chr_len-CONTEXT_LEN):
            logging.warning(
                'Skipping record (near chromosome end): {} {}'.format(chrom, center_pos))
        seq = self.ref_fasta.get_seq(
            chrom, center_pos-CONTEXT_LEN, center_pos+CONTEXT_LEN)
        # print('seq len:{}'.format(len(str(seq))))
        offset = snp_pos - center_pos + CONTEXT_LEN
        if str(seq[offset]).upper() != ref:
            print('WA: chr:{} pos:{} seq:{} ref:{}'.format(
                chrom, snp_pos, seq[offset-5:offset+6], ref))
            exit()
        seq_ref = str(seq)
        seq_alt = seq_ref[:offset] + str(alt) + seq_ref[offset+1:]
        seq_ref = self.one_hot_encode(seq_ref)
        seq_alt = self.one_hot_encode(seq_alt)
        if strand == '-':
            seq_ref = seq_ref[::-1, ::-1].copy()
            seq_alt = seq_alt[::-1, ::-1].copy()
        # calc scores
        score_ref = self.calc_one_seq(seq_ref)
        score_alt = self.calc_one_seq(seq_alt)
        delta = [score_alt, score_ref]
        return delta

    def calc_one_snp_interval(self, chrom, strand, center_posL, center_posR, snp_pos, ref, alt, context_len=5000, check_ref=True):
        CONTEXT_LEN = context_len
        chrom = self.normalise_chrom(
            chrom, list(self.ref_fasta.keys())[0])
        # check boundary
        chr_len = self.ref_seq_len[chrom]
        # print(chr_len)
        if (center_posL < CONTEXT_LEN) or (center_posR > chr_len-CONTEXT_LEN):
            logging.warning(
                'Skipping record (near chromosome end): {} {}'.format(chrom, center_posL))
        seq = self.ref_fasta.get_seq(
            chrom, center_posL-CONTEXT_LEN, center_posR+CONTEXT_LEN)
        # print('seq len:{}'.format(len(str(seq))))
        offset = snp_pos - center_posL + CONTEXT_LEN
        if ref is None:
            ref = str(seq[offset])
        if check_ref or ((len(ref) > 1) and (len(ref) == len(alt))):
            if str(seq[offset]).upper() != str(ref).upper():
                print('WA: chr:{} pos:{} seq:{} ref:{}'.format(
                    chrom, offset, seq[offset], ref))

        seq_ref = str(seq)
        seq_alt = seq_ref[:offset] + str(alt) + seq_ref[offset+len(ref):]
        seq_ref = self.one_hot_encode(seq_ref)
        seq_alt = self.one_hot_encode(seq_alt)
        if strand == '-':
            seq_ref = seq_ref[::-1, ::-1].copy()
            seq_alt = seq_alt[::-1, ::-1].copy()
        # calc scores
        score_ref, score_alt = self.calc_two_seqs(np.array([seq_ref, seq_alt]))
        delta = [score_alt, score_ref]
        return delta

    def calc_one_snp_single(self, chr, strand, center_pos, snp_pos, ref, alt):
        CONTEXT_LEN = 5000
        chr = self.normalise_chrom(
            chr, list(self.ref_fasta.keys())[0])
        # check boundary
        chr_len = self.ref_seq_len[chr]
        if (center_pos < CONTEXT_LEN) or (center_pos > chr_len-CONTEXT_LEN):
            logging.warning(
                'Skipping record (near chromosome end): {} {}'.format(chr, center_pos))
        seq = self.ref_fasta.get_seq(
            chr, center_pos-CONTEXT_LEN, center_pos+CONTEXT_LEN)
        offset = snp_pos - center_pos + CONTEXT_LEN
        if str(seq[offset]).upper() != ref:
            if str(seq[offset]).upper() == alt:
                t = ref
                ref = alt
                alt = t
            else:
                print('WA: chr:{} pos:{} seq:{} ref:{}'.format(
                    chr, offset, seq[offset], ref))
                exit()

        seq_ref = str(seq)
        seq_alt = seq_ref[:offset] + str(alt) + seq_ref[offset+1:]
        seq_ref = self.one_hot_encode(seq_ref)
        seq_alt = self.one_hot_encode(seq_alt)
        if strand == '-':
            seq_ref = seq_ref[::-1, ::-1].copy()
            seq_alt = seq_alt[::-1, ::-1].copy()
        # calc scores
        score_ref = self.calc_one_seq(seq_ref)
        score_alt = self.calc_one_seq(seq_alt)
        score_ref = score_ref.reshape(-1)
        score_alt = score_alt.reshape(-1)
        delta = [score_alt, score_ref]
        return delta

    def calc_one_site(self, chr, strand, center_pos, context_len=4000):
        CONTEXT_LEN = context_len
        chr = self.normalise_chrom(
            chr, list(self.ref_fasta.keys())[0])
        # check boundary
        chr_len = self.ref_seq_len[chr]
        if (center_pos < CONTEXT_LEN) or (center_pos > chr_len-CONTEXT_LEN):
            logging.warning(
                'Skipping record (near chromosome end): {} {}'.format(chr, center_pos))
        seq = self.ref_fasta.get_seq(
            chr, center_pos-CONTEXT_LEN, center_pos+CONTEXT_LEN)
        seq_ref = str(seq)
        seq_ref = self.one_hot_encode(seq_ref)
        if strand == '-':
            seq_ref = seq_ref[::-1, ::-1].copy()
        # calc scores
        score_ref = self.calc_one_seq(seq_ref)
        return score_ref

    def calc_special_seq(self, seq: str, context_len=4000):
        CONTEXT_LEN = context_len
        seq = 'N'*CONTEXT_LEN + seq + 'N'*CONTEXT_LEN
        seq = self.one_hot_encode(seq)
        score_seq = self.calc_one_seq(seq)
        return score_seq


try:
    from sys.stdin import buffer as std_in
    from sys.stdout import buffer as std_out
except ImportError:
    from sys import stdin as std_in
    from sys import stdout as std_out


def initialization(ref_fasta='hg19', grch='grch37', use_spliceai=False, context_len=4000):
    gtf = None
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # ref
    if ref_fasta == 'hg19':
        db = './data/genome_annotation/gencode.v37.annotation.gtf.gz.db'
        gtf = gffutils.FeatureDB(db)
        ref_fasta = './data/genome_annotation/hg19.fa'
    elif ref_fasta == 'hg38':
        db = './data/genome_annotation/gencode.v38.annotation.gtf.gz.db'
        gtf = gffutils.FeatureDB(db)
        ref_fasta = '/data/genome_annotation/hg38.fa'
    else:
        raise NotImplementedError
    if use_spliceai == 'retrained':
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        model = re_sp.SpliceAI(re_sp.L, re_sp.W, re_sp.AR)
        save_dict = torch.load(
            './spliceai_retrained/best.ckpt', map_location=device)
        annotator = Annotator(ref_fasta, grch, models=[
                              model], use_spliceai=False)
        use_spliceai = False
    elif use_spliceai == True:
        annotator = Annotator(ref_fasta, grch, models=[
                              None], use_spliceai=True)
    else:
        from model.model import SpTransformer
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        model = SpTransformer(64, usage_head=11, context_len=context_len)
        save_dict = torch.load(
            './model/weights/SpTransformer_pytorch.ckpt', map_location=device)
        model.load_state_dict(save_dict["state_dict"])
        annotator = Annotator(ref_fasta, grch, models=[model])
    return annotator, device, gtf


def get_gene(gene_db, chr, pos, target_name=None):
    genes = gene_db.region((chr, pos-1, pos-1), featuretype="gene")
    start = []
    end = []
    for gene in genes:
        name = gene["gene_name"]
        if target_name and (str(name[0]) != str(target_name)):
            continue
        strand = gene.strand
        for exon in gene_db.children(gene, featuretype="exon"):
            # print(exon)
            start.append(exon.start)
            end.append(exon.end)
        return (strand, start, end)
    return (None, None, None)


def get_genes(gene_db, chr, pos):
    genes = gene_db.region((chr, pos, pos), featuretype="gene")
    strand = []
    start = []
    end = []
    for gene in genes:
        name = gene["gene_name"]
        strand.append(gene.strand)
        te = 0
        ts = 2147483647
        for exon in gene_db.children(gene, featuretype="exon"):
            # print(exon)
            te = max(te, exon.end)
            ts = min(ts, exon.start)

        start.append(ts)
        end.append(te)
        # return (strand, start, end)
    return strand, start, end


def analysis_exon_inclusion(fname, use_spliceai=False):
    datafile = pd.read_csv(fname, delimiter='\t')
    datafile = datafile[datafile['chr'].isin(
        ['chr1'])]
    # datafile = datafile[datafile['chr'].isin(
    #     ['chr1', 'chr3', 'chr5', 'chr7', 'chr9'])]
    annotator, _, gtf = initialization(
        ref_fasta='hg38', grch='grch38', use_spliceai=use_spliceai)
    pred_list = []
    label_list = []
    for idx, row in tqdm.tqdm(datafile.iterrows(), mininterval=5):
        chr = row['chr']
        strand = row['strand']
        site1 = row['start'] + 1
        site2 = row['end'] - 1
        label = row['exon_inclusion']
        CONTEXT = 4000
        if use_spliceai != False:
            CONTEXT = 5000
        score1 = annotator.calc_one_site(
            chr, strand, site1, context_len=CONTEXT)
        score2 = annotator.calc_one_site(
            chr, strand, site2, context_len=CONTEXT)
        pred = np.mean([score1.reshape(-1), score2.reshape(-1)], axis=0)
        pred_list.append(pred)
        label_list.append(label)
    output_name = 'exon_inclusion_stsplice.npz'
    if use_spliceai == 'retrained':
        output_name = 'exon_inclusion_spliceai_retrained.npz'
    elif use_spliceai == True:
        output_name = 'exon_inclusion_spliceai.npz'

    np.savez(output_name, pred=pred_list, label=label_list)
    return


def filt_clinvar(fname, outputname):  # 弃用
    import tqdm
    df = pd.read_csv(fname, header=33, delimiter='\t')
    df_filter = pd.DataFrame(columns=df.columns)
    cnt = 0
    for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        info = str(row['INFO']).split(';')
        tmp = 0
        if len(info) >= 3:  # have spliceai
            s3 = info[2].split('|')
            i = 0
            while i < len(s3):
                tmp = np.max((float(s3[2]), float(s3[3]),
                             float(s3[4]), float(s3[5]), tmp))
                i += 9
            pass
        s1 = float(info[0][20:])
        s2 = float(info[1][22:])
        if np.max((tmp, s1, s2)) < 0.1:
            continue
        # print((tmp,s1,s2),sep='\t')
        row['spliceai'] = tmp
        row['heart'] = s1
        row['heartsp'] = s2
        df_filter.loc[cnt] = row
        cnt += 1
    df_filter.to_csv(outputname, sep='\t')


def calc_motif(target_chr, target_pos, ref_base, strand):
    annotator, _, gtf = initialization(
        ref_fasta='hg38', grch='grch38', use_spliceai=False)
    d_range = 80
    context_len = 4000
    score_list = []
    cnt = 0
    site_pos = 80
    sequence = annotator.ref_fasta.get_seq(
        target_chr, target_pos-d_range, target_pos+d_range, rc=False)
    pred_alt, pred_ref = annotator.calc_one_snp_interval(
        target_chr, strand, target_pos-d_range, target_pos+d_range, target_pos, sequence[d_range], 'A', context_len=context_len)
    print(pred_ref[site_pos, :])
    for i in range(-d_range, d_range+1):
        tmp = []
        for base in ['A', 'G', 'C', 'T']:
            pred_alt, pred_ref = annotator.calc_one_snp_interval(
                target_chr, strand, target_pos-d_range, target_pos+d_range, target_pos+i, sequence[d_range+i], base, context_len=context_len)
            score_splice = np.max(
                np.abs(pred_alt[site_pos, 7] - pred_ref[site_pos, 7]))
            print(np.where(np.max(pred_ref[:, 1:3], axis=1) > 0.5))
            tmp.append(score_splice)
        score_list.append(tmp)
        cnt += 1
    score_list = np.array(score_list)
    import matplotlib.pyplot as plt
    x = range(score_list.shape[0])
    scores = []
    if strand == '+':
        for pos in range(score_list.shape[0]):
            print(sequence[pos])
            print(np.max(score_list[pos, :]))
            scores.append(
                [(str(sequence[pos]).upper(), np.max(score_list[pos, :])*2)])
    if strand == '-':
        sequence = annotator.ref_fasta.get_seq(
            target_chr, target_pos-d_range, target_pos+d_range, rc=True)
        for pos in range(score_list.shape[0]):
            print(sequence[pos])
            print(np.max(score_list[pos, :]))
            scores.append([(str(sequence[score_list.shape[0] - 1 - pos]).upper(),
                          np.max(score_list[score_list.shape[0]-1-pos, :])*2)])
    # plt.savefig('outputs/motif/res.png')
    from pyseqlogo.pyseqlogo import draw_logo, setup_axis
    plt.rcParams['figure.dpi'] = 300
    fig = plt.figure(figsize=(80, 5))
    ax = plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=0.01)
    draw_logo(scores, yaxis='probability',
              draw_axis=True, ax=ax, coordinate_type='data')
    plt.savefig(
        'outputs/motif/{}_{}_{}.png'.format(target_chr, target_pos, 'kidney'))


def create_genome_db(fname):
    def filter(feat):
        if feat.featuretype not in ["gene", "transcript", "exon"]:
            return False
        elif feat.featuretype in ["transcript", "exon"]:
            present = False
            for tag in 'Ensembl_canonical':
                if "tag" in feat.attributes and tag in feat["tag"]:
                    present = True
            if not present:
                return False
        return feat

    gffutils.create_db(fname, fname + ".db", force=True,
                       disable_infer_genes=True, disable_infer_transcripts=True,
                       transform=filter)


if __name__ == '__main__':
    import sys
    target = sys.argv[1]

    if target == 'Create_annotation_db':
        input_file = sys.argv[2]
        create_genome_db(input_file)

    if target == 'EI_0':
        analysis_exon_inclusion(
            'outputs/exon_inclusion.csv', use_spliceai=False)
    if target == 'EI_1':
        analysis_exon_inclusion(
            'outputs/exon_inclusion.csv', use_spliceai=True)
    if target == 'EI_2':  # retrained spliceai
        analysis_exon_inclusion(
            'outputs/exon_inclusion.csv', use_spliceai='retrained')

    if target == 'Motif':
        calc_motif('chr2', 178547003, 'A', '-')
