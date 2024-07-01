import logging
import tqdm
from torch.nn import functional as F
import torch
from pyfaidx import Fasta
import numpy as np
import pandas as pd
from pyensembl import Genome
import vcf as pyvcf
from model.model import SpTransformer
from model import SpliceAI


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


def one_hot_encode(seq):
    '''
    Parse input RNA sequence into one-hot-encoding format
    '''
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    seq = np.asarray(list(map(int, list(seq))))
    seq = IN_MAP[seq.astype('int8')]
    return seq


class ModelDriver:
    def __init__(self, ref_fasta, models=None, load_db=True) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # self.device = torch.device("cpu")
        print('torch device:{}'.format(self.device))
        if load_db:
            try:
                self.ref_fasta = Fasta(ref_fasta, rebuild=False)
                self.ref_seq_len = {}
                for key in self.ref_fasta.keys():
                    self.ref_seq_len[key] = len(self.ref_fasta[key])
            except IOError as e:
                logging.error('{}'.format(e))
                exit()
            print(f'Loaded fasta from {ref_fasta}')
        # load model to target device
        self.models = [model.to(self.device).eval() for model in models]
        self.limit_protein_coding = False
        pass

    def get_genes(self, gtf: Genome, chrom, pos):
        # lookup gene and strands from gtf database
        # genes = gtf.region((chrom, pos, pos), featuretype="gene")
        chrom = self.normalise_chrom(chrom, 'chr')
        genes = gtf.genes_at_locus(chrom, int(pos))
        strand = []
        start = []
        end = []
        # get strand and start/end position of gene
        for gene in genes:
            if self.limit_protein_coding:
                if gene.biotype != 'protein_coding':
                    continue
            strand.append(gene.strand)
        if len(strand) > 1 and strand[0] == '-' and strand[1] == '+':
            strand = ['+', '-']
        return strand, start, end

    def one_hot_encode(self, seq):
        '''
        Parse input RNA sequence into one-hot-encoding format
        '''
        seq = seq.upper().replace('A', '1').replace('C', '2')
        seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
        seq = np.asarray(list(map(int, list(seq))))
        seq = IN_MAP[seq.astype('int8')]
        return seq

    def normalise_chrom(self, source, target='chr'):
        '''
        Get consistent chromosome format. 
        e.g 'ChrX' -> 'chrX' and 'X' -> 'chrX'

        target: wanted prefix, default as 'chr'
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

    def post_decorate(self, inputs):
        """
        """
        raise NotImplementedError

    def step(self, inputs: torch.Tensor):
        """
        Parameters
        ----------
        input: tensor 
            Encoded sequence, (B, 4, N)
        Returns
        -------
        output: tensor
            Output of selected model
        """
        output = []
        assert len(inputs.size()) == 3
        with torch.no_grad():
            for i in range(len(self.models)):
                out = self.models[i](inputs).cpu().detach()
                out = self.post_decorate(out)
                output.append(out.numpy())
            output = torch.mean(torch.tensor(np.array(output)), dim=0)
        return output

    def calc_single_sequence(self, seq, encode=False):
        """
        Calculate model output for only one sequence

        Parameters
        ---
        seq: str or np.array or torch.Tensor
            input sequence, string or one-hot encoded vector (N,4)

        Returns
        ---
        output: numpy array
            model output
        """
        if encode:
            raise NotImplementedError
        seq = torch.tensor(seq).to(self.device)
        seq = seq.unsqueeze(0).transpose(1, 2)
        res = self.step(seq.float())
        res = res[0].transpose(0, 1).numpy()
        return res

    def calc_batched_sequence(self, seq, encode=False):
        """
        Calculate model output for only one sequence

        Parameters
        ---
        seq: (list of str), np.array, or torch.Tensor
            input sequence, string list or one-hot encoded vector (B,N,4)

        Returns
        ---
        output: numpy array
            model output
        """
        if encode:
            raise NotImplementedError
        seq = torch.tensor(seq).to(self.device)
        seq = seq.transpose(1, 2)  # 　(B, N, 4) -> (B, 4, N)
        res = self.step(seq.float())
        res = res.transpose(1, 2).numpy()  # 　(B, 4, N) -> (B, N, 4)
        return res


class MutationRecord:
    def __init__(self) -> None:
        pass

    def normalise_chrom(self, source, target='chr'):
        '''
        Get consistent chromosome format. 
        e.g 'ChrX' -> 'chrX' and 'X' -> 'chrX'

        target: wanted prefix, default as 'chr'
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


class SNP(MutationRecord):
    def __init__(self, chrom, strand, snp_pos, ref: str, alt: str, center_pos=None) -> None:
        super().__init__()
        chrom = self.normalise_chrom(
            chrom)
        self.chrom = chrom
        self.strand = strand
        self.snp_pos = snp_pos
        if center_pos is None:
            center_pos = snp_pos
        self.center_pos = center_pos
        self.ref = ref
        self.alt = alt

    def parse(self, ref_fasta: Fasta, context_len: int):
        seq = ref_fasta.get_seq(
            self.chrom, self.center_pos-context_len, self.center_pos+context_len)
        offset = self.snp_pos - self.center_pos + context_len
        if str(seq[offset]).upper() != self.ref:
            print('WA: chr:{} pos:{} seq:{} ref:{}'.format(
                self.chrom, self.snp_pos, seq[offset-5:offset+6], self.ref))
            exit()
        seq_ref = str(seq)
        seq_alt = seq_ref[:offset] + str(self.alt) + seq_ref[offset+1:]
        seq_ref = one_hot_encode(seq_ref)
        seq_alt = one_hot_encode(seq_alt)
        if self.strand == '-':
            seq_ref = seq_ref[::-1, ::-1].copy()
            seq_alt = seq_alt[::-1, ::-1].copy()
        return seq_ref, seq_alt


class SNPInterval(SNP):
    def __init__(self, chrom, strand, snp_pos, ref: str, alt: str,
                 center_posL, center_posR, center_pos=None) -> None:
        super().__init__(chrom, strand, snp_pos, ref, alt, center_pos)
        self.center_posL = center_posL
        self.center_posR = center_posR

    def parse(self, ref_fasta: Fasta, context_len: int):
        seq = ref_fasta.get_seq(
            self.chrom, self.center_posL-context_len, self.center_posR+context_len)
        offset = self.snp_pos - self.center_posL + context_len
        if self.ref is None:
            self.ref = str(seq[offset])
        if (len(self.ref) >= 1) and (len(self.ref) == len(self.alt)):
            if str(seq[offset:offset+(len(self.ref))]).upper() != str(self.ref).upper():
                print('WA: chr:{} pos:{} seq:{} ref:{}'.format(
                    self.chrom, offset, seq[offset], self.ref))
        # If reference and alternative sequence have different length,
        # then seq_ref and seq_alt will got different shape, which are unable to
        # be calculated concurrently
        seq_ref = str(seq)
        seq_alt = seq_ref[:offset] + \
            str(self.alt) + seq_ref[offset+len(self.ref):]
        seq_ref = one_hot_encode(seq_ref)
        seq_alt = one_hot_encode(seq_alt)

        if self.strand == '-':
            seq_ref = seq_ref[::-1, ::-1].copy()
            seq_alt = seq_alt[::-1, ::-1].copy()
        return seq_ref, seq_alt

    def align_score(self, ref_fasta: Fasta, ref_score, alt_score):
        """

        Parameters
        ---
        ref_fasta: Fasta
        ref_score: (N, dim) numpy.array
        alt_score: (N, dim) numpy.array

        Return
        ---
        result_ref_score : numpy.array
        result_alt_score : numpy.array
            Aligned score array. Padded with zero matrix at alternated postion
        """
        if self.strand == '-':
            ref_score = ref_score[::-1]
            alt_score = alt_score[::-1]
        offset = self.snp_pos - self.center_posL
        padding = np.zeros(
            (np.abs(len(self.ref)-len(self.alt)), ref_score.shape[1]))
        if len(self.ref) > len(self.alt):  # ref比alt长，需要填补alt序列
            alt_score = np.concatenate(
                [alt_score[:offset],
                 padding,
                 alt_score[offset:]],
                axis=0
            )
        elif len(self.ref) < len(self.alt):
            ref_score = np.concatenate(
                [ref_score[:offset],
                 padding,
                 ref_score[offset:]],
                axis=0
            )
        if self.strand == '-':
            ref_score = ref_score[::-1]
            alt_score = alt_score[::-1]
        return ref_score, alt_score

    def align_sequence(self, ref_fasta: Fasta):
        """
        如果Ref和Alt长度不同，在计算后它们的分数Shape也不同，这里添加一步手动对齐。
        返回对齐后的RNA序列，将长度改变的地方用‘N’来标记
        If the length of Ref and Alt are different, the shape of the calculated scores will also be different.
        Here, an additional manual alignment step is added to align the scores.

        Parameters
        ---
        ref_fasta: Fasta

        Return
        ---
        seq_ref : str
        seq_alt : str
        """
        seq = ref_fasta.get_seq(
            self.chrom, self.center_posL, self.center_posR)
        offset = self.snp_pos - self.center_posL
        if self.ref is None:
            self.ref = str(seq[offset])
        padding = 'N' * np.abs(len(self.ref)-len(self.alt))
        seq_ref = str(seq)
        seq_alt = seq_ref[:offset] + \
            str(self.alt) + seq_ref[offset+len(self.ref):]
        if len(self.ref) > len(self.alt):
            seq_alt = seq_alt[:offset] + padding + seq_alt[offset:]
        elif len(self.ref) < len(self.alt):
            seq_ref = seq_ref[:offset] + padding + seq_ref[offset:]
        return seq_ref, seq_alt


class SNPIntervalMutated(SNPInterval):
    def __init__(self, chrom, strand, snp_pos, ref: str, alt: str, center_posL, center_posR, center_pos=None,
                 pre_mutate_pos=None, pre_mutate_alt=None) -> None:
        super().__init__(chrom, strand, snp_pos, ref,
                         alt, center_posL, center_posR, center_pos)
        self.pre_mutate_pos = pre_mutate_pos
        self.pre_mutate_alt = pre_mutate_alt

    def parse(self, ref_fasta: Fasta, context_len: int):
        seq = ref_fasta.get_seq(
            self.chrom, self.center_posL-context_len, self.center_posR+context_len)
        offset = self.snp_pos - self.center_posL + context_len
        if self.ref is None:
            self.ref = str(seq[offset])
        if (len(self.ref) > 1) and (len(self.ref) == len(self.alt)):
            if str(seq[offset:offset+(len(self.ref))]).upper() != str(self.ref).upper():
                print('WA: chr:{} pos:{} seq:{} ref:{}'.format(
                    self.chrom, offset, seq[offset], self.ref))
        # If reference and alternative sequence have different length,
        # then seq_ref and seq_alt will got different shape, which are unable to
        # be calculated concurrently
        seq_ref = str(seq)
        # The sequence have difference with standard reference sequence
        if self.pre_mutate_pos:
            pre_offset = self.pre_mutate_pos - self.center_posL + context_len
            seq_ref = seq_ref[:pre_offset] + \
                str(self.pre_mutate_alt) + \
                seq_ref[pre_offset+len(self.pre_mutate_alt):]
        ##
        seq_alt = seq_ref[:offset] + \
            str(self.alt) + seq_ref[offset+len(self.ref):]
        seq_ref = one_hot_encode(seq_ref)
        seq_alt = one_hot_encode(seq_alt)
        if self.strand == '-':
            seq_ref = seq_ref[::-1, ::-1].copy()
            seq_alt = seq_alt[::-1, ::-1].copy()
        return seq_ref, seq_alt


class SpTransformerDriver(ModelDriver):
    def __init__(self, ref_fasta, load_db=True, context=4000) -> None:
        # reference genome
        if ref_fasta == 'hg19':
            ref_fasta = './data/data_package/hg19.fa'
            gtf = './data/data_package/hg19.annotation.gtf.gz'
        elif ref_fasta == 'hg38':
            ref_fasta = './data/data_package/hg38.fa'
            gtf = './data/data_package/hg38.annotation.gtf.gz'
        elif load_db:
            raise NotImplementedError
        if load_db:
            self.gtf = Genome(reference_name=ref_fasta,
                              annotation_name='gencode',
                              gtf_path_or_url=gtf)
        seed = 0
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = SpTransformer(128, context_len=context, tissue_num=15,
                              max_seq_len=8192, attn_depth=8, training=False)
        save_dict = torch.load(
            './model/weights/SpTransformer_pytorch.ckpt', map_location='cpu')
        model.load_state_dict(save_dict["state_dict"])
        models = [model]
        super().__init__(ref_fasta, models, load_db)

    def post_decorate(self, outputs):
        outputs[:, :3, :] = F.softmax(outputs[:, :3, :], dim=1)
        outputs[:, 3:, :] = torch.sigmoid(outputs[:, 3:, :])
        return outputs

    def calc_snps(self, snp_list, context_len):
        """
        Calculate splicing scores for input SNP.
        对于负链上的序列，计算完毕后按5'->3'方向输出分数。如果要画图要注意手动反转

        Parameters:
        ---
        snp_list: list of SNP

        Returns:
        ---
        result: numpy.array
        """
        input_list = []
        for i in range(len(snp_list)):
            ref_tensor, alt_tensor = snp_list[i].parse(
                self.ref_fasta, context_len)
            input_list += [ref_tensor, alt_tensor]
        input_list = np.array(input_list)
        result = self.calc_batched_sequence(input_list, encode=False)
        return result

    def calc_snp_misaligned(self, snp: SNPInterval, context_len, use_fasta=None):
        """
        对于负链上的序列，计算完毕后按5'->3'方向输出分数。如果要画图要注意手动反转
        """
        FASTA = use_fasta if use_fasta else self.ref_fasta
        ref_tensor, alt_tensor = snp.parse(FASTA, context_len)
        ref_score = self.calc_single_sequence(ref_tensor, encode=False)
        alt_score = self.calc_single_sequence(alt_tensor, encode=False)
        ref_score, alt_score = snp.align_score(
            FASTA,
            ref_score,
            alt_score
        )
        # print(ref_score.shape,alt_score.shape)
        return [ref_score, alt_score]


class SpliceAIRetrainedDriver(ModelDriver):
    def __init__(self, ref_fasta, load_db=True) -> None:
        # reference genome
        if ref_fasta == 'hg19':
            ref_fasta = '/public/home/shenninggroup/yny/data/hg19/hg19.fa'
            gtf = 'data/data_package/gencode.v19.annotation.gtf.gz'
        elif ref_fasta == 'hg38':
            ref_fasta = '/public/home/shenninggroup/yny/data/hg38/hg38.fa'
            gtf = '/public/home/shenninggroup/yny/data/hg38/gencode.v41.annotation.gtf'
        else:
            raise NotImplementedError
        if load_db:
            self.gtf = Genome(reference_name=ref_fasta,
                              annotation_name='gencode',
                              gtf_path_or_url=gtf)
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        model = SpliceAI.SpliceAI(
            SpliceAI.L, SpliceAI.W, SpliceAI.AR, CL=10000)
        save_dict = torch.load(
            '/public/home/shenninggroup/yny/code/CellSplice/model/weights/SpliceAI-retrained/train.ckpt')
        model.load_state_dict(save_dict["state_dict"])
        models = [model]
        super().__init__(ref_fasta, models)

    def post_decorate(self, outputs):
        outputs[:, :3, :] = F.softmax(outputs[:, :3, :], dim=1)
        usage = torch.ones((outputs.shape[0], 15, outputs.shape[2])) * \
            torch.max(outputs[:, 1:3, :], dim=1, keepdim=True)[0]
        outputs = torch.concat([outputs, usage], dim=1)
        return outputs

    def calc_snps(self, snp_list, context_len):
        """
        Calculate splicing scores for input SNP.
        For negative strand, the output scores are reversed from 5'->3'

        Parameters:
        ---
        snp_list: list of SNP

        Returns:
        ---
        result: numpy.array
        """
        input_list = []
        for i in range(len(snp_list)):
            ref_tensor, alt_tensor = snp_list[i].parse(
                self.ref_fasta, context_len)
            # print(ref_tensor.shape, alt_tensor.shape)
            input_list += [ref_tensor, alt_tensor]
        input_list = np.array(input_list)
        result = self.calc_batched_sequence(input_list, encode=False)
        return result

    def calc_snp_misaligned(self, snp: SNPInterval, context_len):
        """
        For negative strand, the output scores are reversed from 5'->3'
        """
        ref_tensor, alt_tensor = snp.parse(self.ref_fasta, context_len)
        ref_score = self.calc_single_sequence(ref_tensor, encode=False)
        alt_score = self.calc_single_sequence(alt_tensor, encode=False)
        ref_score, alt_score = snp.align_score(
            self.ref_fasta,
            ref_score,
            alt_score
        )
        return [ref_score, alt_score]


def annotate_VCF_general(fname, foutput_name, model_name='SpTransformer', input_CSV=False, sep='\t', column_dict={}, limit=False, ref_fasta='hg38', check_ref=True, temp_format=False):
    if model_name == 'SpTransformer':
        annotator = SpTransformerDriver(ref_fasta, load_db=True)
    elif model_name == 'SpliceAI':
        annotator = SpliceAIRetrainedDriver(ref_fasta, load_db=True)
    gtf = annotator.gtf

    CONTEXT_LEN = 4000
    tis_names = ['Adipose Tissue', 'Blood', 'Blood Vessel', 'Brain', 'Colon', 'Heart', 'Kidney',
                 'Liver', 'Lung', 'Muscle', 'Nerve', 'Small Intestine', 'Skin', 'Spleen', 'Stomach']
    df_output = pd.DataFrame(
        columns=['#CHROM', 'POS', 'ID', 'REF', 'ALT']+['dsplice']+tis_names)
    cnt = 0
    mode = 'w'
    ##
    if input_CSV:
        datafile = pd.read_csv(fname, delimiter=sep, header=0)
        print('in, {}'.format(datafile.shape))
        iter = datafile.iterrows()
    else:
        datafile = pyvcf.Reader(filename=fname, strict_whitespace=True)
        iter = datafile
    ##
    for record in tqdm.tqdm(iter, mininterval=5):
        if input_CSV:
            if temp_format:
                record = record[1]
                chrom = record['Chr']
                pos = record['Coordinate']
                ref, alt = record['Variant'].split('/')
                id = np.nan
            else:
                record = record[1]
                chrom = record[column_dict['chrom']]
                chrom = annotator.normalise_chrom(chrom, 'chr')
                pos = record[column_dict['pos']]
                ref = record[column_dict['ref']]
                alt = record[column_dict['alt']]
                id = record[column_dict['id']
                            ] if 'id' in column_dict else np.nan
        else:
            chrom = str(record.CHROM)
            if len(chrom) > 6:
                continue
            chrom = annotator.normalise_chrom(chrom, 'chr')
            pos = record.POS
            ref = record.REF
            id = record.ID
            alt = record.ALT[0]
        if (ref == '.') or (ref is None) or (len(ref) > 1000):
            ref = ''
        if (alt == '.') or (alt is None) or (len(ref) > 1000):
            alt = ''
        d_range = 100
        if pos <= CONTEXT_LEN:
            continue
        strand, _, _ = annotator.get_genes(gtf, chrom, pos)
        # print(chrom, pos, strand, ref, alt)
        splice_score = 0
        tissue_score = np.zeros((15,))
        for st in strand:
            snp = SNPInterval(
                chrom, st, pos, ref, alt, pos-d_range, pos+d_range, pos
            )
            score_ref, score_alt = annotator.calc_snps([snp], CONTEXT_LEN)
            # general
            alt_splice = np.max(score_alt[:, 1:3], axis=1)
            ref_splice = np.max(score_ref[:, 1:3], axis=1)
            tmp_splice = np.max(np.abs(alt_splice-ref_splice))
            splice_score = np.max([splice_score, tmp_splice])
            # tissue
            mask = ref_splice >= 0.09
            score_ref[:, 3:] *= mask.reshape(-1, 1)
            mask = alt_splice >= 0.09
            score_alt[:, 3:] *= mask.reshape(-1, 1)
            tissue_score = np.max([tissue_score, np.max(
                np.abs(score_alt[:, 3:]-score_ref[:, 3:]), axis=0)], axis=0)
        row = pd.Series({
            '#CHROM': chrom,
            'POS': pos,
            'ID': id,
            'REF': ref,
            'ALT': alt,
            'dsplice': splice_score
        })
        for k, tis_name in enumerate(tis_names):
            row[tis_name] = tissue_score[k]
        df_output.loc[cnt] = row
        cnt += 1
        if cnt % 1000 == 0:
            df_output.to_csv(foutput_name, sep='\t',
                             header=(mode == 'w'), mode=mode)
            mode = 'a'
            df_output.drop(df_output.index, inplace=True)
        if limit and (cnt >= limit):
            break
        pass
    if cnt % 1000 > 0:
        df_output.to_csv(foutput_name, sep='\t',
                         header=(mode == 'w'), mode=mode)


def calc_splice_site(chrom, site, annotator: ModelDriver, strand='+'):
    CONTEXT = 4000
    seq = annotator.ref_fasta.get_seq(
        chrom, site-CONTEXT, site+CONTEXT)
    seq_ref = str(seq)
    seq_ref = one_hot_encode(seq_ref)
    if strand == '-':
        seq_ref = seq_ref[::-1, ::-1].copy()
    score = annotator.calc_single_sequence(seq_ref)
    return score


def analysis_exon_inclusion(fname, use_method='SpTransformer'):
    datafile = pd.read_csv(fname, delimiter='\t')
    datafile = datafile[datafile['chr'].isin(
        ['chr1', 'chr3', 'chr5', 'chr7', 'chr9'])]
    if use_method == 'SpTransformer':
        annotator = SpliceAIRetrainedDriver('hg38', load_db=True)
    else:
        annotator = SpTransformerDriver('hg38', load_db=True)
    pred_list = []
    label_list = []
    for idx, row in tqdm.tqdm(datafile.iterrows(), mininterval=5):
        chrom = row['chr']
        strand = row['strand']
        site1 = row['start'] + 1
        site2 = row['end'] - 1
        label = row['exon_inclusion']
        # calc
        score1 = calc_splice_site(chrom, site1, annotator, strand)
        score2 = calc_splice_site(chrom, site2, annotator, strand)
        #
        pred = np.mean([score1.reshape(-1), score2.reshape(-1)], axis=0)
        pred_list.append(pred)
        label_list.append(label)
    output_name = 'exon_inclusion_sptransformer.npz'
    if use_method == 'retrained':
        output_name = 'exon_inclusion_spliceai_retrained.npz'
    elif use_method == True:
        output_name = 'exon_inclusion_spliceai.npz'

    np.savez(f'outputs/{output_name}', pred=pred_list, label=label_list)
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default=None)
    parser.add_argument('-m', '--model', type=str, default=None)
    parser.add_argument('--index', type=int, default=0)

    args = parser.parse_args()

    if args.task == 'ClinVar':
        if args.model == 'SpTransformer':
            annotate_VCF_general('/public/home/shenninggroup/yny/data/clinvar/clinvar_20220917.vcf',
                                 'outputs/Clinvar_SpTransformer_single7.tsv',
                                 model_name='SpTransformer',
                                 input_CSV=False,
                                 ref_fasta='hg38',
                                 check_ref=True)
        elif args.model == 'SpliceAI-retrained':
            annotate_VCF_general('/public/home/shenninggroup/yny/data/clinvar/clinvar_20220917.vcf',
                                 'outputs/Clinvar_SpliceAI_retrained.tsv',
                                 model_name='SpliceAI',
                                 input_CSV=False,
                                 ref_fasta='hg38',
                                 check_ref=True)

    if args.task == 'Kidney-DKD':
        annotate_VCF_general('/public/home/shenninggroup/jhy123/DKD/wes_dkd_processed/VQSR/QC/AC_DM_DN_DKD_filtered_QD_QUAL_DP10.vcf',
                             'outputs/DKD_0430_res.tsv',
                             model_name='SpTransformer',
                             input_CSV=True,
                             column_dict={
                                 'chrom': 'CHROM', 'ref': 'REF', 'alt': 'ALT', 'pos': 'POS'},
                             ref_fasta='hg38', check_ref=True)

    if args.task == 'Brain-ASC':
        annotate_VCF_general('/public/home/shenninggroup/yny/code/Splice-Pytorch/data/input/ASC_variants.tsv',
                             './outputs/brain_section/ASC_pred_all.tsv',
                             model_name='SpTransformer',
                             input_CSV=True, sep='\t',
                             column_dict={'chrom': 'chrom', 'ref': 'ref', 'alt': 'alt', 'pos': 'pos'}, ref_fasta='hg19', check_ref=True)

    if args.task == 'Brain-BIP-1':
        annotate_VCF_general('/public/home/shenninggroup/yny/tools/snpEff/BipEX_annotated_missense.vcf',
                             './outputs/brain_section/BipEx_pred_missense.tsv',
                             model_name='SpTransformer',
                             input_CSV=False, ref_fasta='hg38', check_ref=True)
        annotate_VCF_general('/public/home/shenninggroup/yny/tools/snpEff/BipEX_annotated_synonymous.vcf',
                             './outputs/brain_section/BipEx_pred_synonymous.tsv',
                             model_name='SpTransformer',
                             input_CSV=False, ref_fasta='hg38', check_ref=True)
    if args.task == 'Brain-BIP-2':
        annotate_VCF_general('/public/home/shenninggroup/yny/tools/snpEff/BipEX_annotated_intron.vcf',
                             'outputs/BipEx_filtered_intron.tsv',
                             model_name='SpTransformer',
                             input_CSV=False, ref_fasta='hg38', check_ref=True)
    if args.task == 'Brain-BIP-3':
        annotate_VCF_general('/public/home/shenninggroup/yny/tools/snpEff/BipEX_annotated_splice_acceptor.vcf',
                             'outputs/BipEx_filtered_splice_acceptor.tsv',
                             model_name='SpTransformer',
                             input_CSV=False, ref_fasta='hg38', check_ref=True)
        annotate_VCF_general('/public/home/shenninggroup/yny/tools/snpEff/BipEX_annotated_splice_donor.vcf',
                             'outputs/BipEx_filtered_splice_donor.tsv',
                             model_name='SpTransformer',
                             input_CSV=False, ref_fasta='hg38', check_ref=True)
        annotate_VCF_general('/public/home/shenninggroup/yny/tools/snpEff/BipEX_annotated_splice_region.vcf',
                             'outputs/BipEx_filtered_splice_region.tsv',
                             model_name='SpTransformer',
                             input_CSV=False, ref_fasta='hg38', check_ref=True)
        annotate_VCF_general('/public/home/shenninggroup/yny/tools/snpEff/BipEX_annotated_stop_gained.vcf',
                             'outputs/BipEx_filtered_stop_gained.tsv',
                             model_name='SpTransformer',
                             input_CSV=False, ref_fasta='hg38', check_ref=True)

    if args.task == 'Brain_SCHEMA':
        index = args.index
        finput = f'/public/home/shenninggroup/yny/code/Splice-Pytorch/data/input/SCHEMA_variants_part{index}.tsv'
        foutput = f'./outputs/brain_section/SCHEMA_pred_part{index}.tsv'
        annotate_VCF_general(finput,
                             foutput,
                             model_name='SpTransformer',
                             input_CSV=True, sep='\t',
                             column_dict={'chrom': 'chrom', 'ref': 'ref', 'alt': 'alt', 'pos': 'pos'}, ref_fasta='hg19', check_ref=True)

    if args.task == 'Tissue-Zscore':
        chrom_list = ['chr'+str(i) for i in range(22, 23)]
        for chrom in chrom_list:
            annotate_VCF_general(f'/public/home/shenninggroup/yny/code/CellSplice/output/tissue_spe_analysis/select_{chrom}.tsv',
                                 f'/public/home/shenninggroup/yny/code/CellSplice/output/tissue_spe_analysis/pred_{chrom}.tsv',
                                 input_CSV=True,
                                 column_dict={
                                     'chrom': 'chrom', 'ref': 'ref', 'alt': 'alt', 'pos': 'pos'},
                                 ref_fasta='hg38', check_ref=True)

    if args.task == 'Tissue-Zscore-Mut':
        index = args.index
        annotate_VCF_general(f'/public/home/shenninggroup/yny/code/CellSplice/outputs/mutagenesis/old_version/variant_{index}.csv',
                             f'/public/home/shenninggroup/yny/code/CellSplice/outputs/mutagenesis/old_version/variant_pred_{index}.csv',
                             input_CSV=True,
                             sep=',',
                             column_dict={
                                 'chrom': 'chrom', 'ref': 'ref', 'alt': 'alt', 'pos': 'pos'},
                             ref_fasta='hg38', check_ref=True)

    if args.task == 'Specific':
        index = args.index
        annotate_VCF_general(f'/public/home/shenninggroup/yny/code/CellSplice/output/site_loss_analysis/filter_result.tsv',
                             f'/public/home/shenninggroup/yny/code/CellSplice/output/site_loss_analysis/filter_prediction.tsv',
                             input_CSV=True,
                             sep='\t',
                             column_dict={
                                 'chrom': 'chrom', 'ref': 'ref', 'alt': 'alt', 'pos': 'pos'},
                             ref_fasta='hg38', check_ref=True)

    if args.task == 'Exon-inclusion':
        analysis_exon_inclusion(
            'outputs/exon_inclusion_15.csv', use_method='retrained')
        analysis_exon_inclusion(
            'outputs/exon_inclusion_15.csv', use_method='SpTransformer')
    ####

    if args.task == 'annotate_DMD_seq':
        finput = f'/public/home/shenninggroup/yny/code/ASO/data/parsed_ASO_table.tsv'
        foutput = f'/public/home/shenninggroup/yny/code/ASO/data/parsed_ASO_table_pred.tsv'
        annotate_VCF_general(finput,
                             foutput,
                             input_CSV=True,
                             sep='\t',
                             column_dict={
                                 'chrom': 'chrom', 'ref': 'ref', 'alt': 'alt', 'pos': 'pos'},
                             ref_fasta='hg38', check_ref=True)
    pass
