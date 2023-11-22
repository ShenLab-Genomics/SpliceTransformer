import pandas as pd
import numpy as np
from pyfaidx import Fasta
import torch
from torch.nn import functional as F
import argparse
import vcf as pyvcf
from model.model import SpTransformer
from pyensembl import Genome
import tqdm
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ''

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
        if (len(self.ref) >= 1):
            if str(seq[offset:offset+(len(self.ref))]).upper() != str(self.ref).upper():
                print('WA: chr:{} pos:{} seq:{} ref:{}'.format(
                    self.chrom, offset, seq[offset], self.ref))
                raise ValueError('Reference sequence not match')
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

    def align_score(self, ref_score, alt_score):
        """
        如果Ref和Alt长度不同，在计算后它们的分数Shape也不同，这里添加一步手动对齐

        Parameters
        ---
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


class ModelDriver:
    def __init__(self, models=None) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.models = [model.to(self.device).eval() for model in models]
        print('torch device:{}'.format(self.device))

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
        修饰模型输出。例如使用softmax激活函数处理。
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
                out = self.models[i](inputs, use_usage_head=True)
                out = self.post_decorate(out)
                output.append(out.cpu().detach().numpy())
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


class SpTransformerDriver(ModelDriver):
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # Load model
        model = SpTransformer(64, usage_head=11, context_len=4000)
        save_dict = torch.load(
            '/home/ningyuan/code/SpTransformer_server/src/SpTransformer/app/model/weights/SpTransformer_pytorch.ckpt',
            map_location=torch.device('cpu'))
        model.load_state_dict(save_dict["state_dict"])
        #
        models = [model]
        super().__init__(models)

    def post_decorate(self, outputs):
        outputs[:, :3, :] = F.softmax(outputs[:, :3, :], dim=1)
        outputs[:, 3:, :] = torch.sigmoid(outputs[:, 3:, :])
        return outputs

    def calc_snps(self, snp_list, context_len, ref_fasta):
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
                ref_fasta, context_len)
            input_list += [ref_tensor, alt_tensor]
        input_list = np.array(input_list)
        result = self.calc_batched_sequence(input_list, encode=False)
        return result

    def calc_snp_misaligned(self, snp: SNPInterval, context_len, ref_fasta):
        """
        对于负链上的序列，计算完毕后按5'->3'方向输出分数。如果要画图要注意手动反转
        """
        ref_tensor, alt_tensor = snp.parse(ref_fasta, context_len)
        ref_score = self.calc_single_sequence(ref_tensor, encode=False)
        alt_score = self.calc_single_sequence(alt_tensor, encode=False)
        ref_score, alt_score = snp.align_score(
            ref_score,
            alt_score
        )
        # print(ref_score.shape,alt_score.shape)
        return [ref_score, alt_score]


class Annotator():
    def __init__(self, limit_protein_coding=False) -> None:
        self.gtf = {}
        self.ref_fasta = {}
        import os
        # load hg19
        gtf_hg19 = 'data/data_package/gencode.v19.annotation.gtf.gz'
        if os.path.exists(gtf_hg19):
            self.gtf['hg19'] = Genome(reference_name='hg19',
                                      annotation_name='gencode.v19',
                                      gtf_path_or_url=gtf_hg19)
        else:
            print('Warning: gencode grch37 gtf file not found, You can download from https://www.gencodegenes.org/human/release_19.html')
        ref_fasta = '/home/ningyuan/data/hg19.fa'
        self.ref_fasta['hg19'] = Fasta(ref_fasta)

        # load hg38
        gtf_hg38 = 'data/data_package/gencode.v38.annotation.gtf.gz'
        if os.path.exists(gtf_hg38):
            self.gtf['hg38'] = Genome(reference_name='hg38',
                                      annotation_name='gencode.v38',
                                      gtf_path_or_url=gtf_hg38)
        else:
            print('Warning: gencode grch38 gtf file not found, You can download from https://www.gencodegenes.org/human/release_38.html')
        ref_fasta = '/home/ningyuan/data/hg38.fa'
        self.ref_fasta['hg38'] = Fasta(ref_fasta)

        # load model
        self.model = SpTransformerDriver()

        # tissue thresholds
        self._tissue_list = ['Adipose Tissue', 'Muscle', 'Blood Vessel',
                             'Brain', 'Kidney', 'Heart', 'Liver', 'Lung', 'Skin', 'Nerve']
        self._threshold = {
            '10tissue_z01_std': {'Adipose Tissue': 0.0154579235, 'Muscle': 0.022955596, 'Blood Vessel': 0.013255985, 'Brain': 0.02150287, 'Kidney': 0.015781531, 'Heart': 0.020252606, 'Liver': 0.03839425, 'Lung': 0.02268705, 'Skin': 0.014823442, 'Nerve': 0.021076491},
            '10tissue_z01_mean': {'Adipose Tissue': 0.009927585, 'Muscle': -0.01940719, 'Blood Vessel': 0.0031568026, 'Brain': -0.0018135239, 'Kidney': 0.009485917, 'Heart': -0.014798415, 'Liver': -0.02471026, 'Lung': 0.018273583, 'Skin': 0.004983168, 'Nerve': 0.014902338},
            '95_10tissue_z01': {'Adipose Tissue': 1.7556416988372803, 'Muscle': 1.5593988776206968, 'Blood Vessel': 1.7310052573680874, 'Brain': 1.6463595330715173, 'Kidney': 1.6878403544425964, 'Heart': 1.8116924464702602, 'Liver': 1.6761198997497555, 'Lung': 1.5648080170154572, 'Skin': 1.77454976439476, 'Nerve': 1.6953504621982574},
            '95_10tissue_z01_negative': {'Adipose Tissue': -1.5660359859466553, 'Muscle': -1.616789847612381, 'Blood Vessel': -1.5676331043243408, 'Brain': -1.6601818740367889, 'Kidney': -1.4874303817749024, 'Heart': -1.5314733326435088, 'Liver': -1.6708995878696442, 'Lung': -1.6792091071605681, 'Skin': -1.5444809913635253, 'Nerve': -1.6566802144050599},
        }

        # other config
        self.limit_protein_coding = limit_protein_coding
        pass

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
            # name = gene["gene_name"]
            '''
            te = 0
            ts = 2147483647
            for exon in gtf.children(gene, featuretype="exon"):
                te = max(te, exon.end)
                ts = min(ts, exon.start)
            '''
            strand.append(gene.strand)
            # start.append(ts)
            # end.append(te)
        return strand, start, end

    def query_scores(self, chrom, pos, ref,
                     alt, ref_genome='hg38'):
        # split input string
        strands, _, _ = self.get_genes(self.gtf[ref_genome], chrom, pos)
        # calculate score
        splice_score = 0
        tissue_score = np.zeros((11,))
        print(strands)
        for strand in strands:
            snp = SNPInterval(chrom, strand, int(pos), ref, alt,
                              int(pos)-100, int(pos)+100)
            score_ref, score_alt = self.model.calc_snp_misaligned(
                snp, 4000, self.ref_fasta[ref_genome])
            # general splicing
            alt_splice = np.max(score_alt[:, 1:3], axis=1)
            ref_splice = np.max(score_ref[:, 1:3], axis=1)
            tmp_splice = np.max(np.abs(alt_splice-ref_splice))
            splice_score = np.max([splice_score, tmp_splice])
            # tissue usage
            mask = ref_splice >= 0.1
            # mask by splicing score
            score_ref[:, 3:] *= mask.reshape(-1, 1)
            mask = alt_splice >= 0.1
            score_alt[:, 3:] *= mask.reshape(-1, 1)
            tissue_score = np.max([tissue_score, np.max(
                np.abs(score_alt[:, 3:]-score_ref[:, 3:]), axis=0)], axis=0)
        # annotate tissue specificity
        flags = []
        for idx, tis in enumerate(self._tissue_list):
            flag = ((tissue_score[idx] - np.mean(tissue_score)) -
                    self._threshold['10tissue_z01_mean'][tis])/self._threshold['10tissue_z01_std'][tis]
            flag = 'Y' if flag > self._threshold['95_10tissue_z01'][tis] else 'N'
            flags.append(flag)
        return splice_score, flags

    def query_detail_scores(self, snv, ref_genome='hg38'):
        raise NotImplementedError

    def annotate_variant_table(self, fname='./clinvar.vcf', foutput_name='./clinvar_result.vcf', ref_genome='hg38'):
        tis_names = ['Adipose Tissue', 'Muscle', 'Blood Vessel', 'Brain', 'Kidney', 'Heart',
                     'Liver', 'Lung', 'Skin', 'Nerve']
        df_output = pd.DataFrame(
            columns=['#CHROM', 'POS', 'ID', 'REF', 'ALT']+['score']+tis_names)

        datafile = pyvcf.Reader(filename=fname, strict_whitespace=True)
        _iter = datafile
        cnt = 0
        mode = 'w'
        for record in tqdm.tqdm(_iter, mininterval=5):
            chrom = str(record.CHROM)
            if len(chrom) > 6:
                continue
            pos = record.POS
            ref = record.REF
            id = record.ID
            alt = record.ALT[0]
            if (ref == '.') or (ref is None):
                ref = ''
            if (alt == '.') or (alt is None):
                alt = ''
            if (len(str(ref)) != 1) or (len(str(alt)) != 1):
                continue
            splice_score, tissue_flag = self.query_scores(chrom, pos, ref,
                                                          alt, ref_genome=ref_genome)
            row = pd.Series({
                '#CHROM': chrom,
                'POS': pos,
                'ID': id,
                'REF': ref,
                'ALT': alt,
                'score': '{:.2f}'.format(splice_score)
            })
            for k, tis_name in enumerate(tis_names):
                row[tis_name] = tissue_flag[k]
            df_output.loc[cnt] = row
            cnt += 1
            if cnt % 1000 == 0:
                df_output.to_csv(foutput_name, sep='\t',
                                 header=(mode == 'w'), mode=mode)
                mode = 'a'
                df_output.drop(df_output.index, inplace=True)
                # cnt = 0
            limit = 50
            if limit and (cnt >= limit):
                break
            pass
        if cnt % 1000 > 0:
            df_output.to_csv(foutput_name, sep='\t',
                             header=(mode == 'w'), mode=mode)


if __name__ == '__main__':
    #
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--input', type=str,
                        default='data/example/input38.vcf')
    parser.add_argument('-O', '--output', type=str,
                        default='data/example/input38.vcf')
    parser.add_argument('--reference', type=str, default='hg38')
    parser.add_argument('--vcf', type=bool, default=True,
                        choices=[True, False])
    parser.add_argument('--protein_coding', type=bool, default=False, choices=[True, False],
                        help='Only consider about protein coding genes, default: False', action="store_true")
    #
    args = parser.parse_args()
    finput = args.input
    foutput = args.output
    ref_genome = args.reference
    annotator = Annotator()
    annotator.annotate_variant_table(
        fname=finput, foutput_name=foutput, ref_genome=ref_genome)
    # TODO: implement annotator for .csv
