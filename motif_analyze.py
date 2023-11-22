import pandas as pd
import numpy as np
from pyfaidx import Fasta
import torch
from torch.nn import functional as F
import tqdm
import gffutils
import logging

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
    def __init__(self, ref_fasta, models=None) -> None:
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
        print(f'Loaded fasta from {ref_fasta}')
        # load model to target device
        self.models = [model.to(self.device).eval() for model in models]

        pass

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
        if (len(self.ref) > 1) and (len(self.ref) == len(self.alt)):
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
        如果Ref和Alt长度不同，在计算后它们的分数Shape也不同，这里添加一步手动对齐

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
    def __init__(self, ref_fasta) -> None:
        # reference genome
        if ref_fasta == 'hg19':
            db = './data/genome_annotation/gencode.v37.annotation.gtf.gz.db'
            # self.gtf = gffutils.FeatureDB(db)
            ref_fasta = '/home/ningyuan/data/hg19.fa'
        elif ref_fasta == 'hg38':
            db = './data/genome_annotation/gencode.v38.annotation.gtf.gz.db'
            # self.gtf = gffutils.FeatureDB(db)
            ref_fasta = '/home/ningyuan/data/hg38.fa'
        else:
            raise NotImplementedError
        from model.model import SpTransformer
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        model = SpTransformer(64, usage_head=11, context_len=4000)
        save_dict = torch.load(
            './model/weights/SpTransformer_pytorch.ckpt')
        model.load_state_dict(save_dict["state_dict"])
        models = [model]
        super().__init__(ref_fasta, models)

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

    def calc_snp_misaligned(self, snp: SNPInterval, context_len):
        """
        对于负链上的序列，计算完毕后按5'->3'方向输出分数。如果要画图要注意手动反转
        """
        ref_tensor, alt_tensor = snp.parse(self.ref_fasta, context_len)
        ref_score = self.calc_single_sequence(ref_tensor, encode=False)
        alt_score = self.calc_single_sequence(alt_tensor, encode=False)
        ref_score, alt_score = snp.align_score(
            self.ref_fasta,
            ref_score,
            alt_score
        )
        # print(ref_score.shape,alt_score.shape)
        return [ref_score, alt_score]

    # Applications

    def test_cover_splicing(self, chrom, strand, posL, posR, interested_site,
                            pre_mutate_pos, pre_mutate_alt,
                            previous_sites=[]):
        chrom = self.normalise_chrom(chrom)
        COVER_LEN = 11
        cover_score_list = []
        for maskL in range(posL, posR):
            maskR = maskL + COVER_LEN
            snp = SNPIntervalMutated(
                chrom, strand, maskL, 'A'*COVER_LEN, '',
                posL, posR,
                center_pos=interested_site,
                pre_mutate_pos=pre_mutate_pos,
                pre_mutate_alt=pre_mutate_alt
            )

            pred_ref, pred_alt = self.calc_snp_misaligned(
                snp, context_len=4000)
            seq_ref, seq_alt = snp.align_sequence(self.ref_fasta)

            cover_score = np.sum(pred_alt[:, 1:3])
            cover_score_list.append(cover_score)
            print(cover_score)

        from matplotlib import pyplot as plt
        import matplotlib.lines as lines
        fig = plt.figure(figsize=(5, 2), dpi=150, constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1)
        plt.title(f'{chrom} {posL} - {posR} Previous mutation:{pre_mutate_pos} change to {pre_mutate_alt}',
                  size=8)

        f = open('./outputs/recover/filtered_seq.txt', 'w')
        for idx, maskL in enumerate(range(posL, posR)):
            maskR = maskL + COVER_LEN
            ax.add_line(lines.Line2D([maskL-posL, maskR - 1 - posL], [cover_score_list[idx], cover_score_list[idx]],
                                     solid_capstyle='butt', solid_joinstyle='miter',
                                     linewidth=0.5, alpha=0.7,
                                     color='#D89C7A',
                                     antialiased=False))
            if cover_score_list[idx] < 0.5:
                f.write(f'>{idx}\n')
                rep = {'a': 'u',
                       'g': 'c',
                       'c': 'g',
                       't': 'a'
                       }
                output_seq = str(seq_ref[maskL-posL:maskR-posL])[::-1]
                output_seq = list(output_seq)
                for i_str in range(len(output_seq)):
                    output_seq[i_str] = rep[output_seq[i_str]]
                output_seq = ''.join(output_seq)
                f.write(output_seq)
                f.write('\n')
        f.close()
        # plt.bar(np.arange(len(cover_score_list)),cover_score_list,color='red',label='Cover score')

        if len(previous_sites) > 0:
            plt.scatter(np.array(previous_sites)-posL, np.ones(len(previous_sites))
                        * 0.5, s=5, c='black', label='previous splice sites')
        plt.xticks(np.arange(pred_ref.shape[0]), str(seq_ref), size=6)
        plt.ylabel('Splicing strength', size=7)
        plt.legend(fontsize=7)
        fig.savefig('./outputs/recover/testcover.jpg', dpi=150)

    def vis_recover_splicing(self, chrom, strand, posL, posR, interested_site, pre_mutate_pos, pre_mutate_alt):
        chrom = self.normalise_chrom(chrom)
        snp = SNPIntervalMutated(
            chrom, strand, interested_site+3, 'AAAAAAA', '',
            posL, posR,
            center_pos=interested_site,
            pre_mutate_pos=pre_mutate_pos,
            pre_mutate_alt=pre_mutate_alt
        )

        pred_ref, pred_alt = self.calc_snp_misaligned(snp, context_len=4000)
        seq_ref, seq_alt = snp.align_sequence(self.ref_fasta)
        print(pred_ref.shape)
        print(pred_alt.shape)
        if snp.strand == '-':
            pred_ref = pred_ref[::-1, :]
            pred_alt = pred_alt[::-1, :]

        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=(5, 2), dpi=150, constrained_layout=True)

        # previous
        plt.subplot(2, 1, 1)
        plt.bar(np.arange(pred_ref.shape[0]),
                pred_ref[:, 1], color='red', label='Acceptor')
        plt.bar(np.arange(pred_ref.shape[0]),
                pred_ref[:, 2], color='blue', label='Donor')
        plt.scatter(interested_site - posL, 0.5, s=5, marker='v', c='black')
        plt.scatter(pre_mutate_pos - posL, 0.5, s=5, marker='v', c='black')
        plt.ylabel('Reference', size=7)
        plt.ylim(0.0, 1.0)
        plt.xticks(np.arange(pred_ref.shape[0]), str(seq_ref), size=6)

        # after
        plt.subplot(2, 1, 2)
        plt.bar(np.arange(pred_alt.shape[0]),
                pred_alt[:, 1], color='red', label='Acceptor')
        plt.bar(np.arange(pred_alt.shape[0]),
                pred_alt[:, 2], color='blue', label='Donor')
        plt.ylabel('Alternative', size=7)
        plt.ylim(0.0, 1.0)
        plt.xticks(np.arange(pred_ref.shape[0]), str(seq_alt), size=6)

        plt.legend(fontsize=7)

        fig.savefig('./outputs/recover/test.jpg', dpi=150)
        plt.close(fig)

    def test_ASO_sequence(self):
        DATA_FILE = './outputs/ASO/Database_ASO_0829.parsed.csv'
        df = pd.read_csv(DATA_FILE, sep=',')
        df['offset'] = df['offset'].astype(int)
        cover_score_list = []
        pred_acc_list = []
        pred_don_list = []
        for idx, row in tqdm.tqdm(df.iterrows()):
            chrom = row['chrom']
            offset = row['offset']
            length = len(row['Sequence'])
            strand = row['strand']
            if strand == 1:
                strand = '+'
            elif strand == -1:
                strand = '-'
            #
            snp = SNPIntervalMutated(
                chrom, strand, offset, 'A'*length, '',
                offset-100, offset+100,
                center_pos=offset
            )
            #
            pred_ref, pred_alt = self.calc_snp_misaligned(
                snp, context_len=4000)
            seq_ref, seq_alt = snp.align_sequence(self.ref_fasta)

            pred_acc_list.append(np.max(pred_alt[:, 1]))
            pred_don_list.append(np.max(pred_alt[:, 2]))
            pos = pred_ref[:, 1] > 0.1
            cover_score = np.sum(pred_alt[pos, 1] - pred_ref[pos, 1])
            pos = pred_ref[:, 2] > 0.1
            cover_score += np.sum(pred_alt[pos, 2] - pred_ref[pos, 2])
            cover_score_list.append(cover_score)
        df['score'] = cover_score_list
        df['pred_acc'] = pred_acc_list
        df['pred_don'] = pred_don_list
        df.to_csv('outputs/ASO/ASO_pred.csv', sep=',')
        pass


if __name__ == '__main__':
    ref_fasta = '/home/ningyuan/data/hg38.fa'
    annotator = SpTransformerDriver('hg19')
    TASK = 1
    if TASK == 0:
        annotator.test_cover_splicing(
            'X',
            '-',
            100654710,
            100654800,
            100654732,
            100654735,
            'T',
            previous_sites=[100654732, 100654788]
        )
    if TASK == 1:
        annotator.test_ASO_sequence()
