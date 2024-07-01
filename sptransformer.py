import pandas as pd
import numpy as np
from pyfaidx import Fasta
import argparse
import vcf as pyvcf
from pyensembl import Genome
import tqdm
import os
from tasks_annotate_mutations import SpTransformerDriver, annotate_VCF_general, SNPInterval


class Annotator():
    def __init__(self, limit_protein_coding=False) -> None:
        self.gtf = {}
        self.ref_fasta = {}
        gtf_hg19 = 'data/data_package/hg19.annotation.gtf.gz'
        if os.path.exists(gtf_hg19):
            self.gtf['hg19'] = Genome(reference_name='hg19',
                                      annotation_name='gencode.v19',
                                      gtf_path_or_url=gtf_hg19)
        else:
            print('Warning: gencode grch37 gtf file not found, You can download from https://www.gencodegenes.org/human/release_19.html')
            print('Please ignore this warning if you are using hg38')
        ref_fasta = 'data/data_package/hg19.fa'
        try:
            self.ref_fasta['hg19'] = Fasta(ref_fasta)
        except Exception as e:
            print('hg19 fasta not found')

        # load hg38
        gtf_hg38 = 'data/data_package/hg38.annotation.gtf.gz'
        if os.path.exists(gtf_hg38):
            self.gtf['hg38'] = Genome(reference_name='hg38',
                                      annotation_name='gencode.v38',
                                      gtf_path_or_url=gtf_hg38)
        else:
            print('Warning: gencode grch38 gtf file not found, You can download from https://www.gencodegenes.org/human/release_38.html')
            print('Please ignore this warning if you are using hg19')
        ref_fasta = 'data/data_package/hg38.fa'
        try:
            self.ref_fasta['hg38'] = Fasta(ref_fasta)
        except Exception as e:
            print('hg38 fasta not found')

        # load model
        self.model = SpTransformerDriver(ref_fasta='', load_db=False)

        # tissue thresholds
        # self._tissue_list = ['Adipose Tissue', 'Muscle', 'Blood Vessel',
        #                      'Brain', 'Kidney', 'Heart', 'Liver', 'Lung', 'Skin', 'Nerve']
        self._tissue_list = ['Adipose Tissue', 'Blood', 'Blood Vessel', 'Brain', 'Colon', 'Heart', 'Kidney',
                             'Liver', 'Lung', 'Muscle', 'Nerve', 'Small Intestine', 'Skin', 'Spleen', 'Stomach']
        self._threshold = {
            # '10tissue_z01_std': {'Adipose Tissue': 0.0154579235, 'Muscle': 0.022955596, 'Blood Vessel': 0.013255985, 'Brain': 0.02150287, 'Kidney': 0.015781531, 'Heart': 0.020252606, 'Liver': 0.03839425, 'Lung': 0.02268705, 'Skin': 0.014823442, 'Nerve': 0.021076491},
            # '10tissue_z01_mean': {'Adipose Tissue': 0.009927585, 'Muscle': -0.01940719, 'Blood Vessel': 0.0031568026, 'Brain': -0.0018135239, 'Kidney': 0.009485917, 'Heart': -0.014798415, 'Liver': -0.02471026, 'Lung': 0.018273583, 'Skin': 0.004983168, 'Nerve': 0.014902338},
            # '95_10tissue_z01': {'Adipose Tissue': 1.7556416988372803, 'Muscle': 1.5593988776206968, 'Blood Vessel': 1.7310052573680874, 'Brain': 1.6463595330715173, 'Kidney': 1.6878403544425964, 'Heart': 1.8116924464702602, 'Liver': 1.6761198997497555, 'Lung': 1.5648080170154572, 'Skin': 1.77454976439476, 'Nerve': 1.6953504621982574},
            # '95_10tissue_z01_negative': {'Adipose Tissue': -1.5660359859466553, 'Muscle': -1.616789847612381, 'Blood Vessel': -1.5676331043243408, 'Brain': -1.6601818740367889, 'Kidney': -1.4874303817749024, 'Heart': -1.5314733326435088, 'Liver': -1.6708995878696442, 'Lung': -1.6792091071605681, 'Skin': -1.5444809913635253, 'Nerve': -1.6566802144050599},
            '15tissue_z01_mean': {'Adipose Tissue': 0.016170055925538048, 'Blood': -0.1466492029720345, 'Blood Vessel': -0.004814086515620099, 'Brain': 0.017831358519558433, 'Colon': 0.03297392759250731, 'Heart': -0.0238562463866956, 'Kidney': 0.03762999484050797, 'Liver': -0.057356272980532315, 'Lung': 0.04777555561381407, 'Muscle': -0.04901297259546376, 'Nerve': 0.03865417911254773, 'Small Intestine': 0.05855564392852844, 'Skin': 0.008105610846240991, 'Spleen': 0.001179770193512726, 'Stomach': 0.022812684877590292},
            '15tissue_z01_std': {'Adipose Tissue': 0.015315100552740593, 'Blood': 0.07765363341706003, 'Blood Vessel': 0.02180354730921543, 'Brain': 0.03280497132872753, 'Colon': 0.018426896200775304, 'Heart': 0.02340998308282473, 'Kidney': 0.022809397819461737, 'Liver': 0.037399009450944404, 'Lung': 0.02457642947897365, 'Muscle': 0.03512151853661846, 'Nerve': 0.025045859953448327, 'Small Intestine': 0.03187442362925183, 'Skin': 0.015884291924608274, 'Spleen': 0.022923747771506405, 'Stomach': 0.01608487821592824},
            '99_15tissue_z01': {'Adipose Tissue': 2.110004782861747, 'Blood': 1.7809029776617185, 'Blood Vessel': 2.1050887688017026, 'Brain': 3.0658170588148344, 'Colon': 2.1240355946850107, 'Heart': 2.2147014853714153, 'Kidney': 2.469709880305002, 'Liver': 1.9371925110340946, 'Lung': 2.6983168423100232, 'Muscle': 1.929355057239116, 'Nerve': 1.9773509247022485, 'Small Intestine': 2.4951760642102356, 'Skin': 2.2288419624404043, 'Spleen': 2.416199967079524, 'Stomach': 2.423878178762257},
            '95_15tissue_z01': {'Adipose Tissue': 1.8050256807163338, 'Blood': 1.2976271940160122, 'Blood Vessel': 1.7301640366913926, 'Brain': 1.5711727792118446, 'Colon': 1.6529818774325018, 'Heart': 1.5579432960092778, 'Kidney': 1.9147687989032427, 'Liver': 1.4824945358487465, 'Lung': 1.7212837086526112, 'Muscle': 1.4469248306980924, 'Nerve': 1.5899046195758109, 'Small Intestine': 1.8459915073972142, 'Skin': 1.838922268820402, 'Spleen': 1.709814212823414, 'Stomach': 1.8189643807065778},
        }

        # other config
        self.limit_protein_coding = limit_protein_coding
        pass

    def get_genes(self, gtf: Genome, chrom, pos):
        # lookup gene and strands from gtf database
        # genes = gtf.region((chrom, pos, pos), featuretype="gene")
        chrom = self.model.normalise_chrom(chrom, 'chr')
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
            # start.append(ts)
            # end.append(te)
        if len(strand) > 1 and strand[0] == '-' and strand[1] == '+':
            strand = ['+', '-']
        return strand, start, end

    def query_scores(self, chrom, pos, ref,
                     alt, ref_genome='hg38', output_raw=False):
        # split input string
        strands, _, _ = self.get_genes(self.gtf[ref_genome], chrom, pos)
        # calculate score
        splice_score = 0
        tissue_score = np.zeros((15,))
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
            mask = ref_splice >= 0.09
            # mask by splicing score
            score_ref[:, 3:] *= mask.reshape(-1, 1)
            mask = alt_splice >= 0.09
            score_alt[:, 3:] *= mask.reshape(-1, 1)
            tissue_score = np.max([tissue_score, np.max(
                np.abs(score_alt[:, 3:]-score_ref[:, 3:]), axis=0)], axis=0)
        # annotate tissue specificity
        flags = []
        for idx, tis in enumerate(self._tissue_list):
            if output_raw:
                flags.append(tissue_score[idx])
            else:
                flag = ((tissue_score[idx] - np.mean(tissue_score)) -
                        self._threshold['15tissue_z01_mean'][tis])/self._threshold['15tissue_z01_std'][tis]
                flag = 'Y' if flag > self._threshold['95_15tissue_z01'][tis] else 'N'
                flags.append(flag)
        return splice_score, flags

    def query_detail_scores(self, snv, ref_genome='hg38'):
        raise NotImplementedError

    def annotate_variant_table(self,
                               fname='./clinvar.vcf',
                               foutput_name='./clinvar_result.vcf',
                               ref_genome='hg38',
                               input_CSV=False,
                               sep=',',
                               column_dict={},
                               output_raw=False):
        tis_names = ['Adipose Tissue', 'Blood', 'Blood Vessel', 'Brain', 'Colon', 'Heart', 'Kidney',
                     'Liver', 'Lung', 'Muscle', 'Nerve', 'Small Intestine', 'Skin', 'Spleen', 'Stomach']
        df_output = pd.DataFrame(
            columns=['#CHROM', 'POS', 'ID', 'REF', 'ALT']+['score']+tis_names)

        try:
            if input_CSV:
                print(f'Reading CSV file:{fname}')
                datafile = pd.read_csv(fname, delimiter=sep, header=0)
                print('in, {}'.format(datafile.shape))
                _iter = datafile.iterrows()
            else:
                print(f'Reading vcf file:{fname}')
                datafile = pyvcf.Reader(filename=fname, strict_whitespace=True)
                _iter = datafile
        except Exception as e:
            print(e)
            print('Error reading file: {}'.format(e))
            return

        cnt = 0
        mode = 'w'
        for record in tqdm.tqdm(_iter, mininterval=5):
            if not input_CSV:
                # read data from vcf file
                chrom = str(record.CHROM)
                if len(chrom) > 6:
                    continue
                pos = record.POS
                ref = record.REF
                id = record.ID
                alt = record.ALT[0]
            else:
                # read data from csv file
                record = record[1]
                chrom = record[column_dict['chrom']]
                chrom = self.model.normalise_chrom(chrom, 'chr')
                pos = record[column_dict['pos']]
                ref = record[column_dict['ref']]
                alt = record[column_dict['alt']]
                id = record[column_dict['id']
                            ] if 'id' in column_dict else np.nan

            if (ref is None) or (ref == '*') or (ref == '.'):
                ref = ''
            if (alt is None) or (alt == '.') or (alt == '*'):
                alt = ''
            if (len(str(ref)) > 1) or (len(str(alt)) > 1):  # skip indels
                continue
            splice_score, tissue_flag = self.query_scores(chrom, pos, ref,
                                                          alt, ref_genome=ref_genome, output_raw=output_raw)

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
                df_output.to_csv(foutput_name, sep=',',
                                 header=(mode == 'w'), mode=mode)
                mode = 'a'
                df_output.drop(df_output.index, inplace=True)
            limit = False
            if limit and (cnt >= limit):
                break
            pass
        if cnt % 1000 > 0:
            df_output.to_csv(foutput_name, sep=',',
                             header=(mode == 'w'), mode=mode)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--input', type=str,
                        default='data/example/input38.vcf')
    parser.add_argument('-O', '--output', type=str,
                        default='data/example/output38.csv')
    parser.add_argument('--reference', type=str, default='hg38',
                        choices=['hg19', 'hg38'],
                        help='Reference genome version')
    parser.add_argument('--vcf', type=bool, default=True,
                        choices=[True, False])
    parser.add_argument('--raw_score', type=bool, default=False,
                        help='Output raw score for each tissue')
    #
    args = parser.parse_args()
    finput = args.input
    foutput = args.output
    ref_genome = args.reference
    output_raw = args.raw_score
    input_CSV = not args.vcf

    annotator = Annotator()
    if input_CSV:
        # The .csv file should have at least the following columns: 'CHROM', 'POS', 'REF', 'ALT'
        # The column names can be different, but the user should provide a dictionary to map the column names to the standard names
        # annotate_VCF_general(finput, foutput, model_name='SpTransformer', input_CSV=True, ref_fasta=ref_genome, check_ref=True,
        #                      column_dict={'CHROM': '#CHROM', 'POS': 'POS', 'REF': 'REF', 'ALT': 'ALT'}, sep=',')
        annotator.annotate_variant_table(fname=finput, foutput_name=foutput, ref_genome=ref_genome,
                                         input_CSV=True, sep=',', column_dict={'chrom': 'CHROM', 'pos': 'POS', 'ref': 'REF', 'alt': 'ALT'}, output_raw=output_raw)
    else:
        # annotate_VCF_general(finput, foutput, model_name='SpTransformer',
        #                      input_CSV=False, ref_fasta=ref_genome, check_ref=True)
        annotator.annotate_variant_table(fname=finput, foutput_name=foutput, ref_genome=ref_genome,
                                         input_CSV=False, output_raw=output_raw)
