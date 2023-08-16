import pandas as pd
import numpy as np
import torch
import argparse
import vcf as pyvcf
import tqdm
from annotator_pytorch import Annotator, get_gene, get_genes, initialization

def visualize_snp(_chr, _snp_pos, _strand, _ref, _alt, _mutation_result='', use_spliceai=False, save_folder=None):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    annotator, _, gtf = initialization(
        ref_fasta='hg38', grch='grch38', use_spliceai=use_spliceai, context_len=2000)
    chr = _chr
    strand = _strand
    snp_pos = _snp_pos
    ref = _ref
    alt = _alt
    print(f'{chr} {strand} {snp_pos} {ref} {alt}')
    mutation_result = _mutation_result
    offset = 0
    d_range = 400
    d_L = snp_pos - d_range + offset
    d_R = snp_pos + d_range + offset
    _, exon_starts, exon_ends = get_gene(gtf, chr, snp_pos)
    pred_alt, pred_ref = annotator.calc_one_snp_interval(
        chr, strand, d_L, d_R, snp_pos, ref, alt, context_len=2000)
    if strand == '-':
        pred_alt = pred_alt[::-1, :]
        print(pred_alt.shape)
        pred_ref = pred_ref[::-1, :]
    if len(ref) != len(alt):
        fill_part = np.zeros((len(ref), 14))
        print(pred_alt.shape)
        print(pred_ref.shape)
        pred_alt = np.concatenate(
            [pred_alt[:snp_pos-d_L], fill_part, pred_alt[snp_pos-d_L:]], axis=0)
        print(pred_alt.shape)
        print(pred_ref.shape)

    # score_splice = np.max(
        # np.abs(pred_alt[:, 1:3] - pred_ref[:, 1:3]), axis=(1))
    if not use_spliceai:
        alt_splice = np.max(pred_alt[:, 1:3], axis=1)
        ref_splice = np.max(pred_ref[:, 1:3], axis=1)
        mask = ref_splice >= 0.1
        pred_ref[:, 3:] *= mask.reshape(-1, 1)
        mask = alt_splice >= 0.1
        pred_alt[:, 3:] *= mask.reshape(-1, 1)
        delta_splice_score = np.max(np.abs(pred_alt[:, 1:3]-pred_ref[:, 1:3]))
    sequence = annotator.ref_fasta.get_seq(
        chr, d_L, d_R)
    fig = plt.figure(figsize=(30, 15))
    plt.suptitle('{}:{}-{}  score:{}'.format(chr, d_L,
                                             d_R, delta_splice_score), fontsize=36)
    if use_spliceai:
        # ref
        plt.subplot(2, 1, 1)
        plt.bar(x=range(len(sequence)),
                height=pred_ref[:, 1], color='red', alpha=0.5, label='Acceptor')
        plt.bar(x=range(len(sequence)),
                height=pred_ref[:, 2], color='blue', alpha=0.5, label='Donor')
        sites = [x for x in range(len(sequence))
                 if ((d_L+x) in (exon_starts + exon_ends))]
        plt.scatter(x=sites, y=[0.5 for x in sites],
                    color='black', label='Annotated Exon')
        # alt
        plt.subplot(2, 1, 2)
        plt.bar(x=range(len(sequence)),
                height=pred_alt[:, 1], color='red', alpha=0.5)
        plt.bar(x=range(len(sequence)),
                height=pred_alt[:, 2], color='blue', alpha=0.5)
        plt.scatter(x=sites, y=[0.5 for x in sites],
                    color='black', label='Annotated Exon')
        plt.savefig(
            'outputs/visual/pltfig_{}_{}_spliceai.jpg'.format('TTN', str(snp_pos)))
    else:
        # ref
        plt.subplot(4, 1, 1)
        plt.ylabel('ref sequence', fontsize=24)
        # mask = np.any((np.max(pred_ref[:, 1:3], axis=1) >= 0.5, np.max(
        #     pred_alt[:, 1:3], axis=1) >= 0.5), axis=0)
        # pred_ref[:, 3:] *= mask.reshape(-1, 1)
        # pred_alt[:, 3:] *= mask.reshape(-1, 1)
        plt.bar(x=range(len(sequence)),
                height=pred_ref[:, 7], width=2, color='grey', alpha=0.8, label='Kidney Usage')
        plt.bar(x=range(len(sequence)),
                height=pred_ref[:, 1], width=2, color='red', alpha=0.5, label='Acceptor')
        plt.bar(x=range(len(sequence)),
                height=pred_ref[:, 2], width=2, color='blue', alpha=0.5, label='Donor')
        sites = [x for x in range(len(sequence))
                 if ((d_L+x) in (exon_starts + exon_ends))]
        plt.scatter(x=sites, y=[0.5 for x in sites],
                    color='black', label='Annotated Exon')
        # plt.scatter(x=[snp_pos-d_L], y=[0.4],
        #             color='red', label='snp position')
        # plt.xticks(range(len(sequence)), sequence, size=20)
        plt.yticks(size=20)
        plt.ylim((0, 1))
        plt.xlim((-1, len(sequence)+1))
        plt.legend(fontsize=18)
        # alt
        plt.subplot(4, 1, 2)
        plt.xlim((-1, len(sequence)+1))
        plt.ylim((0, 1))
        plt.yticks(size=20)
        sequence = list(str(sequence))
        for kk in range(snp_pos-d_L, snp_pos-d_L+len(ref)-len(alt)):
            sequence[kk] = ' '
        # plt.xticks(range(len(sequence)), sequence, size=20)
        plt.xticks(size=16)
        plt.ylabel('Alt sequence', fontsize=24)
        plt.bar(x=range(len(sequence)),
                height=pred_alt[:, 7], width=2, color='grey', alpha=0.8, label='Kidney Usage')
        plt.bar(x=range(len(sequence)),
                height=pred_alt[:, 1], width=2, color='red', alpha=0.5)
        plt.bar(x=range(len(sequence)),
                height=pred_alt[:, 2], width=2, color='blue', alpha=0.5)
        plt.scatter(x=sites, y=[0.5 for x in sites],
                    color='black', label='Annotated Exon')
        tis_names = ['Adipose Tissue', 'Muscle', 'Blood Vessel', 'Brain', 'Kidney', 'Heart',
                     'Liver', 'Lung', 'Skin', 'Nerve', 'Testis']
        print(pred_alt[d_range+1, :])
        print(pred_ref[d_range+1, :])
        print(np.max(np.abs(pred_alt - pred_ref), axis=0))
        # delta adipose
        if save_folder:
            plt.savefig(f'{save_folder}/{mutation_result}_{chr}_{snp_pos}.jpg')
        else:
            plt.savefig(
                'outputs/visual/pltfig_{}_{}.jpg'.format('TTN', str(snp_pos)))
    plt.close(fig)
    npz_path = 'outputs/visual/pred_data_{}_{}.npz'.format('DKD', str(snp_pos))
    if save_folder:
        npz_path = f'{save_folder}/{mutation_result}_{chr}_{snp_pos}.npz'
    np.savez(npz_path,
             pred_ref=pred_ref,
             pred_alt=pred_alt,
             sequence=np.array(str(sequence), dtype=object),
             exon_starts=exon_starts,
             exon_ends=exon_ends,
             d_L=d_L,
             d_R=d_R)
    return


def visualize_single_site(_chr, _snp_pos, _strand, _ref, _alt, _mutation_result='', save_folder=None):
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    annotator, _, gtf = initialization(
        ref_fasta='hg38', grch='grch38', use_spliceai=False, context_len=4000)
    chrom = _chr
    strand = _strand
    snp_pos = _snp_pos
    ref = _ref
    alt = _alt
    print(f'{chrom} {strand} {snp_pos} {ref} {alt}')
    mutation_result = _mutation_result
    offset = 0
    d_range = 400
    d_L = snp_pos - d_range + offset
    d_R = snp_pos + d_range + offset
    _, exon_starts, exon_ends = get_gene(gtf, chrom, snp_pos)
    pred_alt, pred_ref = annotator.calc_one_snp_interval(
        chrom, strand, d_L, d_R, snp_pos, ref, alt, context_len=4000)
    if strand == '-':
        pred_alt = pred_alt[::-1, :]
        print(pred_alt.shape)
        pred_ref = pred_ref[::-1, :]
    if len(ref) != len(alt):
        fill_part = np.zeros((len(ref), 14))
        print(pred_alt.shape)
        print(pred_ref.shape)
        pred_alt = np.concatenate(
            [pred_alt[:snp_pos-d_L], fill_part, pred_alt[snp_pos-d_L:]], axis=0)
        print(pred_alt.shape)
        print(pred_ref.shape)

    alt_splice = np.max(pred_alt[:, 1:3], axis=1)
    ref_splice = np.max(pred_ref[:, 1:3], axis=1)
    mask = ref_splice >= 0.1
    pred_ref[:, 3:] *= mask.reshape(-1, 1)
    mask = alt_splice >= 0.1
    pred_alt[:, 3:] *= mask.reshape(-1, 1)
    delta_splice_score = np.max(np.abs(pred_alt[:, 1:3]-pred_ref[:, 1:3]))

    # kidney usage at column#7
    sequence = annotator.ref_fasta.get_seq(
        chrom, d_L, d_R)
    fig = plt.figure(figsize=(30, 15))

    plt.suptitle('{}:{}-{}  score:{}'.format(chrom, d_L,
                                             d_R, delta_splice_score), fontsize=36)
    # ref
    tissue_list = ['Adipose Tissue', 'Muscle', 'Blood Vessel', 'Brain', 'Kidney', 'Heart',
                   'Liver', 'Lung', 'Skin', 'Nerve']
    for idx, tis in enumerate(tissue_list):
        plt.subplot(12, 1, idx+1)
        plt.ylabel(tis, fontsize=8)
        plt.bar(x=range(len(sequence)),
                height=(pred_ref[:, idx+3] - np.mean(pred_ref[:, 3:-1], axis=1))/0.1, width=2, color='grey', alpha=0.8, label=f'{tis} Usage')
        plt.ylim((-0.3, 0.3))
        plt.xlim((-1, len(sequence)+1))
        plt.axhline(y=0, xmin=-1, xmax=len(sequence))
    plt.subplot(12, 1, 11)
    plt.bar(x=range(len(sequence)),
            height=pred_ref[:, 1], width=2, color='red', alpha=0.5, label='Acceptor')
    plt.bar(x=range(len(sequence)),
            height=pred_ref[:, 2], width=2, color='blue', alpha=0.5, label='Donor')
    sites = [x for x in range(len(sequence))
             if ((d_L+x) in (exon_starts + exon_ends))]
    plt.scatter(x=sites, y=[0.5 for x in sites],
                color='black', label='Annotated Exon')
    plt.yticks(size=20)
    plt.ylim((0, 1))
    plt.xlim((-1, len(sequence)+1))
    plt.legend(fontsize=10)

    if save_folder:
        plt.savefig(f'{save_folder}/{mutation_result}_{chrom}_{snp_pos}.jpg')
    plt.close(fig)
    npz_path = 'outputs/visual/pred_data_{}_{}.npz'.format(chrom, str(snp_pos))
    if save_folder:
        npz_path = f'{save_folder}/{mutation_result}_{chrom}_{snp_pos}.npz'
        np.savez(npz_path,
                 pred_ref=pred_ref,
                 pred_alt=pred_alt,
                 sequence=np.array(str(sequence), dtype=object),
                 exon_starts=exon_starts,
                 exon_ends=exon_ends,
                 d_L=d_L,
                 d_R=d_R)
    return


def annotate_VCF_general(fname, foutput_name, input_CSV=False, sep='\t', column_dict={}, limit=False, ref_fasta='hg38', check_ref=True, temp_format=False):
    annotator, _, gtf = initialization(
        ref_fasta=ref_fasta, grch='grch38', use_spliceai=False)

    CONTEXT_LEN = 4000
    # writer = pyvcf.Writer(output_file, datafile)
    tis_names = ['Adipose Tissue', 'Muscle', 'Blood Vessel', 'Brain', 'Kidney', 'Heart',
                 'Liver', 'Lung', 'Skin', 'Nerve', 'Testis']
    df_output = pd.DataFrame(
        columns=['#CHROM', 'POS', 'ID', 'REF', 'ALT']+['score']+tis_names)
    cnt = 0
    mode = 'w'
    ##
    if input_CSV:
        datafile = pd.read_csv(fname, delimiter=sep, header=0)
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
            # chrom = 'chrom'+str(record.CHROM)
            chrom = annotator.normalise_chrom(chrom, 'chr')
            pos = record.POS
            ref = record.REF
            id = record.ID
            alt = record.ALT[0]
        if (ref == '.') or (ref is None):
            ref = ''
        if (alt == '.') or (alt is None):
            alt = ''
        print(chrom, pos, f'ref:{ref}', f'alt:{alt}')
        # TODO:
        d_range = 100
        # if pos != 85665628:
        #     continue
        if (len(str(ref)) > 1):
            continue
        if (len(str(alt)) > 1):
            continue
        if (len(str(ref)) == 0) and (len(str(alt)) == 0):
            continue
        strand, _, _ = get_genes(gtf, chrom, pos)
        print(chrom, pos)
        print(strand)
        splice_score = 0
        tissue_score = np.zeros((11,))
        for st in strand:
            score_alt, score_ref = annotator.calc_one_snp_interval(
                chrom, st, pos-d_range, pos+d_range, pos, ref, alt, CONTEXT_LEN, check_ref=check_ref)
            d_L = pos - d_range
            if len(ref) > len(alt):
                fill_part = np.zeros((len(ref)-len(alt), 14))
                score_alt = np.concatenate(
                    [score_alt[:pos-d_L], fill_part, score_alt[pos-d_L:]], axis=0)
            if len(ref) < len(alt):
                fill_part = np.zeros((len(alt)-len(ref), 14))
                score_ref = np.concatenate(
                    [score_ref[:pos-d_L], fill_part, score_ref[pos-d_L:]], axis=0)
            # general
            alt_splice = np.max(score_alt[:, 1:3], axis=1)
            ref_splice = np.max(score_ref[:, 1:3], axis=1)
            tmp_splice = np.max(np.abs(alt_splice-ref_splice))
            splice_score = np.max([splice_score, tmp_splice])
            # tissue
            mask = ref_splice >= 0.1
            score_ref[:, 3:] *= mask.reshape(-1, 1)
            mask = alt_splice >= 0.1
            score_alt[:, 3:] *= mask.reshape(-1, 1)
            tissue_score = np.max([tissue_score, np.max(
                np.abs(score_alt[:, 3:]-score_ref[:, 3:]), axis=0)], axis=0)
        row = pd.Series({
            '#CHROM': chrom,
            'POS': pos,
            'ID': id,
            'REF': ref,
            'ALT': alt,
            'score': splice_score
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
            # cnt = 0
        if limit and (cnt >= limit):
            break
        pass
    if cnt % 1000 > 0:
        df_output.to_csv(foutput_name, sep='\t',
                         header=(mode == 'w'), mode=mode)


def ASCOT_tissue_motif_visualization(fname, id_filter, output_name='ASCOT_result_motif_100_0404.tsv',
                                     output_mutation_score='ASCOT_result_motif_100_diff_0404.npz'
                                     ):
    def get_exon_bound(code: str):
        L = code.find(':')
        R = code.find('-', L)
        return str(code[:L]), int(code[L+1:R]), int(code[R+1:])

    def calc_mutagenesis(annotator: Annotator, seq: str, chrom, strand, L, R, target: int):
        score_list = []
        print(L, R)
        for i in range(L, R):
            # if i != target-3:
            #     continue
            # if i - L != 80:
            #     continue
            tmp_score = []
            for alt in ['A', 'G', 'C', 'T']:
                # print(seq[i-L])
                pred_alt, pred_ref = annotator.calc_one_snp(
                    chrom, strand, target, i, seq[i-L], alt, context_len=4000)
                alt_splice = np.max(pred_alt[:, 1:3], axis=1)
                ref_splice = np.max(pred_ref[:, 1:3], axis=1)
                mask = ref_splice >= 0.1
                pred_ref[:, 3:] *= mask.reshape(-1, 1)
                mask = alt_splice >= 0.1
                pred_alt[:, 3:] *= mask.reshape(-1, 1)
                tmp_score.append(pred_alt)
            tmp_score = np.mean(tmp_score, axis=0)
            # print(pred_ref)
            score_list.append(pred_ref-tmp_score)
        return score_list

    gtex = pd.read_csv(fname, delimiter=',')
    if id_filter is not None:
        id_filter = pd.read_csv(id_filter, delimiter='\t')
        gtex = gtex[gtex['exon_id'].isin(id_filter['exon_id'])]
    cassette = gtex[gtex.cassette_exon == 'Yes']
    print(cassette.shape)
    annotator, _, gtf = initialization(
        ref_fasta='hg38', grch='grch38', use_spliceai=False)
    tissue_list = ['Adipose Tissue', 'Muscle', 'Blood Vessel', 'Brain', 'Kidney', 'Heart',
                   'Liver', 'Lung', 'Skin', 'Nerve', 'Testis']
    tissue_score = []
    exon_id = []
    cnt = 0
    seq_list_acc = []
    seq_list_don = []
    seq_mutation_score_acceptor = []
    seq_mutation_score_donor = []
    siteA_list = []
    siteD_list = []
    chrom_list = []
    for idx, row in cassette.iterrows():
        cnt += 1
        # if cnt > 50:
        #     break
        # print(idx)
        chrom, site1, site2 = get_exon_bound(row['exon_location'])
        ###
        # if row['exon_id'] not in ['GT_54867']:
        #     continue
        ###
        strand = row['exon_strand']
        # if strand == '-':
        #     continue
        if strand == '+':
            sequence_acc = annotator.ref_fasta.get_seq(
                chrom, site1-80, site1+20)
            sequence_don = annotator.ref_fasta.get_seq(
                chrom, site2-20+1, site2+80+1)
            site_acc = site1
            site_don = site2
        else:
            sequence_don = annotator.ref_fasta.get_seq(
                chrom, site1-80-1, site1+20-1, rc=False)
            sequence_acc = annotator.ref_fasta.get_seq(
                chrom, site2-20, site2+80, rc=False)
            site_acc = site2
            site_don = site1
        seq_list_acc.append(str(sequence_acc).upper())
        seq_list_don.append(str(sequence_don).upper())
        # only consider about + stand
        # acceptor
        if strand == '+':
            score_list = calc_mutagenesis(annotator, str(
                sequence_acc).upper(), chrom, strand, site_acc-80, site_acc+20, site_acc)
            seq_mutation_score_acceptor.append(score_list)
        else:
            score_list = calc_mutagenesis(annotator, str(
                sequence_acc).upper(), chrom, strand, site_acc-20, site_acc+80, site_acc)
            seq_mutation_score_acceptor.append(score_list)
        # break
        # donor
        if strand == '+':
            score_list = calc_mutagenesis(annotator, str(
                sequence_don).upper(), chrom, strand, site_don-20+1, site_don+80+1, site_don)
            seq_mutation_score_donor.append(score_list)
        else:
            score_list = calc_mutagenesis(annotator, str(
                sequence_don).upper(), chrom, strand, site_don-80-1, site_don+20-1, site_don)
            seq_mutation_score_donor.append(score_list)
        # meta data
        score1 = annotator.calc_one_site(chrom, strand, site1)
        # print(score1)
        score2 = annotator.calc_one_site(chrom, strand, site2)
        siteA_list.append(site_acc)
        siteD_list.append(site_don)
        chrom_list.append(chrom)
        # print(score1.shape)
        exon_id.append(row['exon_id'])
        tissue_score.append(
            np.mean([score1.reshape(-1), score2.reshape(-1)], axis=0))
        # if cnt > 20:
        #     break
    tissue_score = np.array(tissue_score)
    # print(tissue_score.shape)
    output_columns = ['Neither', 'Acceptor', 'Donor']
    output_columns += tissue_list
    df = pd.DataFrame(tissue_score[:, :], columns=output_columns)
    df['exon_id'] = exon_id
    df['acceptor_pos'] = siteA_list
    df['acceptor_seq'] = seq_list_acc
    df['chrom'] = chrom_list
    df['donor_pos'] = siteD_list
    df['donor_seq'] = seq_list_don

    np.savez(output_mutation_score, acc=np.array(
        seq_mutation_score_acceptor), don=np.array(seq_mutation_score_donor))
    df.to_csv(output_name, sep='\t')


def ASCOT_tissue_motif_calc_0712(fname, output_name='ASCOT_result_motif_100_0707.tsv',
                                 output_mutation_score='ASCOT_result_motif_100_diff_0707.npz'
                                 ):
    def get_exon_bound(code: str):
        L = code.find(':')
        R = code.find('-', L)
        return str(code[:L]), int(code[L+1:R]), int(code[R+1:])

    def calc_mutagenesis(annotator: Annotator, seq: str, chrom, strand, L, R, target: int):
        score_list = []
        for i in range(L, R):
            tmp_score = []
            for alt in ['A', 'G', 'C', 'T']:
                pred_alt, pred_ref = annotator.calc_one_snp(
                    chrom, strand, target, i, seq[i-L], alt, context_len=4000)
                alt_splice = np.max(pred_alt[:, 1:3], axis=1)
                ref_splice = np.max(pred_ref[:, 1:3], axis=1)
                mask = ref_splice >= 0.1
                pred_ref[:, 3:] *= mask.reshape(-1, 1)
                mask = alt_splice >= 0.1
                pred_alt[:, 3:] *= mask.reshape(-1, 1)
                tmp_score.append(pred_alt)
            tmp_score = np.mean(tmp_score, axis=0)
            # print(tmp_score)
            score_list.append(pred_ref-tmp_score)
        return score_list

    gtex = pd.read_csv(fname, delimiter=',')
    annotator, _, gtf = initialization(
        ref_fasta='hg38', grch='grch38', use_spliceai=False)
    tissue_list = ['Adipose Tissue', 'Muscle', 'Blood Vessel', 'Brain', 'Kidney', 'Heart',
                   'Liver', 'Lung', 'Skin', 'Nerve', 'Testis']
    tissue_score = []
    exon_id = []
    cnt = 0
    seq_list_acc = []
    seq_list_don = []
    seq_mutation_score_acceptor = []
    seq_mutation_score_donor = []
    siteA_list = []
    siteD_list = []
    chrom_list = []
    for idx, row in tqdm.tqdm(gtex.iterrows()):
        cnt += 1
        print(idx)
        chrom, site1, site2 = get_exon_bound(row['exon_location'])
        strand = row['exon_strand']
        if strand == '-':
            continue
        if strand == '+':
            sequence_acc = annotator.ref_fasta.get_seq(
                chrom, site1-150, site1+50)
            sequence_don = annotator.ref_fasta.get_seq(
                chrom, site2-50+1, site2+150+1)
            site_acc = site1
            site_don = site2
        else:
            sequence_don = annotator.ref_fasta.get_seq(
                chrom, site1-150-1, site1+50-1, rc=True)
            sequence_acc = annotator.ref_fasta.get_seq(
                chrom, site2-50, site2+150, rc=True)
            site_acc = site2
            site_don = site1
        seq_list_acc.append(str(sequence_acc).upper())
        seq_list_don.append(str(sequence_don).upper())
        # only consider about + stand
        # acceptor
        score_list = calc_mutagenesis(annotator, str(
            sequence_acc).upper(), chrom, strand, site_acc-150, site_acc+50, site_acc)
        seq_mutation_score_acceptor.append(score_list)
        # donor
        score_list = calc_mutagenesis(annotator, str(
            sequence_don).upper(), chrom, strand, site_don-50+1, site_don+150+1, site_don)
        seq_mutation_score_donor.append(score_list)
        # meta data
        score1 = annotator.calc_one_site(chrom, strand, site1)
        score2 = annotator.calc_one_site(chrom, strand, site2)
        siteA_list.append(site_acc)
        siteD_list.append(site_don)
        chrom_list.append(chrom)
        exon_id.append(row['exon_id'])
        tissue_score.append(
            np.mean([score1.reshape(-1), score2.reshape(-1)], axis=0))
    tissue_score = np.array(tissue_score)
    output_columns = ['Neither', 'Acceptor', 'Donor']
    output_columns += tissue_list
    df = pd.DataFrame(tissue_score[:, :], columns=output_columns)
    df['exon_id'] = exon_id
    df['acceptor_pos'] = siteA_list
    df['acceptor_seq'] = seq_list_acc
    df['chrom'] = chrom_list
    df['donor_pos'] = siteD_list
    df['donor_seq'] = seq_list_don

    np.savez(output_mutation_score, acc=np.array(
        seq_mutation_score_acceptor), don=np.array(seq_mutation_score_donor))
    df.to_csv(output_name, sep='\t')


def gtex_bias_test2(fname, fout_csv, fout_npz):
    def get_exon_bound(code: str):
        L = code.find(':')
        R = code.find('-', L)
        return str(code[:L]), int(code[L+1:R]), int(code[R+1:])

    def calc_mutagenesis(annotator: Annotator, seq: str, chrom, strand, L, R):
        score_list = []
        for i in range(L, R):
            # tmp_score = []
            for alt in ['A', 'G', 'C', 'T']:
                if seq[i-L].upper() == alt:
                    continue
                d_range = 100
                pred_alt, pred_ref = annotator.calc_one_snp_interval(
                    chrom, strand, i-d_range, i+d_range+1, i, seq[i-L], alt, context_len=4000)
                alt_splice = np.max(pred_alt[:, 1:3], axis=1)
                ref_splice = np.max(pred_ref[:, 1:3], axis=1)
                mask = ref_splice >= 0.1
                pred_ref[:, 3:] *= mask.reshape(-1, 1)
                mask = alt_splice >= 0.1
                pred_alt[:, 3:] *= mask.reshape(-1, 1)
                delta_score = np.max(
                    np.abs(pred_alt[:, :]-pred_ref[:, :]), axis=0)
                score_list.append(delta_score)
        score_list = np.array(score_list)
        return score_list

    gtex = pd.read_csv(fname, delimiter=',')
    annotator, _, gtf = initialization(
        ref_fasta='hg38', grch='grch38', use_spliceai=False)
    tissue_list = ['Adipose Tissue', 'Muscle', 'Blood Vessel', 'Brain', 'Kidney', 'Heart',
                   'Liver', 'Lung', 'Skin', 'Nerve', 'Testis']
    tissue_score = []
    cnt = 0
    seq_mutation_score = []
    chrom_list = []
    for idx, row in gtex.iterrows():
        cnt += 1
        # if cnt > 5:
        #     break
        chrom = row['#CHROM']
        chrom_list.append(chrom)
        site1 = row['POS']
        strand = row['strand']
        sequence_acc = annotator.ref_fasta.get_seq(
            chrom, site1-50, site1+50+1)

        # acceptor
        score_list = calc_mutagenesis(annotator, str(
            sequence_acc).upper(), chrom, strand, site1-50, site1+50+1)
        seq_mutation_score.append(score_list)
        # meta data
        score1 = annotator.calc_one_site(chrom, strand, site1)
        tissue_score.append(score1.reshape(-1))
    tissue_score = np.array(tissue_score)
    output_columns = ['Neither', 'Acceptor', 'Donor']
    output_columns += tissue_list
    df = pd.DataFrame(tissue_score[:, :], columns=output_columns)
    df['chrom'] = chrom_list
    output_name = fout_csv
    output_mutation_score = fout_npz
    np.savez(output_mutation_score, acc=np.array(seq_mutation_score))
    df.to_csv(output_name, sep='\t')


if __name__ == '__main__':
    import sys
    target = sys.argv[1]

    if target == 'ClinVar':
        annotate_VCF_general('/public/home/shenninggroup/yny/data/clinvar/clinvar_20220917.vcf',
                             'outputs/clinvar_annotate_0319.csv')

    if target == 'temporary':  # A toy task to test the program
        annotate_VCF_general('data/example/input38.vcf',
                             'data/example/output38.vcf', input_CSV=False, ref_fasta='hg38')

    if target == 'site_vis':
        visualize_single_site('chr1', 7745000, '+', 'C', 'T',
                              '', save_folder='./outputs/site_visual')

    # 　This task generates a figure to show model scores in a region around a SNP
    if target == 'Region_visualization':
        # visualize_snp('chr2', 179404310, '-',
        #               'T', 'C', '', use_spliceai=True)
        # visualize_snp('chr3', 133948892, '-',
        #               'C', 'T', '')
        visualize_snp('chr2', 165307938, '+',
                      'G', 'A', '')

    if target == 'ASCOT_motif_vis_0428':
        ASCOT_tissue_motif_visualization('data/ASCOT/gtex_psi.csv',
                                         'data/ASCOT/filter_zscore_0428.csv',
                                         'ASCOT_result_motif_100_0428.tsv',
                                         'ASCOT_result_motif_100_diff_0428.npz')

    if target == 'ASCOT_motif_vis_0712_p1':
        ASCOT_tissue_motif_calc_0712('outputs/visual_motif/random_sample/ascot_psi_select_p1.csv',
                                     'outputs/visual_motif/random_sample/ASCOT_result_motif_100_0712_p1.tsv',
                                     'outputs/visual_motif/random_sample/ASCOT_result_motif_100_diff_0712_p1.npz')
    if target == 'ASCOT_motif_vis_0712_p2':
        ASCOT_tissue_motif_calc_0712('outputs/visual_motif/random_sample/ascot_psi_select_p2.csv',
                                     'outputs/visual_motif/random_sample/ASCOT_result_motif_100_0712_p2.tsv',
                                     'outputs/visual_motif/random_sample/ASCOT_result_motif_100_diff_0712_p2.npz')

    if target == 'GTEx_bias_0505_p1':  # to perform the random mutagenesis test
        gtex_bias_test2('data/gtex_bias/random_consensus_sites_0427_p1.csv',
                        'gtex_insilico_100_0427_p1.tsv', 'gtex_insilico_100_0505_1.npz')
    if target == 'GTEx_bias_0505_p2':
        gtex_bias_test2('data/gtex_bias/random_consensus_sites_0427_p2.csv',
                        'gtex_insilico_100_0427_p2.tsv', 'gtex_insilico_100_0505_2.npz')

    if target == 'annotate_DKD':
        annotate_VCF_general('data/DKD/variant_dp_above_10.txt',
                             'outputs/DKD_0505_res.tsv', input_CSV=True, sep='\t',
                             column_dict={'chrom': '#CHROM', 'ref': 'REF', 'alt': 'ALT', 'pos': 'POS'}, ref_fasta='hg38', check_ref=True)

    if target == 'annotate_ASC':
        annotate_VCF_general('./data/input/ASC_variants.tsv',
                             './outputs/brain_section/ASC_pred.tsv', input_CSV=True, sep='\t',
                             column_dict={'chrom': 'chrom', 'ref': 'ref', 'alt': 'alt', 'pos': 'pos'}, ref_fasta='hg19', check_ref=True)

    if target == 'BipEx_intron':
        annotate_VCF_general('/public/home/shenninggroup/yny/tools/snpEff/BipEX_annotated_intron.vcf',
                             'outputs/BipEx_filtered_intron.tsv', input_CSV=False, ref_fasta='hg38', check_ref=True)
    if target == 'BipEx_splice':
        annotate_VCF_general('/public/home/shenninggroup/yny/tools/snpEff/BipEX_annotated_splice_acceptor.vcf',
                             'outputs/BipEx_filtered_splice_acceptor.tsv', input_CSV=False, ref_fasta='hg38', check_ref=True)
        annotate_VCF_general('/public/home/shenninggroup/yny/tools/snpEff/BipEX_annotated_splice_donor.vcf',
                             'outputs/BipEx_filtered_splice_donor.tsv', input_CSV=False, ref_fasta='hg38', check_ref=True)
        annotate_VCF_general('/public/home/shenninggroup/yny/tools/snpEff/BipEX_annotated_splice_region.vcf',
                             'outputs/BipEx_filtered_splice_region.tsv', input_CSV=False, ref_fasta='hg38', check_ref=True)
        annotate_VCF_general('/public/home/shenninggroup/yny/tools/snpEff/BipEX_annotated_stop_gained.vcf',
                             'outputs/BipEx_filtered_stop_gained.tsv', input_CSV=False, ref_fasta='hg38', check_ref=True)

    if target == 'annotate_BIP_missense':
        annotate_VCF_general('/public/home/shenninggroup/yny/tools/snpEff/BipEX_annotated_missense.vcf',
                             './outputs/brain_section/BipEx_pred_missense.tsv', input_CSV=False, ref_fasta='hg38', check_ref=True)

    if target == 'annotate_BIP_synonymous':
        annotate_VCF_general('/public/home/shenninggroup/yny/tools/snpEff/BipEX_annotated_synonymous.vcf',
                             './outputs/brain_section/BipEx_pred_synonymous.tsv', input_CSV=False, ref_fasta='hg38', check_ref=True)

    if target == 'annotate_SCHEMA':
        index = sys.argv[2]
        finput = f'./data/input/SCHEMA_variants_part{index}.tsv'
        foutput = f'./outputs/brain_section/SCHEMA_pred_part{index}.tsv'
        annotate_VCF_general(finput,
                             foutput, input_CSV=True, sep='\t',
                             column_dict={'chrom': 'chrom', 'ref': 'ref', 'alt': 'alt', 'pos': 'pos'}, ref_fasta='hg19', check_ref=True)
    pass
