import numpy as np

nucs = list('ACGT')
alphabet = nucs.copy()
nuc_to_vec = {
    nuc: [
        1 if current_nuc is nuc else 0
        for current_nuc
        in nucs
    ]
    for nuc
    in nucs
}
nuc_to_vec['N'] = np.ones(len(nucs)).astype(int).tolist()
nuc_to_vec['_'] = np.zeros(len(nucs)).astype(int).tolist()
vec_to_nuc = {tuple(v):k for k, v in nuc_to_vec.items()}
nucs.append('N')
nucs.append('_')

def masked_seq_to_upper(seq):
    return seq.upper()

def masked_seq_to_n_mask(seq, alphabet = alphabet):
    return ''.join(
        [
            nuc if nuc in alphabet else 'N'
            for nuc
            in seq
        ]
    )

def masked_seq_to_zero_mask(seq, alphabet = alphabet):
    return ''.join(
        [
            nuc if nuc in alphabet else '_'
            for nuc
            in seq
        ]
    )

def get_degenerate_pct(seq, alphabet = alphabet):
    return (
        100.0 *
        sum(
            [
                1
                for nuc
                in seq
                if nuc not in alphabet
            ]
        ) /
        len(seq)
    )

convert_masked_seq_dict = {
    'upper': masked_seq_to_upper,
    'n': masked_seq_to_n_mask,
    '_': masked_seq_to_zero_mask
}

def seq_to_mat(
    seq,
    nuc_to_vec = nuc_to_vec,
    convert_masked_seq = convert_masked_seq_dict['n']
):
    return [
            nuc_to_vec[nuc]
            for nuc
            in convert_masked_seq(seq)
    ]

def mat_to_seq(mat, vec_to_nuc = vec_to_nuc):
    return ''.join(
        [
            vec_to_nuc[tuple(vec)]
            for vec
            in list(mat)
        ]
    )
