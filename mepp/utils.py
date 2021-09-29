import os

import pandas as pd
import numpy as np
import tensorflow as tf

from os.path import normpath

from tensorflow.math import reduce_sum

from tensorflow.keras import (
    Input,
    Model
)

from tqdm.auto import tqdm

from joblib import Parallel, delayed

from statsmodels.stats.multitest import multipletests

from .onehot_dna import (
    nuc_to_vec,
    convert_masked_seq_dict,
    seq_to_mat
)

from .onehot_dna import (
    get_degenerate_pct
)

def force_cpu_only():
    tf.config.set_visible_devices([], 'GPU')

def manage_gpu_memory(device_index = None):
    physical_devices = tf.config.list_physical_devices('GPU')
    if device_index is None:
        device_indices = range(len(physical_devices))
    else:
        device_indices = [device_index]
    for idx in device_indices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[idx], True)
        except:
            pass

def scored_fasta_dicts_to_df(
    sequence_dict,
    score_dict,
    description_dict
):
    return pd.DataFrame(
        data = [
            (
                sequence_id,
                sequence,
                score_dict[sequence_id],
                description_dict[sequence_id]
            )
            for sequence_id, sequence
            in sequence_dict.items()
        ],
        columns = [
            'sequence_id',
            'sequence',
            'score',
            'description'
        ]
    )

def filter_scored_fasta_df(
    scored_fasta_df,
    degenerate_pct_thresh = 100,
    sequence_length = 'max'
):
    df = scored_fasta_df.copy()
    df['length'] = df['sequence'].map(len)

    if sequence_length == 'max':
        sequence_length = df['length'].max()
    if sequence_length is not None:
        df = df[df['length'] == sequence_length].copy()

    df['degenerate_pct'] = df['sequence'].map(get_degenerate_pct)
    df = (
        df[
            df['degenerate_pct'] <= degenerate_pct_thresh
        ]
        .copy()
        .reset_index(drop = True)
    )

    return df

def order_scored_fasta_df(
    scored_fasta_df,
    shuffle = True,
    sort = True,
    ascending = True,
    sort_column = 'score'
):
    df = scored_fasta_df.copy()

    if shuffle:
        df = (
            df
            .sample(frac = 1)
            .copy()
            .reset_index(drop = True)
        )
    if sort:
        df = (
            df.sort_values(
                by = sort_column,
                ascending = ascending
            )
            .copy()
            .reset_index(drop = True)
        )
        df['rank'] = np.argsort(df[sort_column])

    return df

def onehot(
    sequence,
    nuc_to_vec,
    convert_masked_seq
):
    return np.array(seq_to_mat(
        sequence,
        nuc_to_vec = nuc_to_vec,
        convert_masked_seq = convert_masked_seq
    ))

def scored_fasta_df_to_dataset(
    scored_fasta_df,
    batch_size = 1000,
    nuc_to_vec = nuc_to_vec,
    convert_masked_seq = convert_masked_seq_dict['n'],
    sequence_column = 'sequence',
    onehot_column = 'onehot_sequence',
    score_column = 'score',
    n_jobs = 1,
    progress_wrapper = tqdm
):

    # def onehot(sequence):
    #     return np.array(seq_to_mat(
    #         sequence,
    #         nuc_to_vec = nuc_to_vec,
    #         convert_masked_seq = convert_masked_seq
    #     ))

    df = scored_fasta_df.copy()
    if n_jobs > 1:
        df[onehot_column] = Parallel(n_jobs = n_jobs)(
            delayed(onehot)(
                sequence,
                nuc_to_vec,
                convert_masked_seq
            )
            for sequence
            in progress_wrapper(df[sequence_column])
        )
    else:
        df[onehot_column] = [
            onehot(
                sequence,
                nuc_to_vec,
                convert_masked_seq
            )
            for sequence
            in progress_wrapper(df[sequence_column])
        ]

    onehot_sequences = np.stack(list(df[onehot_column]))
    scores = df[score_column].astype(float)
    data = [
        onehot_sequences,
        list(scores)
    ]

    return tf.data.Dataset.from_tensor_slices(tuple(data)).map(
        lambda onehot_sequences, scores: (
            tf.cast(onehot_sequences, tf.uint8),
            tf.cast(scores, tf.as_dtype('float'))
        )
    ).batch(batch_size)

def get_input_shape_from_dataset(dataset):
    return tuple(dataset.element_spec[0].shape)[1:]

def get_sequence_length_from_dataset(dataset):
    return get_input_shape_from_dataset(dataset)[0]

def get_normalize_onehot_model(model_input):
    normalize_onehot_model_output = (
        model_input /
        reduce_sum(model_input, axis = -1, keepdims = True)
    )
    normalize_onehot_model = Model(
        inputs = model_input,
        outputs = normalize_onehot_model_output
    )
    return normalize_onehot_model

def get_local_gc_model(
    model_input,
    normalize_onehot_model = None
):
    if normalize_onehot_model is None:
        normalize_onehot_model = get_normalize_onehot_model(model_input)
    local_gc_model_output = reduce_sum(
        normalize_onehot_model(model_input)[:,:,1:-1],
        axis = -1,
        keepdims = True
    )
    local_gc_model = Model(
        inputs = model_input,
        outputs = local_gc_model_output
    )
    return local_gc_model

def get_local_pooled_gc_model(
    model_input,
    pool_size = 1,
    strides = 1,
    padding = 'same',
    local_gc_model = None
):
    if local_gc_model is None:
        local_gc_model = get_local_gc_model(model_input)
    model_input_layer = InputLayer(
        input_shape = model_input.shape[1:]
    )
    pool = AveragePooling1D(
        pool_size = pool_size,
        strides = strides,
        padding = padding
    )
    local_pooled_gc_model = Sequential([
        model_input_layer,
        local_gc_model,
        pool
    ])

    return local_pooled_gc_model


def get_global_gc_model(
    model_input,
    local_gc_model = None
):
    if local_gc_model is None:
        local_gc_model = get_local_gc_model(model_input)
    global_gc_model_output = (
        reduce_sum(
            local_gc_model(model_input),
            axis = -2,
            keepdims = True
        ) /
        model_input.shape[-2])
    global_gc_model = Model(
        inputs = model_input,
        outputs = global_gc_model_output
    )
    return global_gc_model

def append_gc_ratios_to_dataset(dataset, gc_ratio_type = 'global'):

    model_input = Input(shape = get_input_shape_from_dataset(dataset))

    normalize_onehot_model = get_normalize_onehot_model(
        model_input
    )

    local_gc_model = get_local_gc_model(
        model_input,
        normalize_onehot_model = normalize_onehot_model
    )

    if gc_ratio_type is 'global':
        gc_model = get_global_gc_model(
            model_input,
            local_gc_model = local_gc_model
        )
    elif gc_ratio_type is 'local':
        gc_model = local_gc_model

    def append_gc_ratio_to_data(onehot_sequences, scores):
        return (
            onehot_sequences,
            scores,
            gc_model(
                tf.cast(
                    onehot_sequences,
                    tf.as_dtype('float')
                )
            )
        )

    return dataset.map(append_gc_ratio_to_data)

def filepaths_df_to_profile_dicts(
    df,
    motif_orientation = '+',
    mepp_plot_format = 'png'
):
    filepaths_df = df.copy()

    filepaths_df['positions_df_filepath'] = (
        filepaths_df['outdir'] +
        '/positions_df.pkl'
    )
    filepaths_df['mepp_plot_filepath'] = (
        filepaths_df['outdir'] +
        f'/mepp_plot.{mepp_plot_format}'
    )
    filepaths_df['motif_matrix_filepath'] = (
        filepaths_df['outdir'] +
        f'/motif_matrix.npy'
    )
    filepaths_df['positions_df_exists'] = (
        filepaths_df['positions_df_filepath']
        .map(normpath)
        .map(os.path.exists)
    )

    filepaths_df = filepaths_df[
        (filepaths_df['orientation'] == motif_orientation) &
        (filepaths_df['positions_df_exists'] == True) &
        (filepaths_df['retval'] == 0)
    ].copy()

    motif_id_to_profile_df = (
        filepaths_df
        .copy()
        .set_index('motif_id')['positions_df_filepath']
        .map(normpath)
        .map(pd.read_pickle)
    ).to_dict()

    motif_id_to_profile = {
        motif_id: list(df['positional_r'])
        for motif_id, df
        in motif_id_to_profile_df.items()
    }

    motif_id_to_mepp_plot = (
        filepaths_df
        .set_index('motif_id')
        ['mepp_plot_filepath']
        .to_dict()
    )

    motif_id_to_motif_matrix = {
        k: np.load(normpath(v), allow_pickle = True)
        for k,v
        in (
            filepaths_df
            .set_index('motif_id')[
                'motif_matrix_filepath'
            ]
            .to_dict()
            .items()
        )
    }

    return dict(
        motif_id_to_profile = motif_id_to_profile,
        motif_id_to_profile_df = motif_id_to_profile_df,
        motif_id_to_mepp_plot = motif_id_to_mepp_plot,
        motif_id_to_motif_matrix = motif_id_to_motif_matrix,
        filepaths_df = filepaths_df
    )

def get_minmax_stats(positional_r_df):
    df = positional_r_df.copy()
    has_permutation = (
        ('permutation_positional_r_lower' in list(df.columns)) and
        ('permutation_positional_r_upper' in list(df.columns))
    )
    col_prefix = ''
    if has_permutation:
        col_prefix = 'permutation_'
    min_df = df[df['positional_r'] == df['positional_r'].min()].head(1)
    max_df = df[df['positional_r'] == df['positional_r'].max()].head(1)
    min_r = list(min_df['positional_r'])[0]
    min_r_pos = list(min_df['position'])[0]
    min_r_pval = list(min_df[f'{col_prefix}positional_r_pval'])[0]
    max_r = list(max_df['positional_r'])[0]
    max_r_pval = list(max_df[f'{col_prefix}positional_r_pval'])[0]
    max_r_pos = list(max_df['position'])[0]
    if np.abs(max_r) >= np.abs(min_r):
        extreme_r = max_r
        extreme_r_pval = max_r_pval
        extreme_r_pos = max_r_pos
    else:
        extreme_r = min_r
        extreme_r_pval = min_r_pval
        extreme_r_pos = min_r_pos
    r_range = max_r - min_r
    
    has_integral = 'integral_r' in list(positional_r_df.columns)
    if has_integral:
        integral_r = positional_r_df['integral_r'].mean()
        integral_r_lower = positional_r_df['permutation_integral_r_lower'].mean()
        integral_r_upper = positional_r_df['permutation_integral_r_upper'].mean()
        integral_r_pval = positional_r_df['permutation_integral_r_pval'].mean()
    else:
        integral_r = None
        integral_r_lower = None
        integral_r_upper = None
        integral_r_pval = None
    
    return dict(
        integral_r = integral_r,
        integral_r_lower = integral_r_lower,
        integral_r_upper = integral_r_upper,
        integral_r_pval = integral_r_pval,
        extreme_r = extreme_r,
        abs_extreme_r = np.abs(extreme_r),
        extreme_r_pval = extreme_r_pval,
        extreme_r_pos = extreme_r_pos,
        max_r = max_r,
        max_r_pval = max_r_pval,
        max_r_pos = max_r_pos,
        min_r = min_r,
        min_r_pval = min_r_pval,
        min_r_pos = min_r_pos,
        r_range = r_range
    )

def get_minmax_stats_df(
    motif_id_to_profile_df,
    mt_method = 'fdr_tsbky',
    mt_alpha = 0.01,
    thorough_mt = True
):
    minmax_stats_df = pd.DataFrame.from_records([
        dict(
            **{'motif_id': k},
            **get_minmax_stats(df)
        )
        for k, df
        in motif_id_to_profile_df.items()
    ])

    if thorough_mt:
        if 'permutation_positional_r_pval' in list(list(motif_id_to_profile_df.values())[0].columns):
            pval_col = 'permutation_positional_r_pval'
        else:
            pval_col = 'positional_r_pval'
        df = pd.concat({k: df[['position', pval_col]].copy() for k, df in motif_id_to_profile_df.items()})
        df = df.reset_index().rename(columns={'level_0':'motif_id'}).drop(columns = ['level_1'])
        df['padj'] = multipletests(
            df[pval_col],
            method = mt_method,
            alpha = mt_alpha)[1]
        df['sig'] = df['padj'] <= mt_alpha

        permutation_positional_r_padj_lu = df.set_index(['motif_id','position'])['padj'].to_dict()
        permutation_positional_r_sig_lu = df.set_index(['motif_id','position'])['sig'].to_dict()

    for prefix in ['extreme', 'max', 'min']:
        pval_col = f'{prefix}_r_pval'
        padj_col = pval_col.replace('pval', 'padj')
        sig_col = padj_col.replace('padj', 'sig')

        if thorough_mt == False:
            padjs = multipletests(
                minmax_stats_df[pval_col],
                method = mt_method,
                alpha = mt_alpha
            )[1]
            sigs = padjs <= mt_alpha
        else:
            padjs = np.nan
            sigs = np.nan

        pval_col_idx = list(minmax_stats_df.columns).index(pval_col)
        minmax_stats_df.drop(
            columns=[
                col
                for col
                in [padj_col, sig_col]
                if col
                in list(minmax_stats_df.columns)
            ],
            inplace = True
        )
        minmax_stats_df.insert(pval_col_idx+1, sig_col, sigs)
        minmax_stats_df.insert(pval_col_idx+1, padj_col, padjs)

        if thorough_mt:
            lu_keys = minmax_stats_df[['motif_id', f'{prefix}_r_pos']].apply(tuple, axis = 1)
            minmax_stats_df[f'{prefix}_r_padj'] = lu_keys.map(permutation_positional_r_padj_lu)
            minmax_stats_df[f'{prefix}_r_sig'] = lu_keys.map(permutation_positional_r_sig_lu)
    
    has_integral = (minmax_stats_df['integral_r'].isna().any() == False)
    if has_integral:
        pval_col = 'integral_r_pval'
        pval_col_idx = list(minmax_stats_df.columns).index(pval_col)
        padj_col = 'integral_r_padj'
        sig_col = 'integral_r_sig'
        padjs = multipletests(
            minmax_stats_df[pval_col],
            method = mt_method,
            alpha = mt_alpha
        )[1]
        sigs = padjs <= mt_alpha
        minmax_stats_df.insert(pval_col_idx+1, sig_col, sigs)
        minmax_stats_df.insert(pval_col_idx+1, padj_col, padjs)
        
    
    return minmax_stats_df

def re_prefix_filepath_dict(filepath_dict, old_prefix, new_prefix):
    return {
        k: new_prefix+v[len(old_prefix):]
        for k, v
        in filepath_dict.items()
    }
