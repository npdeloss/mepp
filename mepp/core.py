import math

import pandas as pd
import numpy as np

import tensorflow as tf

from scipy import stats

from tensorflow.math import (
    reduce_sum,
    sqrt
)

from tensorflow.keras.layers import AveragePooling1D

from .motif_layers import (
    motif_matrix_to_conv_model
)

from .utils import (
    get_sequence_length_from_dataset
)

def process_dataset_with_model(
    dataset,
    motif_conv_model
):

    has_permutations = len(dataset.element_spec[1].shape) > 1

    has_gc = len(dataset.element_spec) > 2

    def process_data(*data):

        onehot_sequences, scores = data[:2]
        if has_gc:
            gc_ratios = data[2]

        motif_scores = motif_conv_model(
            tf.cast(
                onehot_sequences,
                tf.as_dtype('float')
            ),
            training = False
        )
        if has_permutations:
            expanded_scores = tf.expand_dims(scores, 1)
        else:
            expanded_scores = tf.expand_dims(tf.expand_dims(scores, 1), 1)
        if has_gc:
            return motif_scores, expanded_scores, gc_ratios
        else:
            return motif_scores, expanded_scores

    processed_dataset = dataset.map(process_data)
    return processed_dataset

def smooth_processed_dataset(
    processed_dataset,
    motif_score_pool_size = 4,
    motif_score_stride = 1,
    motif_score_padding = 'valid',
    gc_ratio_pool_size = 20,
    gc_ratio_stride = 1,
    gc_ratio_padding = 'valid'
):

    has_gc = False
    local_gc = False
    if len(processed_dataset.element_spec) > 2:
        has_gc = True
        gc_ratios_spec = processed_dataset.element_spec[2]
        if gc_ratios_spec.shape[1] > 1:
            local_gc = True
    
    sequence_length = get_sequence_length_from_dataset(processed_dataset)
    print(sequence_length)
    if motif_score_padding == 'valid':
        motif_score_output_length = int(math.floor((sequence_length - motif_score_pool_size) / motif_score_stride) + 1)
        motif_score_left_pad = (sequence_length - motif_score_output_length)//2
        motif_score_right_pad = sequence_length - motif_score_output_length - motif_score_left_pad
    else:
        motif_score_left_pad = 0
        motif_score_right_pad = 0
    
    motif_score_paddings = tf.constant([
        [0,0],
        [motif_score_left_pad,motif_score_right_pad],
        [0,0]
    ])
    print(motif_score_paddings)
    if local_gc:
        if gc_ratio_padding == 'valid':
            gc_ratio_output_length = int(math.floor((sequence_length - gc_ratio_pool_size) / gc_ratio_stride) + 1)
            gc_ratio_left_pad = (sequence_length - gc_ratio_output_length)//2
            gc_ratio_right_pad = sequence_length - gc_ratio_output_length - gc_ratio_left_pad
        else:
            gc_ratio_left_pad = 0
            gc_ratio_right_pad = 0


        gc_ratio_paddings_ = [[0,0]]*len(gc_ratios_spec.shape)
        gc_ratio_paddings_[1] = [gc_ratio_left_pad,gc_ratio_right_pad]
        gc_ratio_paddings = tf.constant([
           gc_ratio_paddings_
        ])
        print(gc_ratio_paddings)
    
    motif_score_pool = AveragePooling1D(
        pool_size = motif_score_pool_size,
        strides = motif_score_stride,
        padding = motif_score_padding
    )
    
    if local_gc:
        gc_ratio_pool = AveragePooling1D(
            pool_size = gc_ratio_pool_size,
            strides = gc_ratio_stride,
            padding = gc_ratio_padding
        )
    def smooth_data(*data):

        motif_scores, expanded_scores = data[:2]
        smoothed_motif_scores = tf.pad(motif_score_pool(motif_scores), motif_score_paddings)

        if has_gc:
            gc_ratios = data[2]
            if local_gc:
                smoothed_gc_ratios = tf.pad(gc_ratio_pool(gc_ratios), gc_ratio_paddings)
            else:
                smoothed_gc_ratios = gc_ratios
            return (
                smoothed_motif_scores,
                expanded_scores,
                smoothed_gc_ratios
            )
        else:
            return (
                smoothed_motif_scores,
                expanded_scores
            )

    smoothed_dataset = processed_dataset.map(smooth_data)
    return smoothed_dataset


def get_motif_score_matrix_from_processed_dataset(processed_dataset):
    return tf.concat(
        [
            motif_score_batch
            for motif_score_batch
            in processed_dataset.map(
                lambda *data: data[0]
            )
        ],
        axis = 0
    )

def get_scores_from_processed_dataset(processed_dataset):
    return tf.concat(
        [
            score_batch
            for score_batch
            in processed_dataset.map(
                lambda *data: data[1]
            )
        ],
        axis = 0
    )

def get_score_rank_df(scores, sort = True):
    if sort:
        ranks = np.argsort(scores)
    else:
        ranks = np.arange(len(scores))
    return pd.DataFrame(dict(
        rank = ranks,
        score = scores
    ))

r_ab_num = lambda sum_a, sum_b, sum_ab, n: (
    (n*sum_ab) -
    (sum_a*sum_b)
)
r_ab_den = lambda sum_a, sum_b, sum_aa, sum_bb, n: sqrt(
    ((n*sum_aa)-(sum_a*sum_a))*
    ((n*sum_bb)-(sum_b*sum_b))
)
r_ab = lambda sum_a, sum_b, sum_aa, sum_bb, sum_ab, n: (
    r_ab_num(sum_a, sum_b, sum_ab, n) /
    r_ab_den(sum_a, sum_b, sum_aa, sum_bb, n)
)

def get_positional_correlation_from_processed_dataset(
    processed_dataset
):

    control_gc = False

    output_shape = processed_dataset.element_spec[0].shape[1:]
    if len(processed_dataset.element_spec) > 2:
        control_gc = True

    n = 0

    sum_x = tf.zeros(output_shape)
    sum_y = tf.zeros(output_shape)
    if control_gc:
        sum_z = tf.zeros(output_shape)

    sum_xx = tf.zeros(output_shape)
    sum_yy = tf.zeros(output_shape)
    if control_gc:
        sum_zz = tf.zeros(output_shape)

    sum_xy = tf.zeros(output_shape)
    if control_gc:
        sum_xz = tf.zeros(output_shape)
        sum_yz = tf.zeros(output_shape)

    for data in processed_dataset:
        motif_scores, expanded_scores = data[:2]

        if control_gc:
            gc_ratios = data[2]

        n += motif_scores.shape[0]

        x = motif_scores
        y = expanded_scores
        if control_gc:
            z = gc_ratios

        sum_x += reduce_sum(x, 0)
        sum_y += reduce_sum(y, 0)

        if control_gc:
            sum_z += reduce_sum(z, 0)

        sum_xx += reduce_sum(x*x, 0)
        sum_yy += reduce_sum(y*y, 0)

        if control_gc:
            sum_zz += reduce_sum(z*z, 0)

        sum_xy += reduce_sum(x*y, 0)
        if control_gc:
            sum_xz += reduce_sum(x*z, 0)
            sum_yz += reduce_sum(y*z, 0)

    r_xy = r_ab(sum_x, sum_y, sum_xx, sum_yy, sum_xy, n)
    if control_gc:
        r_xz = r_ab(sum_x, sum_z, sum_xx, sum_zz, sum_xz, n)
        r_yz = r_ab(sum_y, sum_z, sum_yy, sum_zz, sum_yz, n)

        r_xy_z_num = (r_xy - r_xz*r_yz)
        r_xy_z_den = sqrt((1-(r_xz*r_xz))*(1-(r_yz*r_yz)))
        r_xy_z = r_xy_z_num / r_xy_z_den
        r = r_xy_z
    else:
        r = r_xy

    return r

def get_positional_r_df(
    r,
    center = None,
    channel_index = 0,
    confidence_interval_pct = 95,
    permutation_mode = ['full', 'semi'][1]
):
    has_extra_dim = len(r.shape) > 1
    if has_extra_dim:
        r_ = r
    else:
        r_ = np.expand_dims(r, -1)

    has_permutations = r_.shape[1] > 1
    if has_permutations:
        channel_index = 0

    positional_r_df = pd.DataFrame(dict(
        positional_r = r_[:,channel_index].flatten()
    ))
    positional_r_df['position'] = list(range(positional_r_df.shape[0]))
    sequence_length = positional_r_df.shape[0]
    if center is None:
        center = sequence_length//2
    positional_r_df['position'] -= center

    if has_permutations:
        # Calculate confidence intervals
        alpha = 1.0 - (confidence_interval_pct/100.0)
        percentile_lower = 100.0*alpha/2
        percentile_upper = 100.0 - 100.0*alpha/2
        q = (percentile_lower, percentile_upper)
        r_percentiles = np.percentile(
            r_[:,1:], q, axis=-1
        )
        
        # Calculate permutation p-values
        positional_r_pvals = np.mean(
            (
                np.abs(r_[:,1:]) >=
                np.abs(r_[:,0:1])
            ).astype(float),
            axis = -1
        )
        
        semi_positional_r_percentiles = np.percentile(
            r_[:,1:].flatten(), q
        )
        
        semi_positional_r_pvals = np.mean(
            (
                np.expand_dims(np.abs(r_[:,1:]).flatten(), axis = 0) >=
                np.abs(r_[:,0:1])
            ).astype(float),
            axis = -1
        )

        # Format confidence intervals and p-values to dataframe
        positional_r_df[
            'permutation_full_positional_r_lower'
        ] = r_percentiles[0]
        positional_r_df[
            'permutation_full_positional_r_upper'
        ] = r_percentiles[1]
        positional_r_df[
            'permutation_full_positional_r_pval'
        ] = positional_r_pvals
        
        # Format confidence intervals and p-values to dataframe
        positional_r_df[
            'permutation_semi_positional_r_lower'
        ] = semi_positional_r_percentiles[0]
        positional_r_df[
            'permutation_semi_positional_r_upper'
        ] = semi_positional_r_percentiles[1]
        positional_r_df[
            'permutation_semi_positional_r_pval'
        ] = semi_positional_r_pvals
        
        for suffix in ['positional_r_lower', 'positional_r_upper', 'positional_r_pval']:
             positional_r_df[f'permutation_{suffix}'] = positional_r_df[f'permutation_{permutation_mode}_{suffix}']
        
        # Get integral of deviation from 0 across positions
        integral_r = np.trapz(np.abs(r_), axis = 0)
        integral_r_percentiles = np.percentile(
            integral_r[1:].flatten(), q
        )
        integral_r_pval = np.mean(
            (
                integral_r[1:].flatten() >=
                integral_r[0]
            ).astype(float),
            axis = -1
        )
        positional_r_df[
            'integral_r'
        ] = integral_r[0]
        positional_r_df[
            'permutation_integral_r_lower'
        ] = integral_r_percentiles[0]
        positional_r_df[
            'permutation_integral_r_upper'
        ] = integral_r_percentiles[1]
        positional_r_df[
            'permutation_integral_r_pval'
        ] = integral_r_pval
        
    return positional_r_df

import scipy.stats as st

def add_parametric_confidence_intervals_to_positional_r_df(
    positional_r_df,
    num_samples,
    confidence_interval_pct = 95
):
    df = positional_r_df.copy()

    r = positional_r_df['positional_r']
    n = num_samples
    alpha = 1.0 - (confidence_interval_pct/100.0)

    z_margin = st.norm.ppf(1.0 - (alpha)/2.0)
    z_margin *= np.sqrt(1.0 / (n - 3))
    eps = np.finfo(r.dtype).eps

    z_r = 0.5 * np.log((1 + r + eps)/(1 - r + eps))
    z_lower = z_r - z_margin
    z_upper = z_r + z_margin

    z_to_r = lambda z: (
        (np.exp(2 * z + eps) - 1.0) /
        (np.exp(2 * z + eps) + 1.0)
    )

    r_lower = z_to_r(z_lower)
    r_upper = z_to_r(z_upper)
    df['positional_r_lower'] = r_lower
    df['positional_r_upper'] = r_upper

    positional_t = r / np.sqrt((1 - (r * r))/(n-2))
    positional_r_pval = stats.t.sf(np.abs(positional_t), n-1)*2
    df['positional_r_pval'] = positional_r_pval

    return df

def get_motif_counts_by_position(
    motif_score_matrix
):
    return np.sum(
        (
            (motif_score_matrix > 0.0)
            .astype(np.uintc)
        ),
        axis = 0
    ).astype(np.float)

def get_motif_counts_by_position_df(motif_score_matrix, center = None):
    motif_counts_by_position = get_motif_counts_by_position(motif_score_matrix)
    motif_counts_by_position_df = pd.DataFrame(dict(
      position = np.array(range(motif_counts_by_position.shape[0])),
      count = motif_counts_by_position
    ))
    sequence_length = motif_score_matrix.shape[1]
    if center is None:
        center = sequence_length//2
    motif_counts_by_position_df['position'] -= center
    return motif_counts_by_position_df

def smooth_motif_counts_by_position_df(
    motif_counts_by_position_df,
    window = 5
):
    smoothed_motif_counts_by_position_df = (
        motif_counts_by_position_df
        .set_index('position')
        .rolling(
            window,
            center = True
        )
        .mean()
        .reset_index()
        .ffill()
        .bfill()
    )
    return smoothed_motif_counts_by_position_df

def get_motif_density_by_rank(motif_score_matrix):
    return np.sum(
        (
            (motif_score_matrix > 0.0)
            .astype(np.uintc)
        ),
        axis = 1
    ).astype(np.float)

def get_motif_density_by_rank_df(motif_score_matrix):
    motif_density_by_rank = get_motif_density_by_rank(motif_score_matrix)
    return pd.DataFrame(dict(
        rank = np.array(range(motif_density_by_rank.shape[0])),
        density = motif_density_by_rank,
    ))

def smooth_motif_density_by_rank_df(
    motif_density_by_rank_df,
    window
):
    smoothed_motif_density_by_rank_df = (
        motif_density_by_rank_df
        .set_index('rank')
        .rolling(
            window,
            center = True
        )
        .mean()
        .reset_index()
        .ffill()
        .bfill()
    )
    return smoothed_motif_density_by_rank_df

def dataset_and_motif_matrix_to_profile_data(
    dataset,
    motif_matrix,
    orientation = '+',
    motif_margin = 0,
    motif_pseudocount = 0.0001,
    motif_pvalue = 0.0001,
    bg = None,
    confidence_interval_pct = 95,
    center = None
):
    channel_index = 0
    # Create motif model
    sequence_length = get_sequence_length_from_dataset(dataset)
    if center is None:
        center = sequence_length//2
    if center < 0:
        center = 0
    if center > sequence_length:
        center = sequence_length - 1
    # print(orientation)
    if orientation is '+':
        # print(f'{orientation} confirmed')
        motif_conv_model = motif_matrix_to_conv_model(
            motif_matrix,
            sequence_length,
            revcomp = False,
            dual = False
        )
    elif orientation is '-':
        # print(f'{orientation} confirmed')
        motif_conv_model = motif_matrix_to_conv_model(
            motif_matrix,
            sequence_length,
            revcomp = True,
            dual = False
        )
    else:
        # print(f'{orientation} confirmed')
        motif_conv_model = motif_matrix_to_conv_model(
            motif_matrix,
            sequence_length,
            revcomp = False,
            dual = True
        )

    profile_data = dict(
    )

    # Process dataset for motif match scores
    processed_dataset = process_dataset_with_model(
        dataset,
        motif_conv_model
    )

    # Smooth processed dataset
    if motif_margin > 0:
        smoothed_processed_dataset = smooth_processed_dataset(
            processed_dataset,
            motif_score_pool_size = 1 + motif_margin * 2
        )
    else:
        smoothed_processed_dataset = processed_dataset

    # Calculate positional correlation
    r = get_positional_correlation_from_processed_dataset(
        smoothed_processed_dataset
    )

    # Retrieve sequence scores
    scores = (
        get_scores_from_processed_dataset(
            processed_dataset
        )[:,0,channel_index]
        .numpy()
    )

    # Retrieve motif score matrix
    motif_score_matrix = (
        get_motif_score_matrix_from_processed_dataset(
            processed_dataset
        )[:,:,channel_index]
        .numpy()
    )

    # Generate dataframes
    score_rank_df = get_score_rank_df(
        scores,
        sort = False
    )

    motif_counts_by_position_df = get_motif_counts_by_position_df(
        motif_score_matrix,
        center = center
    )

    if motif_margin > 0:
        smoothed_motif_counts_by_position_df = (
            smooth_motif_counts_by_position_df(
                motif_counts_by_position_df,
                window = 1 + motif_margin * 2
            )
        )
    else:
        smoothed_motif_counts_by_position_df = motif_counts_by_position_df

    motif_density_by_rank_df = get_motif_density_by_rank_df(
        motif_score_matrix
    )
    
    positional_r_df = add_parametric_confidence_intervals_to_positional_r_df(
        get_positional_r_df(
            np.nan_to_num(r.numpy(), 0.0),
            confidence_interval_pct = confidence_interval_pct,
            center = center
        ),
        num_samples = len(scores),
        confidence_interval_pct = confidence_interval_pct
    )

    # Consolidate position dataframes

    positions_df = (
        positional_r_df
        .copy()
    )

    positions_df['count'] = list(motif_counts_by_position_df['count'])
    positions_df['smoothed_count'] = list(smoothed_motif_counts_by_position_df['count'])


    # Consolidate rank dataframes

    ranks_df = score_rank_df.copy()
    ranks_df['density'] = list(motif_density_by_rank_df['density'])

    return (
        motif_score_matrix,
        positions_df,
        ranks_df,
        processed_dataset,
        smoothed_processed_dataset
    )
