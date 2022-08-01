"""Helper script to learn motifs given positional profiles using neural networks"""

import os
import sys
import click
import multiprocessing

from os.path import normpath

from contextlib import redirect_stdout

import pandas as pd
import numpy as np

import random

import tensorflow as tf
import tensorflow.keras as keras

from joblib import Parallel, delayed
from tqdm.auto import tqdm

from logomaker import transform_matrix

from scipy.stats import linregress

from .io import (
    scored_fasta_filepath_to_dicts,
    save_dataset,
    load_dataset
)
from .utils import (
    force_cpu_only,
    manage_gpu_memory,
    order_scored_fasta_df,
    filter_scored_fasta_df,
    scored_fasta_dicts_to_df,
    scored_fasta_df_to_dataset,
)

from .plot import motif_matrix_to_df
from .html import get_logo_df
from .io import motif_matrix_file_to_dicts
from .io import motif_matrix_filepath_to_dicts
from .html import motif_matrix_to_logo_data_uri

from .learn_motifs import (
    revcomp_augment_dataset, 
    revcomp_dataset, 
    reformat_dataset
)

from .learn_motifs import (
    amplify_motifs, 
    extract_motifs, 
    filter_motifs_by_information_content, 
    motifs_to_dict, 
    motif_matrix_dict_to_file
)

from .io import motif_matrix_filepath_to_dicts
from .motif_scanning import (
    scan_motifs_parallel, 
    format_scan_results
)

def mask_sequence(sequence, index, length, mask_char = 'N'):
    return (
        sequence[:index] + 
        ''.join([mask_char] * length) +
        sequence[index+length:]
    )[:len(sequence)]

def multimask_sequence(
    sequence, 
    indices, 
    lengths, 
    mask_char = 'N'
):
    masked_sequence = sequence
    for index, length in zip(indices, lengths):
        masked_sequence = mask_sequence(
            masked_sequence, 
            index, 
            length, 
            mask_char = mask_char
        )
    return masked_sequence

def seq_to_gc(
    seq, 
    nuc_to_gc = {
        'A':0.0,
        'C':1.0,
        'G':1.0,
        'T':0.0,
        'N':0.25
    }
):
    nuc_to_gc_keys = set(nuc_to_gc.keys())
    gc = np.sum([
        nuc_to_gc[nuc] 
        for nuc 
        in list(seq) 
        if nuc in nuc_to_gc_keys
    ])
    return gc
    

def mask_scored_fasta_df(
    scored_fasta_df,
    mask_motif_matrix_dict,
    mask_motif_pseudocount = 0.1,
    mask_motif_pvalue = 0.001,
    mask_char = '_',
    replace = True,
    n_jobs = 1
):
    # Scan for motifs to mask
    sequences_dict = scored_fasta_df.set_index('sequence_id')['sequence'].to_dict()
    scan_results_df = format_scan_results(
        scan_motifs_parallel(
            mask_motif_matrix_dict,
            sequences_dict,
            pseudocount = mask_motif_pvalue,
            pval = mask_motif_pvalue,
            n_jobs = n_jobs,
            progress_wrapper = tqdm
        )
    )[0].rename(
        columns={'peak_id':'sequence_id'}
    )
    
    # Calculate length of masked regions
    mask_motif_length_dict = {
        k: v.shape[1]
        for k,v
        in mask_motif_matrix_dict.items()
    }
    scan_results_df['motif_length'] = (
        scan_results_df['motif_id']
        .map(mask_motif_length_dict)
    )
    
    masked_sequences_dict = sequences_dict
    for sequence_id, subset_df in tqdm(scan_results_df.groupby('sequence_id')):
        num_masked_motifs = subset_df.shape[0]
        num_unique_masked_motifs = len(set(subset_df['motif_id']))
        # print(f'Masking {num_masked_motifs} ({num_unique_masked_motifs} unique) motifs from {sequence_id}')
        indices = list(subset_df['instance_position'])
        lengths = list(subset_df['motif_length'])
        sequence = sequences_dict[sequence_id]
        masked_sequence = multimask_sequence(
            sequence, 
            indices, 
            lengths, 
            mask_char = mask_char
        )
        num_modified_bases = np.sum([1 for a,b in zip(list(sequence),list(masked_sequence)) if a!=b])
        # print(f'Masked {num_modified_bases} bases')
        log_vals = [
            'masking_data',
            num_masked_motifs,
            num_unique_masked_motifs,
            sequence_id,
            num_modified_bases,
            len(sequence)
        ]
        # print('\t'.join([f'{val}' for val in log_vals]))
        masked_sequences_dict[sequence_id] = masked_sequence
    if replace == True:
        output_col = 'sequence'
    else:
        output_col = 'masked_sequence'
    scored_fasta_df[output_col] = (
        scored_fasta_df['sequence_id']
        .map(masked_sequences_dict)
    )
    return scored_fasta_df
    

def positionalize_dataset_scores(dataset, positional_profile):
    if len(positional_profile.shape) == 2:
        return dataset.map(
            lambda sequences, scores: (
                sequences, 
                (
                    tf.expand_dims(
                        tf.cast(
                            positional_profile, 
                            scores.dtype
                        ), 
                        axis=0
                    ) * 
                    tf.expand_dims(
                        scores,
                        axis = -1
                    )
                )
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        return dataset.map(
            lambda sequences, scores: (
                sequences, 
                (
                    tf.expand_dims(
                        tf.cast(
                            positional_profile, 
                            scores.dtype
                        ), 
                        axis=0
                    ) * 
                    scores
                )
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

import math

def generate_positional_conv_model(
    input_layer, 
    num_motifs = 320, 
    motif_length = 8, 
    seed = 10, 
    motif_margin=2, 
    unstranded = True
):
    conv_model = keras.Sequential([
        input_layer,
        keras.layers.Conv1D(
            32, 
            2,activation='relu', 
            kernel_initializer = keras.initializers.GlorotUniform(seed=seed)
        ),
        keras.layers.Conv1D(
            num_motifs, 
            motif_length-1,activation='relu', 
            kernel_initializer = keras.initializers.GlorotUniform(seed=seed+1),
            # kernel_regularizer = OrthoRegularizer(
            #     factor=0.01, 
            #     mode='rows'
            # )
        )
    ])
    
    revcomp_conv_model = keras.Model(
        inputs = input_layer,
        outputs = (
            conv_model(input_layer[:,::-1,::-1])
            [:,::-1,:]
        )
    )

    dual_conv_model = keras.Model(
        inputs = input_layer,
        outputs = tf.math.reduce_max(
            [
                conv_model(input_layer), 
                revcomp_conv_model(input_layer)
            ]
            , axis = 0
        )
    )
    
    if unstranded:
        chosen_conv_model = dual_conv_model
    else:
        chosen_conv_model = conv_model
    
    pool_size = 1+motif_margin*2
    post_conv_model = keras.Sequential([
        input_layer,
        chosen_conv_model,
        keras.layers.Dropout(0.1, seed = seed + 2),
        keras.layers.AveragePooling1D(
            pool_size, 
            strides = 1
        )
    ])
  
    total_pad = input_layer.shape[1]-post_conv_model.output_shape[1]
    left_pad = total_pad//2
    right_pad = total_pad-left_pad
    
    model = keras.Sequential([
        input_layer,
        post_conv_model,
        keras.layers.ZeroPadding1D(padding=(left_pad, right_pad)),
#         keras.layers.Conv1D(
#             num_motifs, 
#             1,
#             groups=num_motifs,
#             activation='linear', 
#             kernel_initializer = keras.initializers.GlorotUniform(seed=seed+2),
#             kernel_regularizer = 'l1_l2',
#             use_bias = False
#         )
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mse','mae'])
    return model, conv_model, post_conv_model


def convert_masked_seq_with_underscores(seq, alphabet = list('ACGT_')):
    return ''.join(
        [
            nuc if nuc in alphabet else 'N'
            for nuc
            in seq
        ]
    )

def get_shifted_positional_profiles(
    profile_df, 
    num_motifs,
    offset_start, 
    offset_end
):
    offsets = np.arange(
        offset_start, 
        offset_end+1
    )
    offset_profile_cols = [
        f'profile_offset_{offset}' 
        for offset in offsets
    ]
    
    offsets_and_offset_profile_cols = zip(
        offsets, 
        offset_profile_cols
    )
    
    for offset, offset_profile_col in offsets_and_offset_profile_cols:
        profile_df[offset_profile_col] = (
            profile_df['profile']
            .shift(offset)
            .bfill()
            .ffill()
        )
    total_profile_cols = offset_profile_cols * num_motifs
    print('\n'.join(total_profile_cols))
    positional_profiles = (
        profile_df[total_profile_cols]
        .values
    )
    
    print('positional_profiles.shape[-1] = {positional_profiles.shape[-1]}')
    
    num_shifted_profiles = (
        len(offset_profile_cols)
    )
    
    return (
        positional_profiles, 
        profile_df,
        num_shifted_profiles,
        offset_profile_cols
    )

@click.command()
# Filepaths
@click.option(
    '--fa',
    'scored_fasta_filepath',
    type = str,
    required = True,
    help = (
        'Path to a scored fasta file, '
        'where sequence headers are of the form: '
        '">sequence_name sequence_score".'
    )
)
@click.option(
    '--profile',
    'profile_filepath',
    type = str,
    required = True,
    help = (
        'Path to a tab-separated text file '
        'with two columns labeled "position" and "profile"'
        'describing positions along a sequence '
        'and the target profile for enriching a motif '
        'at those positions. '
    )
)
@click.option(
    '--out',
    'out_filepath',
    type = str,
    required = True,
    help = (
        'Create this directory and write output to it.'
    )
)
# Positional parameters
@click.option(
    '--center',
    'center',
    type = int,
    default = None,
    help = (
        '0-based offset from the start of the sequence to center profiles on. '
        'Default: Set the center to half the sequence length, rounded down'
    )
)
@click.option(
    '--margin',
    'motif_margin',
    type = int,
    default = 2,
    help = (
        'Number of bases along either side of motif to "blur" motif matches '
        'for smoothing. '
        'Default: 2'
    )
)
# Dataset parameters
@click.option(
    '--dgt',
    'degenerate_pct_thresh',
    type = float,
    default = 100.0,
    help = (
        'Percentage of sequence that can be degenerate '
        '(Not A, C, G, or T) before being rejected from the analysis. '
        'Useful for filtering out repeats. '
        'Default: 100'
    )
)
@click.option(
    '--batch',
    'batch_size',
    type = int,
    default = 1000,
    help = (
        'Size of batches for Tensorflow datasets. '
        'Default: 1000'
    )
)
# @click.option(
#     '--val',
#     'validation_fraction',
#     type = float,
#     default = 0.10,
#     help = (
#         'Fraction of data used for validation. '
#         'Default: 0.10'
#     )
# )
@click.option(
    '--gc',
    'gc_control',
    flag_value = True,
    default = True,
    help = (
        'Regresses out effect of GC content on sequence score.'
        'Default: GC control is enabled'
    )
)
@click.option(
    '--nogc',
    'gc_control',
    flag_value = False,
    default = True,
    help = (
        'Disables GC control. See --gc.'
        'Default: GC control is enabled'
    )
)
# Motif parameters
@click.option(
    '--motifs',
    'num_motifs',
    type = int,
    default = 320,
    help = (
        'Number of motifs to learn. '
        'Default: 320'
    )
)
@click.option(
    '--length',
    'motif_length',
    type = int,
    default = 8,
    help = (
        'Length of motifs to learn. '
        'Default: 8'
    )
)
@click.option(
    '--orientation',
    'orientation',
    type = click.Choice([
        '+',
        '-',
        '+/-'
    ]),
    default = '+',
    help = (
        'Orientation of motifs. '
        '+ and - are equivalent. '
        '+/- enables strand-invariant motif learning, '
        'which is useful if the input data are unstranded.'
        'Default: +'
    )
)
@click.option(
    '--motif-prefix',
    'motif_prefix',
    type = str,
    default = 'denovo_motif_',
    help = (
        'Prefix motif names with this string.'
        'Default: denovo_motif_'
    )
)
@click.option(
    '--shifts',
    'profile_shift',
    type = int,
    default = 0,
    help = (
        'Make motifs learn from '
        'shifted versions of the profile. '
        'The profile will be shifted within the range of +/- profile_shift.'
        'The final number of motifs will be num_motifs * (2*profile_shift + 1).'
        'Default: Do not shift (0)'
    )
)
# Masking parameters
@click.option(
    '--mask-motifs',
    'mask_motifs_filepath',
    type = str,
    default = None,
    help = (
        'File of motifs to mask out of the dataset, '
        'in JASPAR format.'
    )
)
@click.option(
    '--mask-motifs-list',
    'mask_motifs_list_filepath',
    type = str,
    default = None,
    help = (
        'File of motif ids for motifs to mask out of the dataset, '
        'one motif id per line. '
        'Use to mask only the listed motifs from mask_motifs_filepath, '
        'retaining the rest. '
        'Default: use all motifs from mask_motifs_filepath'
    )
)
@click.option(
    '--mask-pcount',
    'mask_motif_pseudocount',
    type = float,
    default = 0.1,
    help = (
        'Pseudocount for setting masked motif match threshold via MOODS. '
        'Default: 0.1'
    )
)
@click.option(
    '--mask-pval',
    'mask_motif_pvalue',
    type = float,
    default = 0.05,
    help = (
        'P-value for setting motif match threshold via MOODS. '
        'Default: 0.001'
    )
)
# Model parameters
@click.option(
    '--seed',
    'seed',
    type = int,
    default = 1000,
    help = (
        'Random seed for shuffling and initialization. '
        'Default: 1000'
    )
)
# Training parameters
@click.option(
    '--epochs',
    'epochs',
    type = int,
    default = 1000,
    help = (
        'Maximum number of epochs for training. '
        'Default: 1000'
    )
)
@click.option(
    '--no-early-stopping',
    'early_stopping',
    default = True,
    flag_value = False,
    help = (
        'Disable early stopping of training, to train for the maximum number of epochs. '
        'Default: Enable early stopping.'
    )
)
@click.option(
    '--patience',
    'early_stopping_patience',
    type = int,
    default = 10,
    help = (
        'Number of epochs to wait for early stopping. '
        'Default: 1000'
    )
)
@click.option(
    '--mindelta',
    'early_stopping_min_delta',
    type = float,
    default = 0.0,
    help = (
        'Minimum delta for early stopping. '
        'Default: 0'
    )
)
# Execution parameters
@click.option(
    '--jobs',
    'n_jobs',
    type = int,
    default = multiprocessing.cpu_count(),
    help = (
        'Number of jobs '
        'for CPU multiprocessing. '
        'Default: Use all cores'
    )
)
@click.option(
    '--nogpu',
    'no_gpu',
    is_flag=True,
    help = (
        'Disable use of GPU. '
    )
)
@click.option(
    '--quiet',
    'quiet',
    default = False,
    is_flag = True,
    flag_value = True,
    help = (
        'Do not write combined motifs to stdout. '
        'Default: Write combined motifs to stdout.'
    )
)
def main(
    # Filepaths
    scored_fasta_filepath,
    profile_filepath,
    out_filepath,
    # Positional parameters
    center = None,
    motif_margin = 2,
    # Dataset parameters
    degenerate_pct_thresh = 100.0,
    batch_size = 1000,
    validation_fraction = 0.10,
    gc_control = True,
    # Motif parameters
    num_motifs = 320,
    motif_length = 8,
    orientation = '+',
    motif_prefix = 'positional_denovo_motif_',
    profile_shift = 0,
    # Masking parameters
    mask_motifs_filepath = None,
    mask_motifs_list_filepath = None,
    mask_motif_pseudocount = 0.0001,
    mask_motif_pvalue = 0.0001,
    # Model parameters
    seed = 10,
    # Training parameters
    epochs = 1000,
    early_stopping = True,
    early_stopping_patience = 5,
    early_stopping_min_delta = 0,
    # Execution parameters
    n_jobs = multiprocessing.cpu_count(),
    no_gpu = False,
    quiet = False
):
    with redirect_stdout(sys.stderr):
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Setup output

        filtered_data_df_tsv_filepath = normpath(f'{out_filepath}/filtered_data_df.tsv')
        filtered_data_df_pkl_filepath = normpath(f'{out_filepath}/filtered_data_df.pkl')
        dataset_filepath = normpath(f'{out_filepath}/dataset')

        os.makedirs(
            normpath(out_filepath), 
            exist_ok = True
        )

        os.makedirs(
            dataset_filepath,
            exist_ok = True
        )

        # Manage GPU memory usage

        if no_gpu:
            force_cpu_only()
        else:
            manage_gpu_memory()

        # Load scored sequence data
        (
            sequence_dict,
            score_dict,
            description_dict
        ) = scored_fasta_filepath_to_dicts(scored_fasta_filepath)

        # Filter and sort sequence data
        scored_fasta_df = order_scored_fasta_df(
            filter_scored_fasta_df(
                scored_fasta_dicts_to_df(
                    sequence_dict,
                    score_dict,
                    description_dict
                ),
                degenerate_pct_thresh = degenerate_pct_thresh
            )
        )
        
        # Mask motifs, if applicable
        if mask_motifs_filepath != None:
            mask_motif_matrix_dict_ = (
                motif_matrix_filepath_to_dicts(
                    mask_motifs_filepath
                )[0]
            )
            
            mask_motifs_list = [k for k in mask_motif_matrix_dict_.keys()]
            if mask_motifs_list_filepath != None:
                with open(mask_motifs_list_filepath) as f:
                    mask_motifs_list_raw = [
                        l.strip() 
                        for l 
                        in f.readlines()
                    ]
                    mask_motifs_list_filtered = list(
                        set(mask_motifs_list) & 
                        set(mask_motifs_list_raw)
                    )
                    mask_motifs_list = mask_motifs_list_filtered
            
            mask_motif_matrix_dict = {
                k: v
                for k, v
                in mask_motif_matrix_dict_.items()
                if k in mask_motifs_list
            }
            
            num_mask_motifs = len(list(mask_motif_matrix_dict.keys()))
            print(f'Masking out {num_mask_motifs} motif(s)')
            if num_mask_motifs > 0:
                scored_fasta_df = mask_scored_fasta_df(
                    scored_fasta_df,
                    mask_motif_matrix_dict,
                    mask_motif_pseudocount = mask_motif_pseudocount,
                    mask_motif_pvalue = mask_motif_pvalue,
                    n_jobs = n_jobs
                )
            
        
        # Normalize scores
        scored_fasta_df['original_score'] = scored_fasta_df['score']
        
        scored_fasta_df['z_score'] = (
            (
                scored_fasta_df['original_score'] - 
                scored_fasta_df['original_score'].mean()
            ) /
            (
                scored_fasta_df['original_score'].std()
            )
        )
        
        scored_fasta_df['score'] = scored_fasta_df['z_score']
        
        # Control GC
        print(f'gc_control = {gc_control}')
        if gc_control:
            scored_fasta_df['gc_ratio'] = (
                (
                    scored_fasta_df['sequence']
                    .map(seq_to_gc)
                ) /
                (
                    scored_fasta_df['sequence']
                    .map(len)
                    .astype(float)
                )
            )
            
            scored_fasta_df['z_gc_ratio'] = (
                (
                    scored_fasta_df['gc_ratio'] - 
                    scored_fasta_df['gc_ratio'].mean()
                ) /
                (
                    scored_fasta_df['gc_ratio'].std()
                )
            )
            
            (
                gc_model_slope, 
                gc_model_intercept, 
                gc_model_r_value, 
                gc_model_p_value, 
                gc_model_std_err
            ) = linregress(
                scored_fasta_df['z_gc_ratio'],
                scored_fasta_df['score']
            )
            
            scored_fasta_df['predicted_score_from_gc_ratio'] = (
                (
                    scored_fasta_df['z_gc_ratio'] * 
                    gc_model_slope
                ) +
                gc_model_intercept
            )
            
            scored_fasta_df['residual_score_from_gc_ratio'] = (
                scored_fasta_df['score'] -
                scored_fasta_df['predicted_score_from_gc_ratio']
            )
            
            scored_fasta_df['z_residual_score_from_gc_ratio'] = (
                (
                    scored_fasta_df['residual_score_from_gc_ratio'] - 
                    scored_fasta_df['residual_score_from_gc_ratio'].mean()
                ) /
                (
                    scored_fasta_df['residual_score_from_gc_ratio'].std()
                )
            )
            
            scored_fasta_df['score'] = (
                scored_fasta_df['z_residual_score_from_gc_ratio']
            )
        
        
        # Write filtered data to file
        scored_fasta_df.to_csv(
            filtered_data_df_tsv_filepath,
            sep = '\t',
            index = False
        )
        
        scored_fasta_df.to_pickle(filtered_data_df_pkl_filepath)

        # Pad to meet batch size
        scored_fasta_df_padding = (
            scored_fasta_df.sample(
                n = (batch_size - scored_fasta_df.shape[0]%batch_size), 
                random_state = seed
            )
            .copy()
        )
        padded_scored_fasta_df = (
            pd.concat([
                scored_fasta_df, 
                scored_fasta_df_padding
            ])
            .sample(frac = 1.0, random_state = seed)
            .copy()
            .reset_index()
        )

        # Get validation and training batch amounts
        num_batches = padded_scored_fasta_df.shape[0]//batch_size
        num_validation_batches = np.max([int(np.round(validation_fraction * num_batches)), 1])
        num_training_batches = num_batches - num_validation_batches
        

        # print(f'{num_validation_batches} validation batches')
        # print(f'{num_training_batches} training batches')
        
        if np.min([num_validation_batches, num_training_batches]) < 1:
            print(
                (
                    'Not enough batches to have validation and training datasets. '
                    'Re-run analysis with a smaller batch size.'
                ),
                file = sys.stderr
            )
        
        # Load profile
        profile_df = pd.read_csv(
            profile_filepath,
            sep = '\t'
        )
        
        if 'positional_r' in list(profile_df.columns):
            profile_df['profile'] = profile_df['positional_r']
        
        
        # Align profile to current dataset positions
        max_sequence_length = scored_fasta_df['sequence'].map(len).max()
        positions = np.array(list(range(max_sequence_length)))
        
        if center is None:
            center = max_sequence_length//2
        positions = (positions - center)
        
        positions_df = pd.DataFrame({'position':positions})
        
        profile_df = (
            positions_df
            .merge(profile_df[['position', 'profile']], how = 'left')
            .bfill()
            .ffill()
        )
        
        # Extract positional profile to scale with scores
        positional_profile = tf.constant(profile_df['profile'])
        # positional_r_vec = tf.constant(profile_df['profile'])
        # positional_profile = positional_r_vec
        
        # Shift profiles to increase motif diversity
        if profile_shift > 0:
            print('Shift profiles')
            (
                positional_profiles, 
                profile_df,
                num_shifted_profiles,
                offset_profile_cols
            ) = get_shifted_positional_profiles(
                profile_df, 
                num_motifs,
                -profile_shift,
                profile_shift
            )
            print(f'num_motifs = {num_motifs}')
            positional_profile = positional_profiles
            num_motifs = positional_profile.shape[-1]
            print(f'num_motifs = {num_motifs}')
        print(f'profile_shift = {profile_shift}')
        print('Profile shape:')
        print(positional_profile.shape)
        
        # Create dataset
        original_dataset = scored_fasta_df_to_dataset(
            padded_scored_fasta_df,
            batch_size = batch_size,
            convert_masked_seq = convert_masked_seq_with_underscores,
            n_jobs = 1
        )

#         revcomp_augmented_dataset = revcomp_augment_dataset(
#             original_dataset, 
#             batch_size = batch_size, 
#             random_seed = seed
#         )
#         dataset_rev = revcomp_dataset(
#             revcomp_augmented_dataset, 
#             batch_size = batch_size
#         )
#         dataset = (
#             revcomp_augmented_dataset
#             .concatenate(dataset_rev)
#             .prefetch(tf.data.AUTOTUNE)
#             .cache()
#         )
        
        # No reverse complementation
        dataset = (
            original_dataset
            .prefetch(tf.data.AUTOTUNE)
            .cache()
        )

        
        # Save dataset
        save_dataset(dataset, dataset_filepath)

        # Split dataset into validation and training
        # validation_dataset = (
        #     positionalize_dataset_scores(
        #         reformat_dataset(
        #             load_dataset(dataset_filepath)
        #         ), 
        #         positional_profile
        #     )
        #     .take(num_validation_batches) 
        # )
        training_dataset = (
            positionalize_dataset_scores(
                reformat_dataset(
                    load_dataset(dataset_filepath)
                ), 
                positional_profile
            )
            # .skip(num_validation_batches)
        )
       
        # Determine input layer properties
        sequences, scores = (
            positionalize_dataset_scores(
                reformat_dataset(
                    load_dataset(dataset_filepath)
                ),
                positional_profile
            )
            .take(1)
            .get_single_element()
        )
        input_layer = (
            keras.layers.Input(
                type_spec = tf.TensorSpec.from_tensor(sequences)
            )
        )

        # Generate model
        
        unstranded = False
        if orientation is '+/-':
            print('Assuming data is unstranded. Will learn motifs with orientation invariance.')
            unstranded = True
        
        model_type = 'positional_conv'
        generate_model = generate_positional_conv_model
        model_name = f'{model_type}_seed_{seed}_{num_motifs}_motifs_{motif_length}bp_margin_{motif_margin}'
        model, conv_model, post_conv_model = generate_model(
            input_layer, 
            num_motifs, 
            motif_length, 
            seed,
            motif_margin,
            unstranded
        )
        
        # Print model summaries
        print('model summary')
        model.summary()
        print('conv_model summary')
        conv_model.summary()
        print('post_conv_model summary')
        post_conv_model.summary()
        
        # Print score dimension
        print(scores.shape)
        
        # Get model output dimensions
        num_model_output_dims = len(list(model.layers[-1].output_shape))
        num_score_dims = len(list(scores.shape))

        # Match model output dimensions
        for i in list(range(num_model_output_dims-num_score_dims)):
            # validation_dataset = validation_dataset.map(
            #     lambda sequence, score: (sequence, tf.expand_dims(score, -1)), 
            #     num_parallel_calls=tf.data.AUTOTUNE
            # )
            training_dataset = training_dataset.map(
                lambda sequence, score: (sequence, tf.expand_dims(score, -1)), 
                num_parallel_calls=tf.data.AUTOTUNE
            )


        # validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE).cache()
        training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE).cache()

        # Train model
        callbacks = [
            keras.callbacks.experimental.BackupAndRestore(
                normpath(f'{out_filepath}/{model_name}.model_backups/')
            ),
            keras.callbacks.ModelCheckpoint(
                normpath(f'{out_filepath}/{model_name}.model_checkpoints/')
            ),
            keras.callbacks.CSVLogger(
                normpath(f'{out_filepath}/{model_name}.training_log.csv')
            )
        ]

        if early_stopping:
            print('Training with early stopping', file = sys.stderr )
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor = 'loss',
                patience = early_stopping_patience, 
                min_delta = early_stopping_min_delta
            ))


        print('Training model', file = sys.stderr )
        model.fit(
            training_dataset,
            epochs=epochs,
            # validation_data=validation_dataset,
            callbacks=callbacks,
            verbose = 'auto',
        )

        # Extract motifs
        print('Extracting motifs', file = sys.stderr )
        motifs_, information_matrices = amplify_motifs(
            extract_motifs(
                conv_model, 
                dataset, 
                n_jobs = n_jobs, 
                unstranded = unstranded
            )
        )
        motifs = filter_motifs_by_information_content(
            motifs_, 
            information_matrices, 
            min_information_content = 0.0
        )

        # Format motifs
        motif_matrix_dict = motifs_to_dict(motifs, motif_prefix)

        # Write motifs to file
        print('Writing motifs', file = sys.stderr )
        denovo_motifs_filepath = normpath(
            f'{out_filepath}/{model_name}.denovo_motifs.txt'
        )

        with open(denovo_motifs_filepath, 'w') as f:
            motif_matrix_dict_to_file(motif_matrix_dict, f)
    
    with redirect_stdout(sys.stdout):
        if quiet == False:
            with open(denovo_motifs_filepath) as f:
                print(f.read(), file = sys.stdout)
    
    with redirect_stdout(sys.stderr):
        # Write motif logos to file
        print('Writing motif logos', file = sys.stderr )
        motif_matrix_df = get_logo_df(motif_matrix_dict, n_jobs = n_jobs)
        denovo_motifs_html_filepath = normpath(
            f'{out_filepath}/{model_name}.denovo_motifs.html'
        )

        denovo_motifs_logo_filepath = normpath(
            f'{out_filepath}/{model_name}.denovo_motifs.pkl'
        )

        header_include_html = '\n'.join([
            '<link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.0.1/css/bootstrap.min.css"/>'
            '<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.0.1/js/bootstrap.bundle.min.js"></script>'

        ])
        title = f'Logos for {denovo_motifs_filepath}'
        head_html = f'<meta charset="UTF-8"><title>{title}</title>{header_include_html}'
        html = (
            f'''<!DOCTYPE html><head>{head_html}</head>'''
            f'''<body>{motif_matrix_df.style.render()}</body></html>'''
            f'\n'
        )

        with open(denovo_motifs_html_filepath, 'w') as f:
            f.write(html)

        motif_matrix_df.to_pickle(denovo_motifs_logo_filepath)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover   
