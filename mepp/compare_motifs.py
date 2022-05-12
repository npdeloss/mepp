"""Helper script to compare known and denovo motifs"""

import os
import sys
import click
import multiprocessing

from os.path import normpath

import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .io import motif_matrix_file_to_dicts

from .plot import motif_matrix_to_df
from .html import get_logo_df
from .io import motif_matrix_file_to_dicts
from .html import motif_matrix_to_logo_data_uri
from .html import get_interactive_table_html
from .learn_motifs import motif_matrix_dict_to_file
from itertools import combinations

def get_html_logo_for_motif_matrix(motif_matrix):
    data_uri = motif_matrix_to_logo_data_uri(motif_matrix)
    return f'<img src="{data_uri}">'

def offset_motifs(motif_a, motif_b, offset, reverse_complement = False):
    if reverse_complement == True:
        return offset_motifs(motif_a, motif_b[::-1,::-1], offset, reverse_complement = False)
    offset_a = -offset if offset < 0 else 0
    offset_b = offset if offset > 0 else 0

    motif_a_length = motif_a.shape[1]
    motif_b_length = motif_b.shape[1]
    motif_width = motif_a.shape[0]

    left_pad_a = np.ones((motif_width, offset_a)) * (1.0 / motif_width)
    left_pad_b = np.ones((motif_width, offset_b)) * (1.0 / motif_width)

    left_padded_a_length = offset_a + motif_a_length
    left_padded_b_length = offset_b + motif_b_length
    left_padded_max_length = np.max((left_padded_a_length, left_padded_b_length))

    right_pad_a_length = np.abs(left_padded_a_length - left_padded_max_length)
    right_pad_b_length = np.abs(left_padded_b_length - left_padded_max_length)

    right_pad_a = np.ones((motif_width, right_pad_a_length)) * (1.0 / motif_width)
    right_pad_b = np.ones((motif_width, right_pad_b_length)) * (1.0 / motif_width)

    motif_a_padded = np.concatenate((left_pad_a, motif_a, right_pad_a), axis = 1)
    motif_b_padded = np.concatenate((left_pad_b, motif_b, right_pad_b), axis = 1)

    left_clip = np.max((offset_a, offset_b))
    right_clip = np.min((left_padded_a_length, left_padded_b_length))

    motif_a_clipped = motif_a_padded[:, left_clip : right_clip]
    motif_b_clipped = motif_b_padded[:, left_clip : right_clip]
    
    corrcoef = np.corrcoef(motif_a_clipped.flatten(), motif_b_clipped.flatten())[0][1]
    
    return motif_a_padded, motif_b_padded, motif_a_clipped, motif_b_clipped, corrcoef

def align_motifs(motif_a, motif_b, min_overlap = 6):
    motif_a_length = motif_a.shape[1]
    motif_b_length = motif_b.shape[1]
    max_offset = motif_a_length - min_overlap
    min_offset = min_overlap - motif_b_length
    offsets = list(range(min_offset, max_offset + 1))

    orientation_to_reverse_complement = {
        'forward': False, 
        'reverse': True
    }

    best_alignment = (
        (
            0, 
            'forward'
        ) + 
        offset_motifs(
            motif_a, 
            motif_b, 
            offset = 0, 
            reverse_complement = orientation_to_reverse_complement['forward']
        )
    )

    best_alignment_score = best_alignment[6]

    for orientation in orientation_to_reverse_complement:
        for offset in offsets:
            current_alignment = (
                (
                    offset, 
                    orientation) + 
                offset_motifs(
                    motif_a, 
                    motif_b, 
                    offset = offset, 
                    reverse_complement = orientation_to_reverse_complement[orientation]
                )
            )

            current_alignment_score = current_alignment[6]

            if current_alignment_score > best_alignment_score:
                best_alignment = current_alignment
                best_alignment_score = best_alignment[6]

    return best_alignment


matsoftmax = lambda x: softmax(x, axis = 0)

def combine_motif_matrices(x1, x2):
    x1_m = x1.min(axis = 0, keepdims = True)
    x2_m = x2.min(axis = 0, keepdims = True)
    x1_d = x1 - x1_m
    x2_d = x2 - x2_m
    x3 = (x1_d + x2_d)
    # x3 = x3 + np.array([x1_m, x2_m]).min(axis = 0)
    x3 = x3 + np.ones(x3.shape) * np.finfo(x3.dtype).eps
    x3 = x3 / x3.sum(axis = 0, keepdims = True)
    return x3

def compare_motif_dicts(
    known_motif_matrix_dict, 
    denovo_motif_matrix_dict = None, 
    min_overlap = 6, 
    min_corrcoef = 0.6, 
    logos = True,
    combine = False,
    n_jobs = 1, 
    progress_wrapper = tqdm
):
    if denovo_motif_matrix_dict != None:
        known_and_denovo_pairings = [
            (
                known_motif_id, 
                denovo_motif_id
            ) 
            for known_motif_id 
            in known_motif_matrix_dict 
            for denovo_motif_id 
            in denovo_motif_matrix_dict
        ]
    else:
        print('Detected self comparison of motif dictionary against self. Only comparing unique combinations.', file = sys.stderr)
        denovo_motif_matrix_dict = known_motif_matrix_dict
        known_and_denovo_pairings = list(combinations(list(known_motif_matrix_dict.keys()), 2))
    
#     def get_alignment_from_pairing(pairing):
#         known_motif_id, denovo_motif_id = pairing

#         known_motif_matrix = known_motif_matrix_dict[known_motif_id]
#         denovo_motif_matrix = denovo_motif_matrix_dict[denovo_motif_id]
        
#         retval = (
#             (
#                 known_motif_id, 
#                 denovo_motif_id
#             ) + 
#             align_motifs(
#                 known_motif_matrix, 
#                 denovo_motif_matrix, 
#                 min_overlap = min_overlap
#             )
#         )
        
#         return retval
    
    def get_alignment_from_pairing(
        a_id,
        b_id,
        a,
        b,
        min_overlap
    ):
        retval = (
            (
                a_id, 
                b_id
            ) + 
            align_motifs(
                a, 
                b, 
                min_overlap = min_overlap
            )
        )
        
        return retval
    known_and_denovo_alignment_results_tups = (
        Parallel(n_jobs = n_jobs)(
            delayed(
                get_alignment_from_pairing
            )(a_id, b_id, a,b,min_overlap) 
            for a_id, b_id, a,b 
            in progress_wrapper([
                (
                    a_id,
                    b_id,
                    known_motif_matrix_dict[a_id], 
                    denovo_motif_matrix_dict[b_id]
                )
                for a_id, b_id
                in known_and_denovo_pairings
            ])
        )
    )
    
    known_and_denovo_alignment_results_df = pd.DataFrame(
        known_and_denovo_alignment_results_tups, 
        columns = [
            'known_motif_id', 
            'denovo_motif_id', 
            'offset', 
            'orientation', 
            'known_motif_matrix_padded', 
            'denovo_motif_matrix_padded', 
            'known_motif_matrix_clipped', 
            'denovo_motif_matrix_clipped', 
            'corrcoef'
        ]
    )
    
    filtered_known_and_denovo_alignment_results_df = (
        known_and_denovo_alignment_results_df[
            known_and_denovo_alignment_results_df['corrcoef'] >= min_corrcoef
        ]
        .sort_values(by = 'corrcoef', ascending = False)
        .reset_index(drop = True)
        .copy()
    )
    
    if logos:
    
        filtered_known_and_denovo_alignment_results_df['known_motif_matrix_padded_str'] = (
            filtered_known_and_denovo_alignment_results_df['known_motif_matrix_padded']
            .map(np.array2string)
        )
        filtered_known_and_denovo_alignment_results_df['denovo_motif_matrix_padded_str'] = (
            filtered_known_and_denovo_alignment_results_df['denovo_motif_matrix_padded']
            .map(np.array2string)
        )

        known_filtered_df = (
            filtered_known_and_denovo_alignment_results_df[[
                'known_motif_matrix_padded_str', 
                'known_motif_matrix_padded'
            ]]
            .drop_duplicates([
                'known_motif_matrix_padded_str'
            ])
        )
        denovo_filtered_df = (
            filtered_known_and_denovo_alignment_results_df[[
                'denovo_motif_matrix_padded_str', 
                'denovo_motif_matrix_padded'
            ]].drop_duplicates([
                'denovo_motif_matrix_padded_str'
            ])
        )

        known_filtered_df['known_motif_matrix_padded_logo'] = Parallel(n_jobs = n_jobs)(
            delayed(get_html_logo_for_motif_matrix)(matrix) 
            for matrix 
            in progress_wrapper(
                list(
                    known_filtered_df['known_motif_matrix_padded']
                )
            )
        )

        denovo_filtered_df['denovo_motif_matrix_padded_logo'] = Parallel(n_jobs = n_jobs)(
            delayed(get_html_logo_for_motif_matrix)(matrix) 
            for matrix 
            in progress_wrapper(
                list(
                    denovo_filtered_df['denovo_motif_matrix_padded']
                )
            )
        )
        
        filtered_known_and_denovo_alignment_results_df = (
            filtered_known_and_denovo_alignment_results_df
            .merge(
                known_filtered_df.drop(
                    'known_motif_matrix_padded', 
                    axis = 1
                ), 
                on = ['known_motif_matrix_padded_str']
            )
            .merge(
                denovo_filtered_df.drop(
                    'denovo_motif_matrix_padded', 
                    axis = 1
                ), 
                on = ['denovo_motif_matrix_padded_str']
            )
            .sort_values(by = 'corrcoef', ascending = False).reset_index(drop = True)
        )
        filtered_known_and_denovo_alignment_results_df['motif_alignment_logo'] = (
            filtered_known_and_denovo_alignment_results_df['known_motif_matrix_padded_logo'] + 
            '<br>' + 
            filtered_known_and_denovo_alignment_results_df['denovo_motif_matrix_padded_logo']
        )
        
        filtered_known_and_denovo_alignment_results_df = filtered_known_and_denovo_alignment_results_df.drop(
            [
                'known_motif_matrix_padded_str',
                'denovo_motif_matrix_padded_str'
            ], 
            axis = 1
        )
    
    if combine:
        filtered_known_and_denovo_alignment_results_df['combined_motif_matrix'] = (
            filtered_known_and_denovo_alignment_results_df[[
                'known_motif_matrix_padded',
                'denovo_motif_matrix_padded'
            ]]
            .apply(tuple, axis = 1)
            .map(lambda x: combine_motif_matrices(*x))
        )
        if logos:
            filtered_known_and_denovo_alignment_results_df['combined_motif_matrix_logo'] = Parallel(n_jobs = n_jobs)(
                delayed(get_html_logo_for_motif_matrix)(matrix) 
                for matrix 
                in progress_wrapper(
                    list(
                        filtered_known_and_denovo_alignment_results_df['combined_motif_matrix']
                    )
                )
            )
    
    return filtered_known_and_denovo_alignment_results_df



@click.command()
@click.option(
    '--motifs',
    'motifs_filepath',
    type = str,
    required = True,
    help = (
        'Path to a motif matrices file in JASPAR format. '
        'Preferably a denovo motif matrices file. '
        'if --known-motifs is not specified, '
        'this will be compared against itself. '
        'As a start, one can be obtained through the JASPAR website at: '
        'http://jaspar.genereg.net/downloads/'
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
@click.option(
    '--known-motifs',
    'known_motifs_filepath',
    type = str,
    default = None,
    help = (
        'Path to a known motif matrices file in JASPAR format.'
        'As a start, one can be obtained through the JASPAR website at: '
        'http://jaspar.genereg.net/downloads/ '
        'Default: None'
    )
)
@click.option(
    '--overlap',
    'min_overlap',
    type = int,
    default = 6,
    help = (
        'Minimum overlap for correlated motifs. '
        'Default: 6'
    )
)
@click.option(
    '--corrcoef',
    'min_corrcoef',
    type = float,
    default = 0.6,
    help = (
        'Minimum correlation for correlated motifs. '
        'Default: 0.6'
    )
)
@click.option(
    '--combine',
    'combine',
    default = False,
    flag_value = True,
    help = (
        'Combine motifs. '
        'Default: Do not combine motifs.'
    )
)
@click.option(
    '--motif-prefix',
    'motif_prefix',
    type = str,
    default = 'combined_motif_',
    help = (
        'Prefix motif names with this string.'
        'Default: combined_motif_'
    )
)
@click.option(
    '--no-logos',
    'logos',
    default = True,
    flag_value = False,
    help = (
        'Do not render logos. '
        'Default: Render logos.'
    )
)
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
    motifs_filepath,
    out_filepath,
    known_motifs_filepath = None,
    min_overlap = 6,
    min_corrcoef = 0.6,
    combine = False,
    motif_prefix = 'combined_motif_',
    logos = True,
    n_jobs = multiprocessing.cpu_count(),
    quiet = False
):
    os.makedirs(
        normpath(out_filepath), 
        exist_ok = True
    )
    
    with open(motifs_filepath) as f:
        motif_matrix_dict = motif_matrix_file_to_dicts(f)[0]
    
    if known_motifs_filepath is None:
        known_motifs_filepath = motifs_filepath
        with open(known_motifs_filepath) as f:
            known_motif_matrix_dict = motif_matrix_file_to_dicts(f)[0]
        motifs_filepath = None
        motif_matrix_dict = None
    else:
        with open(known_motifs_filepath) as f:
            known_motif_matrix_dict = motif_matrix_file_to_dicts(f)[0]
        
    filtered_known_and_denovo_alignment_results_df = compare_motif_dicts(
        known_motif_matrix_dict, 
        motif_matrix_dict, 
        min_overlap = min_overlap, 
        min_corrcoef = min_corrcoef, 
        logos = logos,
        combine = combine,
        n_jobs = n_jobs
    )

    remove_matrix_cols = lambda df: df[[
        col 
        for col 
        in list(df.columns) 
        if (
            (col.endswith('_matrix')==False) and 
            (col.endswith('_matrix_clipped')==False) and 
            (col.endswith('_matrix_padded')==False)
        )
    ]]
    
    filtered_known_and_denovo_alignment_results_display_df = (
        remove_matrix_cols(
            filtered_known_and_denovo_alignment_results_df
        )
    )
    
    if combine:
        (
            filtered_known_and_denovo_alignment_results_df
            .insert(
                3, 
                'combined_motif_id', 
                list(range(
                    1,
                    filtered_known_and_denovo_alignment_results_df.shape[0]+1
                ))
            )
        )
        filtered_known_and_denovo_alignment_results_df['combined_motif_id'] = (
            f'{motif_prefix}' + 
            (
                filtered_known_and_denovo_alignment_results_df['combined_motif_id']
                .astype(str)
            )
        )
        combined_motif_matrix_dict = (
            filtered_known_and_denovo_alignment_results_df
            [['combined_motif_id','combined_motif_matrix']]
            .copy()
            .set_index('combined_motif_id')
            ['combined_motif_matrix']
            .to_dict()
        )
        combined_motifs_filepath = normpath(
            f'{out_filepath}/combined_motifs.txt'
        )
        
        with open(combined_motifs_filepath, 'w') as f:
            motif_matrix_dict_to_file(
                combined_motif_matrix_dict, 
                f
            )
        if quiet == False:
            with open(combined_motifs_filepath) as f:
                print(f.read())

    
    html = get_interactive_table_html(
        filtered_known_and_denovo_alignment_results_display_df,
        filtered_known_and_denovo_alignment_results_display_df.style,
        title = (
            f'Motif comparison'
        ),
        search_panes_cols = [
            'offset',
            'orientation'
        ],
        hidden_cols = []
    )
    
    html_filepath = normpath(f'{out_filepath}/motif_comparison.html')
    
    with open(html_filepath, 'w') as f:
        f.write(html)
    
if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover    