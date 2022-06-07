"""Helper script to query MEPP motifs for correlation against a positional profile"""

import os
import sys
import click
import multiprocessing

from os.path import normpath

from contextlib import redirect_stdout

import pandas as pd
import numpy as np

from .batch import orientation_to_filepath
from .html import get_interactive_table_html

import multiprocessing
from joblib import Parallel, delayed
from tqdm.auto import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt
from .plot import plot_sparkline
from .html import fig_to_bitmap_data_uri, get_interactive_table_html, style_results_df
from .batch import motif_id_and_orientation_to_filepath
from scipy.stats import zscore
from slugify import slugify

def get_mepp_profile_dfs(mepp_filepath):
    mepp_filepaths_df = pd.read_csv(
        normpath(f'{mepp_filepath}/filepaths.tsv'),
        sep = '\t'
    ).set_index(['motif_id', 'orientation'])
    mepp_filepaths_df['profile_filepath'] = (
        mepp_filepaths_df['outdir'] + 
        '/positions_df.pkl'
    ).map(normpath)
    mepp_filepaths_df['profile_filepath_exists'] = (
        mepp_filepaths_df['profile_filepath']
        .map(os.path.exists)
    )

    mepp_profile_dfs = (
        mepp_filepaths_df[
            mepp_filepaths_df['profile_filepath_exists']
        ]
        ['profile_filepath']
        .map(pd.read_pickle).to_dict()
    )
    return mepp_profile_dfs

def correlate_mepp_and_target_profiles(
    mepp_profile_df, 
    target_profile_df, method = 'pearson'
):
    return  (
        mepp_profile_df
        .merge(
            target_profile_df, 
            how = 'left'
        )
        .bfill()
        .ffill()
        [[
            'positional_r', 
            'profile'
        ]]
        .corr(method = method)
        .values
        .flatten()[1]
    )

def get_mepp_target_profile_correlations(
    mepp_profile_dfs, 
    target_profile_df, 
    method = 'pearson', 
    n_jobs = 1
):
    vs = Parallel(n_jobs = n_jobs)(
        delayed(correlate_mepp_and_target_profiles)
        (df, target_profile_df) 
        for k, df
        in tqdm(list(
            mepp_profile_dfs.items()
        ))
    )
    
    ks = [k for k,v in mepp_profile_dfs.items()]
    
    mepp_profile_correlations = {
        k: v
        for k, v
        in zip(ks, vs)
    }
    
    return mepp_profile_correlations

def get_result_table_dfs(mepp_filepath):
    mepp_filepaths_df = pd.read_csv(
        normpath(f'{mepp_filepath}/filepaths.tsv'),
        sep = '\t'
    )
    orientations = sorted(list(set(mepp_filepaths_df['orientation'])))
    result_table_df_filepaths_by_orientation = {
        orientation: normpath(
            f'{mepp_filepath}/results_table_orientation_' +
            suffix +
            '.pkl'
        )
        for orientation, suffix
        in orientation_to_filepath.items()
        if orientation in orientations
    }
    result_table_dfs_by_orientation = {
        orientation: pd.read_pickle(fp)
        for orientation, fp
        in result_table_df_filepaths_by_orientation.items()
        if os.path.exists(fp)
    }
    return result_table_dfs_by_orientation

def positional_dfs_to_comparison_sparkline_data_uri(
    positional_r_df,
    target_profile_df,
    xlims = None,
    ylims = None,
    figure_width = 5,
    figure_height = 1,
    figsize = None,
    dpi = 100,
    format = 'png',
    target_profile_line_kwargs = dict(color='tab:orange',alpha=1.0),
    target_profile_fill_kwargs = dict(color='tab:orange',alpha=0.25),
    **kwargs
):
    if figsize is None:
        figsize = (figure_width, figure_height)
    fig = plt.figure(figsize = figsize, dpi = dpi)
    fig.subplots_adjust(
        left = 0,
        right = 1,
        bottom = 0,
        top = 1,
        wspace = 0,
        hspace = 0
    )
    ax = fig.add_subplot()

    scaled_target_profile_df = (
        positional_r_df[['position', 'positional_r']]
        .merge(target_profile_df[['position', 'profile']], how = 'left')
        .bfill()
        .bfill()
    )
    
    positional_r_range = (
        scaled_target_profile_df['positional_r'].max() - 
        scaled_target_profile_df['positional_r'].min()
    )
    
    profile_range = (
        scaled_target_profile_df['profile'].max() - 
        scaled_target_profile_df['profile'].min()
    )
    if profile_range > 0:
        scaled_target_profile_df['scaled_profile'] = (
            (scaled_target_profile_df['profile'] - scaled_target_profile_df['profile'].min()) /
            profile_range
        )
        scaled_target_profile_df['scaled_profile'] = (
            scaled_target_profile_df['scaled_profile'] * 
            positional_r_range
        ) + scaled_target_profile_df['positional_r'].min()
    else:
        scaled_target_profile_df['scaled_profile'] = (
            scaled_target_profile_df['profile'].mean()
        )
    
    top = np.max([
        positional_r_df['positional_r'].max(),
        scaled_target_profile_df['scaled_profile'].max()
    ])
    
    bottom = np.min([
        positional_r_df['positional_r'].min(),
        scaled_target_profile_df['scaled_profile'].min()
    ])
    
    ylims = [bottom, top]
    
    sparkline = plot_sparkline(
        positional_r_df,
        xlims = xlims,
        ylims = ylims,
        ax = ax,
        figsize = figsize,
        **kwargs
    )
    
    ax.plot(
        scaled_target_profile_df['position'],
        scaled_target_profile_df['scaled_profile'],
        **target_profile_line_kwargs
        
    )
    
    ax.fill_between(
        scaled_target_profile_df['position'],
        scaled_target_profile_df['scaled_profile'],
        np.zeros(len(scaled_target_profile_df['position'])),
        **target_profile_fill_kwargs
    )
    
    ax.get_xaxis().set_visible(False)
    data_uri = fig_to_bitmap_data_uri(
        fig,
        format = format,
        dpi = dpi,
        savefig_kwargs = dict(
            transparent=True,
            bbox_inches=0
        )
    )
    plt.close(fig)
    return data_uri

def make_profile_comparison_sparklines_data_uris(
    mepp_profile_dfs,
    target_profile_df,
    n_jobs = 1
):
    vs = Parallel(n_jobs = n_jobs)(
        delayed(
            positional_dfs_to_comparison_sparkline_data_uri
        )(df, target_profile_df) 
        for k, df
        in tqdm(list(mepp_profile_dfs.items()))
    )
    
    ks = [k for k,v in mepp_profile_dfs.items()]
    
    data_uris = {
        k: v
        for k,v
        in zip(ks, vs)
    }
    return data_uris

def compare_mepp_filepath_to_target_profile_df(
    mepp_filepath,
    target_profile_df,
    method = 'pearson',
    n_jobs = 1
):
    # Calculate profile correlations against target
    mepp_profile_dfs = get_mepp_profile_dfs(mepp_filepath)
    mepp_profile_correlations = get_mepp_target_profile_correlations(
        mepp_profile_dfs, 
        target_profile_df,
        method = method, 
        n_jobs = n_jobs
    )
    
    # Generate comparison sparklines
    comparison_sparkline_data_uris = make_profile_comparison_sparklines_data_uris(
        mepp_profile_dfs,
        target_profile_df,
        n_jobs = n_jobs
    )
    
    # Collect result tables for all motif orientations
    result_table_dfs = get_result_table_dfs(mepp_filepath)
    result_table_df = (
        pd.concat(result_table_dfs)
        .reset_index()
        .rename(columns={'level_0':'orientation'})
        .drop('level_1', axis = 1)
        .drop(['clustering_order', 'dendrogram'], axis = 1)
    )
    profile_keys = (
        result_table_df[['motif_id', 'orientation']]
        .apply(tuple, axis = 1)
    )
    result_table_df['correlation_with_target'] = (
        profile_keys
        .map(mepp_profile_correlations)
    )
    result_table_df['comparison_sparkline'] = (
        profile_keys
        .map(comparison_sparkline_data_uris)
        .map(lambda x: f'<img src="{x}">')
    )
    
    result_table_df['plot_filepath'] = (
        profile_keys
        .map(
            lambda x: motif_id_and_orientation_to_filepath(*x)
        )
        .map(lambda x: tuple((
            f'{x}/mepp_plot.{ext}'
            for ext
            in ['png','svg']
        )))
        .map(lambda x: (
            x + 
            tuple((
                os.path.exists(normpath(
                    f'{mepp_filepath}/{x_i}'
                ))
                for x_i
                in x
            ))
        ))
        .map(
            lambda x: x[0] if x[2] else x[1] if x[3] else '#'
        )
    )

    result_table_df = (
        result_table_df
        .sort_values(
            by = 'correlation_with_target', 
            ascending = False
        )
        .reset_index(drop = True)
    )
    profile_comparison_df = result_table_df[[
        'motif_id',
        'orientation',
        'logo',
        'correlation_with_target',
        'comparison_sparkline',
        'extreme_r',
        'extreme_r_pos',
        'abs_extreme_r',
        'extreme_r_padj',
        'extreme_r_sig',
        'min_r',
        'min_r_pos',
        'max_r',
        'max_r_pos',
        'plot_filepath'
    ]].copy()
    return profile_comparison_df

def profile_comparison_df_to_html(
    profile_comparison_df, 
    query_name = 'Given profile on MEPP results'
):
    result_table_df = profile_comparison_df
    linked_result_table_df = (
        result_table_df
        .copy()
        .drop(
            'plot_filepath', 
            axis = 1
        )
    )

    linked_result_table_df_style = style_results_df(
        linked_result_table_df,
        image_cols = [
            'logo',
            'comparison_sparkline'
        ],
        signed_cols = ['extreme_r'],
        unsigned_cols = ['abs_extreme_r'],
        sig_cols = [
            'extreme_r_sig',
        ]
    )


    linked_result_table_df_style = (
        linked_result_table_df_style
        .bar(
            subset = 'correlation_with_target',
            color = '#ade6bb',
            align =  'left',
            vmin = 0,
            vmax = (
                linked_result_table_df
                ['correlation_with_target']
                .max()
            )
        )
    )

    link_cols = [
        'motif_id',
        'logo',
        'comparison_sparkline'
    ]

    for link_col in link_cols:
        links = (
            result_table_df['plot_filepath']
            .map(lambda x: f'<a href="{x}">') +
            linked_result_table_df[link_col] +
            '</a>'
        )
        linked_result_table_df[link_col] = links

    html = get_interactive_table_html(
        linked_result_table_df,
        linked_result_table_df_style,
        title = f'Profile query: {query_name}',
        search_panes_cols = [
            'orientation',
            'extreme_r_sig'
        ],
        hidden_cols = [
            'abs_extreme_r',
            'extreme_r_padj',
            'extreme_r_sig',
            'min_r',
            'min_r_pos',
            'max_r',
            'max_r_pos',
        ]
    )
    
    return html

@click.command()
@click.option(
    '--mepp',
    'mepp_filepath',
    type = str,
    required = True,
    help = (
        'Path to a MEPP run directory '
        'to compare profiles from.'
    )
)
@click.option(
    '--profile',
    'profile_filepath',
    type = str,
    required = True,
    help = (
        'Path to a tab-separated text file '
        'with two columns: position, and profile'
        'describing positions along a sequence '
        'and the target profile for enriching a motif '
        'at those positions. '
    )
)
@click.option(
    '--method',
    'correlation_method',
    type = click.Choice([
        'pearson',
        'spearman'
    ], case_sensitive = False),
    default = 'pearson',
    help = (
        'Type of correlation to use for comparison. '
        'Default: pearson'
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
def main(
    mepp_filepath,
    profile_filepath,
    query_name = None,
    correlation_method = 'pearson',
    n_jobs = multiprocessing.cpu_count()
):
    # Set default query name from profile filepath
    if query_name is None:
        query_name = (
            os.path.basename(
                os.path.splitext(profile_filepath)[0]
            )
        )
    
    # Load target profile
    target_profile_df = pd.read_csv(
        normpath(profile_filepath), 
        sep = '\t'
    )
    if 'positional_r' in list(target_profile_df.columns):
        target_profile_df['profile'] = target_profile_df['positional_r']
    target_profile_df = target_profile_df[['position', 'profile']].copy()
    
    # Generate table and HTML
    profile_comparison_df = compare_mepp_filepath_to_target_profile_df(
        normpath(mepp_filepath),
        target_profile_df,
        method = slugify(correlation_method),
        n_jobs = n_jobs
    )
    html = profile_comparison_df_to_html(
        profile_comparison_df, 
        query_name = query_name
    )
    
    # Write outputs into MEPP directory
    
    base_filepath = normpath(f'{mepp_filepath}/query_{slugify(query_name)}')
    tsv_filepath = f'{base_filepath}.tsv'
    pkl_filepath = f'{base_filepath}.pkl'
    html_filepath = f'{base_filepath}.html'

    profile_comparison_df.to_csv(tsv_filepath, sep = '\t', index = False)
    profile_comparison_df.to_pickle(pkl_filepath)
    with open(html_filepath, 'w') as f:
        f.write(html)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover    
