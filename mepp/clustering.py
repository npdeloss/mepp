import os
import io

import multiprocessing

import base64
from io import (
    BytesIO,
    StringIO
)

from os.path import normpath

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from joblib import Parallel, delayed

from scipy.cluster.hierarchy import (
    linkage,
    to_tree,
    cut_tree,
    dendrogram
)

from .batch import (
    orientation_to_filepath
)

from .utils import(
    filepaths_df_to_profile_dicts,
    get_minmax_stats_df,
    re_prefix_filepath_dict
)

from .html import(
    fig_to_bitmap_data_uri,
    get_logo_df,
    get_sparkline_df,
    link_results_df,
    style_results_df,
    get_interactive_table_html,
    get_hover_logo_df,
    get_hover_heatmap_df,
    join_clustermap_df,
    get_clustermap_html
)

def get_linkage_matrix(
    motif_id_to_profile,
    method = 'average',
    metric = 'correlation'
):
    motif_ids_, profiles_ = zip(*list(motif_id_to_profile.items()))
    labels = motif_ids = list(motif_ids_)
    profiles = np.nan_to_num(np.array(profiles_))

    Z = linkage(
        profiles,
        method = method,
        metric = metric
    )
    return Z, labels

def get_sub_dendrogram_data_uri(
    Z,
    labels,
    pos,
    dpi = 100,
    figsize = None,
    ypad = 5,
    format = 'png',
    dendrogram_kwargs = dict(
        p=30,
        truncate_mode=None,
        color_threshold=None,
        get_leaves=True,
        count_sort=False,
        distance_sort=False,
        show_leaf_counts=True,
        leaf_font_size=None,
        leaf_rotation=None,
        leaf_label_func=None,
        show_contracted=False,
        link_color_func=None,
        above_threshold_color='C0'
    )
):
    subfig = plt.figure(
        figsize = figsize,
        dpi = dpi
    )
    subfig.subplots_adjust(
        left = 0,
        right = 1,
        bottom = 0,
        top = 1,
        wspace = 0,
        hspace = 0
    )
    subax = subfig.add_subplot()
    subd = dendrogram(
        Z,
        ax = subax,
        labels = labels,
        no_labels = False,
        orientation='left',
        **dendrogram_kwargs
    )
    subax.set_ylim(
        pos - ypad,
        pos + ypad
    )
    for k,v in subax.spines.items():
        v.set_visible(False)
    subax.get_xaxis().set_visible(False)
    subax.get_yaxis().set_visible(False)
    data_uri = fig_to_bitmap_data_uri(
        subfig,
        format = format,
        savefig_kwargs = dict(
            transparent=True,
            bbox_inches=0
        )
    )
    plt.close(subfig)
    return data_uri

def get_dendrogram_df(
    Z,
    labels,
    figure_width = 3,
    figure_height = 1.25,
    dpi = 100,
    figsize = None,
    fig = None,
    ax = None,
    format = 'png',
    n_jobs = multiprocessing.cpu_count(),
    progress_wrapper = tqdm,
    dendrogram_kwargs = dict(
        p=30,
        truncate_mode=None,
        color_threshold=None,
        get_leaves=True,
        count_sort=False,
        distance_sort=False,
        show_leaf_counts=True,
        leaf_font_size=None,
        leaf_rotation=None,
        leaf_label_func=None,
        show_contracted=False,
        link_color_func=None,
        above_threshold_color='C0'
    )
):
    if figsize is None:
        figsize = (figure_width, figure_height)
    if ax is None:
        fig = plt.figure(figsize = figsize, dpi = dpi)
        ax = fig.add_subplot()


    d = dendrogram(
        Z,
        labels = labels,
        no_labels = False,
        orientation='left',
        **dendrogram_kwargs
    )

    # Assign leaf colors
    leaf_colors = ['none'] * len(d['leaves'])
    for xs, ys, c in zip(d['icoord'], d['dcoord'], d['color_list']):
        for xi, yi in zip(xs,ys):
            if (xi % 10 == 5) and (yi == 0):
                leaf_colors[(int(xi)-5) // 10] = c
    label_to_leaf_color = {k:v for k, v in zip(d['ivl'], leaf_colors)}

    if n_jobs > 1:
        data_uris = Parallel(n_jobs=n_jobs)(
            delayed(get_sub_dendrogram_data_uri)(
                Z,
                labels,
                pos,
                dpi = dpi,
                figsize = figsize,
                ypad = 5,
                format = format,
                dendrogram_kwargs = dendrogram_kwargs
            )
            for pos
            in progress_wrapper(list(
                ax.get_yticks()
            ))
        )

    else:
        data_uris = [
            get_sub_dendrogram_data_uri(
                Z,
                labels,
                pos,
                dpi = dpi,
                figsize = figsize,
                ypad = 5,
                format = format,
                dendrogram_kwargs = dendrogram_kwargs
            )
            for pos
            in progress_wrapper(list(
                ax.get_yticks()
            ))
        ]

    label_to_data_uri = {
        k: v
        for k, v
        in zip(d['ivl'], data_uris)
    }
    df = pd.DataFrame(
        dict(label = d['ivl'][::-1])
    )
    df['dendrogram'] = (
        df['label']
        .map(label_to_data_uri)
        .map(lambda x: f'<img src="{x}">')
    )
    df['color'] = df['label'].map(label_to_leaf_color)
    cluster_colors = list(df['color'])
    current_cluster = 0
    clusters = [current_cluster]
    current_cluster_color = cluster_colors[0]
    for cluster_color in cluster_colors[1:]:
        if current_cluster_color == cluster_color:
            clusters.append(current_cluster)
        else:
            current_cluster_color = cluster_color
            current_cluster = current_cluster + 1
            clusters.append(current_cluster)
    df['cluster'] = clusters
    df = df[['dendrogram','label','cluster','color']].copy()
    plt.close(fig)
    return df

def get_hover_dendrogram_df(
    Z,
    labels,
    dendrogram_width = 2,
    row_height = 0.05,
    hover_row_height = 0.5,
    dpi = 100,
    format = 'png',
    n_jobs = multiprocessing.cpu_count(),
    progress_wrapper = tqdm,
    dendrogram_kwargs = dict(
        p=30,
        truncate_mode=None,
        color_threshold=None,
        get_leaves=True,
        count_sort=False,
        distance_sort=False,
        show_leaf_counts=True,
        leaf_font_size=None,
        leaf_rotation=None,
        leaf_label_func=None,
        show_contracted=False,
        link_color_func=None,
        above_threshold_color='C0'
    )
):
    small_dendrogram_df = get_dendrogram_df(
        Z,
        labels,
        figure_width = dendrogram_width,
        figure_height = row_height,
        dpi = dpi,
        format = format,
        n_jobs = n_jobs,
        progress_wrapper = progress_wrapper,
        dendrogram_kwargs = dendrogram_kwargs
    ).rename(columns={'label':'motif_id'})

    hover_small_dendrogram_df = get_dendrogram_df(
        Z,
        labels,
        figure_width = dendrogram_width,
        figure_height = hover_row_height,
        dpi = dpi,
        format = format,
        n_jobs = n_jobs,
        progress_wrapper = progress_wrapper,
        dendrogram_kwargs = dendrogram_kwargs
    ).rename(columns={'label':'motif_id'})

    small_dendrogram_df = small_dendrogram_df.merge(
        hover_small_dendrogram_df
        .rename(columns={
            'dendrogram':'hover_dendrogram'
        })
    )

    small_dendrogram_df['dendrogram'] = (
        small_dendrogram_df['dendrogram']
        .map(lambda x: x.replace(
            '<img ',
            '<img class="dendrogram" '
        ))
    )
    small_dendrogram_df['hover_dendrogram'] = (
        small_dendrogram_df['hover_dendrogram']
        .map(lambda x: x.replace(
            '<img ',
            '<img class="hover-dendrogram" '
        ))
    )
    small_dendrogram_df['dendrogram'] = (
        small_dendrogram_df['dendrogram'] +
        small_dendrogram_df['hover_dendrogram']
    )
    small_dendrogram_df = (
        small_dendrogram_df
        .drop(columns=['hover_dendrogram']))

    return small_dendrogram_df

def filepaths_df_to_clustering_results(
    out_filepath,
    center = None,
    motif_orientation = '+',
    filepaths_df = None,
    cluster_method = 'average',
    cluster_metric = 'correlation',
    mepp_plot_format = 'png',
    mt_method = 'fdr_tsbky',
    mt_alpha = 0.01,
    thorough_mt = True,
    dendrogram_width = 2,
    logo_width = 2,
    heatmap_width = None,
    hover_row_height = 0.5,
    row_height = 0.05,
    axis_height = 0.5,
    table_image_dpi = 100,
    table_image_format = 'png',
    n_jobs = multiprocessing.cpu_count(),
    progress_wrapper = tqdm
):
    # Get filepaths of all motif profiles
    if filepaths_df is None:
        filepaths_filepath = normpath(f'{out_filepath}/filepaths.tsv')
        filepaths_df = pd.read_csv(filepaths_filepath, sep = '\t')

    # Map motif IDs to profiles
    profile_dicts = filepaths_df_to_profile_dicts(
        filepaths_df,
        motif_orientation,
        mepp_plot_format = mepp_plot_format
    )

    motif_id_to_profile = profile_dicts['motif_id_to_profile']
    motif_id_to_mepp_plot = profile_dicts['motif_id_to_mepp_plot']
    motif_id_to_profile_df = profile_dicts['motif_id_to_profile_df']
    motif_id_to_motif_matrix = profile_dicts['motif_id_to_motif_matrix']

    re_prefixed_filepath_dict = re_prefix_filepath_dict(
        motif_id_to_mepp_plot,
        f'{normpath(out_filepath)}/',
        ''
    )

    # Get linkage matrix
    Z, labels = get_linkage_matrix(
        motif_id_to_profile,
        method = cluster_method,
        metric = cluster_metric
    )

    # Get clustering and dendrogram dataframe
    dendrogram_df = get_dendrogram_df(
        Z,
        labels,
        dpi = table_image_dpi,
        format = table_image_format,
        n_jobs = n_jobs,
        progress_wrapper = progress_wrapper
    ).rename(columns={'label':'motif_id'})

    # Get summary profile statistics
    minmax_stats_df = get_minmax_stats_df(
        motif_id_to_profile_df,
        mt_method = mt_method,
        mt_alpha = mt_alpha,
        thorough_mt = True
    )

    # Get motif logos
    logo_df = get_logo_df(
        motif_id_to_motif_matrix,
        dpi = table_image_dpi,
        format = table_image_format,
        n_jobs = n_jobs,
        progress_wrapper = progress_wrapper
    )

    # Get sparklines
    max_r = minmax_stats_df['max_r'].max()
    min_r = minmax_stats_df['min_r'].min()
    top = np.max(np.abs([max_r, min_r]))
    bottom = -top


    scaled_sparkline_df = get_sparkline_df(
        motif_id_to_profile_df,
        ylims = (bottom, top),
        dpi = table_image_dpi,
        format = table_image_format,
        n_jobs = n_jobs,
        progress_wrapper = progress_wrapper
    ).rename(columns={'positional_r_profile_sparkline':'positional_r_profile_scaled_sparkline'})

    sparkline_df = get_sparkline_df(
        motif_id_to_profile_df,
        ylims = None,
        dpi = table_image_dpi,
        format = table_image_format,
        n_jobs = n_jobs,
        progress_wrapper = progress_wrapper
    )

    # Combine dataframes
    results_df = (
        dendrogram_df
        .merge(logo_df)
        .merge(scaled_sparkline_df)
        .merge(sparkline_df)
        .merge(minmax_stats_df)
    ).sort_values(by = ['extreme_r_sig', 'abs_extreme_r'], ascending = [False, False]).reset_index().rename(columns={'index':'clustering_order'})

    # Add links to dataframe
    linked_results_df = link_results_df(
        results_df.copy(),
        re_prefixed_filepath_dict
    )

    # Make interactive HTML table
    results_df_html = get_interactive_table_html(
        linked_results_df,
        title = (
            f'{out_filepath} orientation '
            f'{motif_orientation} '
            f'Motif Enrichment Positional Profiling Results '
            f'- MEPP'
        )
    )

    # Get heatmap
    hover_heatmap_df = get_hover_heatmap_df(
        motif_id_to_profile_df,
        center = center,
        cmap = 'bwr',
        heatmap_width = None,
        row_height = row_height,
        hover_row_height = hover_row_height,
        axis_height = axis_height,
        dpi = table_image_dpi,
        format = table_image_format,
        n_jobs = n_jobs,
        progress_wrapper = progress_wrapper
    )
    hover_heatmap_df = link_results_df(
        hover_heatmap_df.copy(),
        re_prefixed_filepath_dict,
        link_columns = [
            'motif_id',
            list(hover_heatmap_df.columns)[1]
        ]
    )

    # Plot dendrogram
    hover_dendrogram_df = get_hover_dendrogram_df(
        Z,
        labels,
        dendrogram_width = dendrogram_width,
        row_height = row_height,
        hover_row_height = hover_row_height,
        dpi = table_image_dpi,
        format = table_image_format,
        n_jobs = n_jobs,
        progress_wrapper = progress_wrapper
    )
    hover_dendrogram_df = link_results_df(
        hover_dendrogram_df.copy(),
        re_prefixed_filepath_dict,
        link_columns = [
            'motif_id'
        ]
    )

    # Make logos
    hover_logo_df = get_hover_logo_df(
        motif_id_to_motif_matrix,
        logo_width = logo_width,
        row_height = row_height,
        hover_row_height = hover_row_height,
        dpi = table_image_dpi,
        format = table_image_format,
        n_jobs = n_jobs,
        progress_wrapper = progress_wrapper
    )
    hover_logo_df = link_results_df(
        hover_logo_df.copy(),
        re_prefixed_filepath_dict,
        link_columns = [
            'motif_id',
            'logo'
        ]
    )

    # Write clustermap HTML
    clustermap_html = get_clustermap_html(
        join_clustermap_df(
            hover_dendrogram_df,
            hover_logo_df,
            hover_heatmap_df
        ),
        title = (
            f'{out_filepath} orientation '
            f'{motif_orientation} '
            f'Motif Enrichment Positional Profiling Clustermap '
            f'- MEPP'
        )
    )

    # Return results
    results = dict(
        clustermap_html = clustermap_html,
        results_html = results_df_html,
        results_df = results_df,
        linkage_matrix = Z,
        labels = labels
    )
    return results

def cluster_profiles(
    motif_id_to_profile,
    dendrogram_depth,
    method = 'average',
    metric = 'correlation'
):
    # if (dendrogram_level is None) or dendrogram_level > dendrogram_depth:
    #     dendrogram_level = dendrogram_depth
    motif_ids_, profiles_ = zip(*list(motif_id_to_profile.items()))

    motif_ids = list(motif_ids_)
    profiles = np.nan_to_num(np.array(profiles_))

    linkage_matrix = linkage(
        profiles,
        method = method,
        metric = metric
    )
    linkage_tree_root = to_tree(linkage_matrix)
    ids_cluster_ordered = linkage_tree_root.pre_order(lambda x: x.id)

    # Order motif ids and profiles by clustering
    motif_ids_cluster_ordered = [
        motif_ids[idx]
        for idx
        in ids_cluster_ordered
    ]

    profiles_cluster_ordered = np.nan_to_num(np.array([
        motif_id_to_profile[motif_id]
        for motif_id
        in motif_ids_cluster_ordered
    ]))

    motif_id_to_cluster_order = {
        motif_id: cluster_order
        for cluster_order, motif_id
        in enumerate(motif_ids_cluster_ordered)
    }

    # Convert heatmap data to table
    clustered_profiles_df = pd.DataFrame(
        profiles_cluster_ordered,
        columns = (
            np.arange(profiles_cluster_ordered.shape[1]) -
            profiles_cluster_ordered.shape[1]//2
        ),
        index = motif_ids_cluster_ordered
    )

    position_columns = list(clustered_profiles_df.columns)

    clustered_profiles_df = (
        clustered_profiles_df
        .reset_index()
        .rename(columns={'index':'motif_id'})
        .copy()
    )

    # Get partial dendrogram data
    # dendrogram_depth = dendrogram_level
    n_clusters = list(np.arange(dendrogram_depth)+1)
    cutree = cut_tree(linkage_matrix, n_clusters)
    motif_id_cluster_assignments = {
        motif_ids[idx]: list(cluster_assignments+1)
        for idx, cluster_assignments
        in enumerate(cutree)
    }

    dendrogram_motif_ids_df = pd.DataFrame(
        [
            motif_id_cluster_assignments[motif_id] +[motif_id]
            for motif_id
            in motif_ids_cluster_ordered
        ],
        columns = (
            [
                f'clustering_step_{step}'
                for step
                in n_clusters
            ] +
            ['motif_id']
        )
    )

    # Combine dendrogram and heatmap data together
    dendrogram_clustered_profiles_df = pd.concat(
        [
            dendrogram_motif_ids_df[
                list(dendrogram_motif_ids_df.columns)[:-1]
            ],
            clustered_profiles_df
        ],
        axis = 1
    )

    clustered_profiles_df = dendrogram_clustered_profiles_df

    return clustered_profiles_df, linkage_matrix, linkage_tree_root
