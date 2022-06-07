import numpy as np
import pandas as pd

from skimage.measure import block_reduce
from skimage.transform import resize

import logomaker

import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

from .core import (
    smooth_motif_density_by_rank_df
)

from .onehot_dna import alphabet

def get_pixel_dimensions(fig, ax):
    ax_pixel_bbox = (
        ax
        .get_window_extent()
        .transformed(
            fig.dpi_scale_trans.inverted()
        )
    )
    pixel_width,pixel_height = ax_pixel_bbox.width*fig.dpi, ax_pixel_bbox.height*fig.dpi
    return pixel_width, pixel_height

def get_aspect_tensor(shape, pixel_width, pixel_height):
    aspect_tensor = np.array([
        shape[0]/pixel_height,
        shape[1]/pixel_width
    ])
    aspect_tensor = np.clip(aspect_tensor, 1.0, np.max(aspect_tensor))
    aspect_tensor = tuple(np.round(aspect_tensor).astype(int))
    aspect_tensor = tuple([max(1, x) for x in aspect_tensor[:]])
    return aspect_tensor

def compress_matrix_to_pixels(
    matrix,
    pixel_width,
    pixel_height,
    func = np.mean,
    cval = 0
):
    aspect_tensor = get_aspect_tensor(matrix.shape, pixel_width, pixel_height)
    compressed_matrix = block_reduce(
        matrix,
        tuple(aspect_tensor),
        func
    )
    return compressed_matrix

def clip_motif_score_matrix(motif_score_matrix, sigma = 1.0):
    masked_motif_score_matrix = np.ma.masked_equal(motif_score_matrix,0.0)
    motif_score_mean = masked_motif_score_matrix.mean()
    motif_score_std = masked_motif_score_matrix.std()
    clipped_motif_score_matrix = np.clip(
        motif_score_matrix,
        0.0,
        motif_score_mean+motif_score_std*sigma
    )
    return clipped_motif_score_matrix

def plot_matrix_with_compression(
    matrix,
    func = np.mean,
    cval = 0,
    ax = None,
    fig = None,
    resize_kwargs = dict(
        mode='edge'
    ),
    decompress = False,
    **kwargs
):
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    pixel_width, pixel_height = get_pixel_dimensions(fig, ax)

    compressed_matrix = compress_matrix_to_pixels(
        matrix,
        pixel_width,
        pixel_height,
        func = func,
        cval = cval
    )
    if decompress:
        decompressed_matrix = resize(
            compressed_matrix,
            matrix.shape,
            **resize_kwargs
        )
    else:
        decompressed_matrix = compressed_matrix
    return ax.imshow(
        decompressed_matrix,
        **kwargs
    )

def plot_motif_score_matrix(
    motif_score_matrix,
    sigma = 1.0,
    decompress = False,
    cmap = 'gray',
    interpolation = 'bicubic',
    center = None,
    fig = None,
    ax = None
):
    if sigma is not None:
        clipped_motif_score_matrix = (
            clip_motif_score_matrix(
                motif_score_matrix,
                sigma = sigma
            )
        )
    else:
        clipped_motif_score_matrix = motif_score_matrix

    sequence_length = motif_score_matrix.shape[1]
    if center is None:
        center = sequence_length//2
    return plot_matrix_with_compression(
        clipped_motif_score_matrix,
        decompress = decompress,
        fig = fig,
        ax = ax,
        func = np.max,
        cmap = cmap,
        aspect = 'auto',
        interpolation = interpolation,
        origin = 'lower',
        extent = [
            -center,
            -center + sequence_length,
            0,
            clipped_motif_score_matrix.shape[0]
        ]
    )

def smooth_motif_density_by_rank_df_to_pixels(
    motif_density_by_rank_df,
    pixel_height,
    smoothing_factor = 5
):
    pixel_width = 1
    aspect_tensor = get_aspect_tensor(
        (motif_density_by_rank_df.shape[0],1),
        pixel_width,
        pixel_height)

    vertical_margin = aspect_tensor[0] * smoothing_factor
    window = 1 + vertical_margin*2

    smoothed_motif_density_by_rank_df = smooth_motif_density_by_rank_df(
        motif_density_by_rank_df,
        window
    )

    return smoothed_motif_density_by_rank_df

def plot_score_rank_df(
    score_rank_df,
    ax = None,
    **kwargs
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    ax.plot(
        score_rank_df['score'],
        score_rank_df['rank'],
        **kwargs
    )

def plot_positional_r_df(
    positional_r_df,
    ax = None,
    plot_kwargs = {},
    fill_between_kwargs = dict(alpha = 0.25),
    axhline_kwargs = dict(
        linewidth = 1,
        linestyle = '--',
        color = 'k'
    )
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    ax.plot(
        positional_r_df['position'],
        positional_r_df['positional_r'],
        **plot_kwargs
    )

    has_confidence_intervals = (
        ('positional_r_lower' in positional_r_df.columns) and
        ('positional_r_upper' in positional_r_df.columns)
    )

    has_permutation_confidence_intervals = (
        ('permutation_positional_r_lower' in positional_r_df.columns) and
        ('permutation_positional_r_upper' in positional_r_df.columns)
    )

    if has_permutation_confidence_intervals:

        ax.fill_between(
          positional_r_df['position'],
          positional_r_df['permutation_positional_r_lower'],
          positional_r_df['permutation_positional_r_upper'],
          **fill_between_kwargs
        )
    elif has_confidence_intervals:

        ax.fill_between(
          positional_r_df['position'],
          positional_r_df['positional_r_lower'],
          positional_r_df['positional_r_upper'],
          **fill_between_kwargs
        )

    ax.axhline(
        0,
        **axhline_kwargs
    )

def plot_motif_counts_by_position_df(
    motif_counts_by_position_df,
    ax = None,
    **kwargs
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    ax.plot(
        motif_counts_by_position_df['position'],
        motif_counts_by_position_df['count'],
        **kwargs
    )

def plot_motif_density_by_rank_df(
    motif_density_by_rank_df,
    smoothing_factor = 5,
    fig = None,
    ax = None,
    **kwargs
):
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    smoothed_motif_density_by_rank_df = (
        smooth_motif_density_by_rank_df_to_pixels(
            motif_density_by_rank_df,
            get_pixel_dimensions(fig, ax)[1],
            smoothing_factor = smoothing_factor
        )
    )

    ax.plot(
        smoothed_motif_density_by_rank_df['density'],
        smoothed_motif_density_by_rank_df['rank'],
        **kwargs
    )

def motif_matrix_to_df(motif_matrix, alphabet = alphabet):
    return (
        pd.DataFrame(
            motif_matrix
        )
        .T
        .rename(
            columns = {
                k:v
                for k,v
                in enumerate(
                    list(
                        alphabet
                    )
                )
            },
            index = {
                i:
                i+1
                for i
                in range(
                    motif_matrix.shape[1]
                )
            }
        )
    )

def plot_motif_matrix(motif_matrix, alphabet = alphabet, **kwargs):

    motif_logo = logomaker.Logo(
        logomaker.transform_matrix(
            motif_matrix_to_df(
                motif_matrix,
                alphabet = alphabet
            ),
            from_type = 'probability',
            to_type = 'information'
        ),
        **kwargs
    )

    # style using Logo methods
    motif_logo.style_spines(visible=False)
    motif_logo.style_spines(spines=['left', 'bottom'], visible=True)
    motif_logo.style_xticks(fmt='%d', anchor=0)

    # style using Axes methods
    motif_logo.ax.set_ylabel('bits', labelpad=-1)
    motif_logo.ax.xaxis.set_ticks_position('none')
    motif_logo.ax.xaxis.set_tick_params(pad=-1)
    motif_logo.ax.set_ylim(0.0, 2.0)

    return motif_logo

def setup_motif_heatmap_axes(
    fig,
    gridspec_kwargs = dict(
        width_ratios = [2,3,1],
        height_ratios = [1,3,1],
        wspace = 0.1,
        hspace = 0.1,
        nrows = 3,
        ncols = 3
    )
):
    gs = GridSpec(
      figure = fig,
      **gridspec_kwargs
    )

    ax_logo = fig.add_subplot(gs[0,0])
    ax_score_rank = fig.add_subplot(gs[1,0])
    ax_rank = fig.add_subplot(gs[1,-1])
    ax_motifs = fig.add_subplot(gs[1,1])
    ax_pos = fig.add_subplot(gs[0,1])
    ax_r = fig.add_subplot(gs[2,1])

    ax_motifs.set_xticklabels([])
    ax_motifs.set_yticklabels([])

    ax_rank.set_yticklabels([])

    ax_score_rank.set_ylabel('Sequence score rank')
    ax_score_rank.set_xlabel('Sequence score')
    ax_score_rank.xaxis.tick_bottom()
    ax_score_rank.yaxis.tick_left()

    ax_pos.set_ylabel('Motif count')
    ax_pos.yaxis.set_label_position('right')
    ax_pos.set_xlabel('Motif position')
    ax_pos.xaxis.tick_top()
    ax_pos.yaxis.tick_right()

    ax_rank.set_xlabel('Motif density')
    ax_rank.xaxis.tick_bottom()
    ax_rank.set_ylabel('Sequence score rank')
    ax_rank.yaxis.set_label_position('right')
    ax_rank.yaxis.tick_right()

    ax_r.set_xlabel('Motif position')
    ax_r.xaxis.tick_bottom()

    ax_r.set_ylabel('Positional \nPearson \ncorrelation \ncoefficient')
    ax_r.yaxis.tick_left()

    ax_pos.get_shared_x_axes().join(ax_pos, ax_motifs)
    ax_pos.get_shared_x_axes().join(ax_pos, ax_r)
    ax_score_rank.get_shared_y_axes().join(ax_score_rank, ax_motifs)
    ax_score_rank.get_shared_y_axes().join(ax_score_rank, ax_rank)

    axes = dict(
        logo = ax_logo,
        count_position = ax_pos,
        score_rank = ax_score_rank,
        motif_heatmap = ax_motifs,
        density_rank = ax_rank,
        correlation_position = ax_r
    )

    return axes, gs

def plot_profile_data(
    motif_score_matrix,
    positions_df,
    ranks_df,
    motif_matrix = None,
    title = None,
    figure_width = 10,
    figure_height = 10,
    figsize = None,
    fig = None,
    axes = None,
    rank_smoothing_factor = 5,
    motif_score_sigma = 1,
    motif_score_cmap = 'gray',
    decompress_matrix = False
):
    if figsize is None:
        figsize = (figure_width, figure_height)
    if fig is None:
        fig = plt.figure(figsize = figsize)
    if axes is None:
        axes, gs = setup_motif_heatmap_axes(fig)

    if motif_matrix is not None:
        plot_motif_matrix(motif_matrix, ax = axes['logo'])

    plot_motif_counts_by_position_df(
        (
            positions_df[['position', 'smoothed_count']]
            .rename(columns={'smoothed_count':'count'})
        ),
        ax = axes['count_position']
    )

    has_permutation_confidence_intervals = (
        ('permutation_positional_r_lower' in positions_df.columns) and
        ('permutation_positional_r_upper' in positions_df.columns)
    )
    fill_between_kwargs = dict(alpha = 0.25)
    if has_permutation_confidence_intervals:
        fill_between_kwargs['color'] = 'tab:gray'
        fill_between_kwargs['alpha'] = 0.50

    plot_positional_r_df(
        positions_df,
        ax = axes['correlation_position'],
        fill_between_kwargs = fill_between_kwargs
    )
    
    plot_score_rank_df(
        ranks_df,
        ax = axes['score_rank']
    )

    plot_motif_density_by_rank_df(
        ranks_df,
        smoothing_factor = rank_smoothing_factor,
        fig = fig,
        ax = axes['density_rank']
    )
    
    center = list(positions_df['position']).index(0)
    
    plot_motif_score_matrix(
        motif_score_matrix,
        sigma = motif_score_sigma,
        cmap = motif_score_cmap,
        fig = fig,
        ax = axes['motif_heatmap'],
        decompress = decompress_matrix,
        center = center
    )

    if title is not None:
        plt.suptitle(title, figure = fig)

    return fig, axes

def plot_sparkline(
    positional_r_df,
    xlims = None,
    ylims = None,
    ax = None,
    r_line_kwargs = dict(color='tab:blue',alpha=1.0),
    r_fill_kwargs = dict(color='tab:blue',alpha=0.25),
    null_fill_kwargs = dict(color='tab:gray',alpha=0.50),
    zero_hline_kwargs = dict(color='black',alpha=0.5,linewidth=1,linestyle='--'),
    zero_vline_kwargs = dict(color='black',alpha=0.01,linewidth=1,linestyle='--'),
    hide_axis = True,
    figure_width = 5,
    figure_height = 1,
    figsize = None
):
    df = positional_r_df.copy()
    if figsize is None:
        figsize = (figure_width, figure_height)
    if ax is None:
        fig = plt.figure(figsize = figsize, dpi = dpi)
        ax = fig.add_subplot()

    positions = df['position']
    rs = df['positional_r']
    zeros = np.zeros(len(positions))

    has_permutation = (
        ('permutation_positional_r_lower' in df.columns) and
        ('permutation_positional_r_upper' in df.columns)
    )

    if has_permutation:
        null_r_lowers = df['permutation_positional_r_lower']
        null_r_uppers = df['permutation_positional_r_upper']

    ax.plot(
        positions,
        rs,
        **r_line_kwargs
    )
    ax.fill_between(
        positions,
        rs,
        zeros,
        **r_fill_kwargs
    )

    ax.axhline(
        0,
        **zero_hline_kwargs
    )

    ax.axvline(
        0,
        **zero_vline_kwargs
    )

    if has_permutation:
        ax.fill_between(
            positions,
            null_r_lowers,
            null_r_uppers,
            **null_fill_kwargs
        )

    if hide_axis:
        for k,v in ax.spines.items():
            v.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    if xlims is not None:
        left, right = xlims
        ax.set_xlim(left, right)
    else:
        left = positions.min()
        right = positions.max()
        ax.set_xlim(left, right)
    if ylims is not None:
        bottom, top = ylims
        ax.set_ylim(bottom, top)
    return ax

def plot_heatrow(
    positional_r_df,
    imshow_args = dict(
        cmap = 'bwr'
    ),
    hide_axis = True,
    figure_width = 5,
    figure_height = 1,
    figsize = None,
    ax = None,
):
    df = positional_r_df.copy()
    if figsize is None:
        figsize = (figure_width, figure_height)
    if ax is None:
        fig = plt.figure(figsize = figsize, dpi = dpi)
        ax = fig.add_subplot()

    positions = df['position']
    rs = df['positional_r']

    ax.imshow(
        [list(rs)],
        **imshow_args
    )

    if hide_axis:
        for k,v in ax.spines.items():
            v.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    return ax
