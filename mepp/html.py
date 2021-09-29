import html

import json

import base64
from io import (
    BytesIO,
    StringIO
)

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import multiprocessing

from tqdm.auto import tqdm

from joblib import Parallel, delayed

from .utils import (
    filepaths_df_to_profile_dicts,
    get_minmax_stats_df
)

from .plot import (
    plot_motif_matrix,
    plot_sparkline,
    plot_heatrow
)

def fig_to_bitmap_data_uri(
    fig,
    format = 'png',
    dpi = 100,
    savefig_kwargs = dict(
        transparent=True,
        bbox_inches='tight'
    )
):
    img = BytesIO()
    fig.savefig(
        img,
        format = format,
        dpi = dpi,
        **{
            k: v
            for k, v
            in savefig_kwargs.items()
            if k not in set([
                'dpi',
                'format'
            ])
        }
    )
    img.seek(0)
    base64_str = base64.b64encode(img.read()).decode("UTF-8")
    data_uri_format = format
    if format == 'svg':
        data_uri_format = 'svg+xml'
    return (
        f'data:image/{data_uri_format};'
        f'base64,{base64_str}'
    )
def motif_matrix_to_logo_data_uri(
    motif_matrix,
    ax = None,
    figure_width = 5,
    figure_height = 1,
    figsize = None,
    dpi = 100,
    format = 'png'
):
    if figsize is None:
        figsize = (figure_width, figure_height)
    if ax is None:
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

    motif_logo = plot_motif_matrix(motif_matrix, ax = ax)
    fig = motif_logo.fig
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

def get_logo_df(
    motif_id_to_motif_matrix,
    figure_width = 5,
    figure_height = 1,
    figsize = None,
    dpi = 100,
    format = 'png',
    n_jobs = multiprocessing.cpu_count(),
    progress_wrapper = tqdm
):
    df = pd.DataFrame(
        dict(
            motif_id = list(
                motif_id_to_motif_matrix.keys()
            )
        )
    )

    if n_jobs > 1:
        data_uris = Parallel(n_jobs=n_jobs)(
            delayed(motif_matrix_to_logo_data_uri)(
                motif_matrix,
                figure_width = figure_width,
                figure_height = figure_height,
                figsize = figsize,
                dpi = dpi,
                format = format
            )
            for motif_matrix
            in progress_wrapper(list(
                df['motif_id']
                .map(motif_id_to_motif_matrix)
            ))
        )

    else:
        data_uris = [
            motif_matrix_to_logo_data_uri(
                motif_matrix,
                figure_width = figure_width,
                figure_height = figure_height,
                figsize = figsize,
                dpi = dpi,
                format = format
            )
            for motif_matrix
            in progress_wrapper(list(
                df['motif_id']
                .map(motif_id_to_motif_matrix)
            ))
        ]
    motif_id_to_data_uri = {
        k: v
        for k, v
        in zip(list(df['motif_id']), data_uris)
    }
    df['logo'] = (
        df['motif_id']
        .map(motif_id_to_data_uri)
        .map(lambda x: f'<img src="{x}">')
    )
    return df

def get_hover_logo_df(
    motif_id_to_motif_matrix,
    logo_width = 2,
    row_height = 0.05,
    hover_row_height = 0.5,
    dpi = 100,
    format = 'png',
    n_jobs = multiprocessing.cpu_count(),
    progress_wrapper = tqdm
):
    small_logo_df = get_logo_df(
        motif_id_to_motif_matrix,
        figure_width = logo_width,
        figure_height = row_height,
        dpi = dpi,
        format = format,
        n_jobs = n_jobs,
        progress_wrapper = progress_wrapper
    )

    hover_small_logo_df = get_logo_df(
        motif_id_to_motif_matrix,
        figure_width = logo_width,
        figure_height = hover_row_height,
        dpi = dpi,
        format = format,
        n_jobs = n_jobs,
        progress_wrapper = progress_wrapper
    )

    small_logo_df = small_logo_df.merge(
        hover_small_logo_df.rename(columns={
            'logo':'hover_logo'
        })
    )
    small_logo_df['logo'] = (
        '<img class="logo" title="' +
        small_logo_df['motif_id'] +
        '" ' +
        (
            small_logo_df['logo']
            .map(lambda x: x.replace(
                '<img ', ''
            ))
        )
    )
    small_logo_df['hover_logo'] = (
        '<img class="hover-logo" title="' +
        small_logo_df['motif_id'] +
        '" ' +
        (
            small_logo_df['hover_logo']
            .map(lambda x: x.replace(
                '<img ', ''
            ))
        )
    )
    small_logo_df['logo'] = (
        small_logo_df['logo'] +
        small_logo_df['hover_logo']
    )
    small_logo_df = (
        small_logo_df
        .drop(columns=['hover_logo'])
    )

    return small_logo_df

def positional_r_df_to_sparkline_data_uri(
    positional_r_df,
    xlims = None,
    ylims = None,
    figure_width = 5,
    figure_height = 1,
    figsize = None,
    dpi = 100,
    format = 'png',
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

    sparkline = plot_sparkline(
        positional_r_df,
        xlims = xlims,
        ylims = ylims,
        ax = ax,
        figsize = figsize,
        **kwargs
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

def get_sparkline_df(
    motif_id_to_profile_df,
    ylims = None,
    figure_width = 5,
    figure_height = 1,
    figsize = None,
    dpi = 100,
    format = 'png',
    n_jobs = multiprocessing.cpu_count(),
    progress_wrapper = tqdm
):
    df = pd.DataFrame(
        dict(
            motif_id = list(
                motif_id_to_profile_df.keys()
            )
        )
    )

    if n_jobs > 1:
        data_uris = Parallel(n_jobs=n_jobs)(
            delayed(positional_r_df_to_sparkline_data_uri)(
                positional_r_df,
                ylims = ylims,
                figure_width = figure_width,
                figure_height = figure_height,
                dpi = dpi,
                format = format
            )
            for positional_r_df
            in progress_wrapper(list(
                df['motif_id']
                .map(motif_id_to_profile_df)
            ))
        )

    else:
        data_uris = [
            positional_r_df_to_sparkline_data_uri(
                positional_r_df,
                ylims = ylims,
                figure_width = figure_width,
                figure_height = figure_height,
                dpi = dpi,
                format = format
            )
            for positional_r_df
            in progress_wrapper(list(
                df['motif_id']
                .map(motif_id_to_profile_df)
            ))
        ]
    motif_id_to_data_uri = {
        k: v
        for k, v
        in zip(list(df['motif_id']), data_uris)
    }
    df['positional_r_profile_sparkline'] = (
        df['motif_id']
        .map(motif_id_to_data_uri)
        .map(lambda x: f'<img src="{x}">')
    )
    return df

def link_results_df(
    results_df,
    motif_id_to_mepp_plot,
    motif_id_column = 'motif_id',
    link_columns = [
        'motif_id',
        'positional_r_profile_scaled_sparkline',
        'positional_r_profile_sparkline'
    ],
    escaped_columns = ['motif_id']
):
    link_tags = (
        '<a target="_blank" href="' +
        (
            results_df[motif_id_column]
            .map(motif_id_to_mepp_plot)
        ) +
        '">'
    )
    for link_column in link_columns:
        if link_column in escaped_columns:
            results_df[link_column] = (
                link_tags +
                results_df[link_column].map(lambda x: html.escape(x)).astype(str) +
                '</a>'
            )
        else:
            results_df[link_column] = (
                link_tags +
                results_df[link_column] +
                '</a>'
            )
    return results_df

def style_results_df(
    results_df,
    minmax_vals = None,
    bottom_top_vals = None,
    image_cols = [
        'dendrogram',
        'logo',
        'positional_r_profile_scaled_sparkline',
        'positional_r_profile_sparkline'
    ],
    signed_cols = ['extreme_r'],
    unsigned_cols = ['abs_extreme_r'],
    min_cols = ['min_r'],
    max_cols = ['max_r'],
    sig_cols = [
        'integral_r_sig',
        'extreme_r_sig',
        'max_r_sig',
        'min_r_sig',
    ],
    up_color = '#e6bbad',
    down_color = '#add8e6',
    unsigned_color = '#e6d8ad',
    sig_color = '#ade6bb'
):
    results_df_style = results_df.style
    if minmax_vals == None:
        min_val = results_df[min_cols].values.min()
        max_val = results_df[max_cols].values.max()
    else:
        min_val, max_val = minmax_vals
    if bottom_top_vals == None:
        top = np.max(np.abs([max_val, min_val]))
        bottom = -top
    else:
        bottom, top = bottom_top_vals

    results_df_style = (
        results_df_style
        .bar(
            subset = signed_cols,
            color = [down_color, up_color],
            align =  'zero',
            vmin = bottom,
            vmax = top
        )
        .bar(
            subset = unsigned_cols,
            color = unsigned_color,
            align =  'left',
            vmin = 0,
            vmax = np.max(top, 0)
        )
        .bar(
            subset = max_cols,
            color = up_color,
            align =  'zero',
            vmin = 0,
            vmax = np.max(max_val, 0)
        )
        .bar(
            subset = min_cols,
            color = down_color,
            align =  'zero',
            vmin = np.min(min_val, 0),
            vmax = 0
        )
        .apply(
            lambda x:
                np.where(
                    x,
                    f'background-color: {sig_color}',
                    None
                ),
            axis = 1,
            subset = sig_cols
        )
        .applymap(
            lambda x: (
                'background-repeat: repeat-x; '
                'background-size: 100% 50%; '
                'background-position-y: 100%;'
            ),
            subset = (
                signed_cols +
                unsigned_cols +
                min_cols +
                max_cols
            )
        )
        .apply(
            lambda x: ['padding: 0px']*len(x),
            axis = 1,
            subset = image_cols
        )
    )
    return results_df_style

def get_interactive_table_html(
    results_df,
    results_df_style = None,
    title = (
        f'Motif Enrichment Positional Profiling Results '
        f'- MEPP'
    ),
    style_results_df_kwargs = {},
    search_panes_cols = [
        'cluster',
        'integral_r_sig',
        'extreme_r_sig',
        'max_r_sig',
        'min_r_sig'
    ],
    hidden_cols = [
        'clustering_order',
        'positional_r_profile_sparkline',
        'dendrogram',
        'color',
        'integral_r',
        'integral_r_lower',
        'integral_r_upper',
        'integral_r_pval',
        'integral_r_padj',
        'integral_r_sig'
    ],
    header_include_html = '\n'.join([
    '<link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.0.1/css/bootstrap.min.css"/>'
    '<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/v/bs5/jq-3.3.1/jszip-2.5.0/dt-1.10.25/b-1.7.1/b-colvis-1.7.1/b-html5-1.7.1/b-print-1.7.1/fh-3.1.9/sp-1.3.0/sl-1.3.3/datatables.min.css"/>'
    '<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/5.0.1/js/bootstrap.bundle.min.js"></script>'
    '<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.36/pdfmake.min.js"></script>'
    '<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/pdfmake/0.1.36/vfs_fonts.js"></script>'
    '<script type="text/javascript" src="https://cdn.datatables.net/v/bs5/jq-3.3.1/jszip-2.5.0/dt-1.10.25/b-1.7.1/b-colvis-1.7.1/b-html5-1.7.1/b-print-1.7.1/fh-3.1.9/sp-1.3.0/sl-1.3.3/datatables.min.js"></script>'
    ]),
    table_classes = [
        'table',
        'table-striped',
        'table-hover',
        'table-bordered',
        'table-sm',
        'table-responsive',
        'w-100',
        'mw-100'
    ]
):
    if results_df_style is None:
        results_df_style = style_results_df(results_df, **style_results_df_kwargs)

    search_panes_cols_idxs = [
        idx+1
        for idx, col in
        enumerate(list(results_df.columns))
        if col in search_panes_cols
    ]
    # print(search_panes_cols_idxs)
    search_panes_not_cols_idxs = [
        idx+1
        for idx, col in
        enumerate(list(results_df.columns))
        if col not in search_panes_cols
    ]
    # print(search_panes_not_cols_idxs)
    hidden_cols_idxs = [
        idx+1
        for idx, col in
        enumerate(list(results_df.columns))
        if col in hidden_cols
    ]
    # print(hidden_cols_idxs)
    data_table_config = dict(
        fixedHeader = True,
        lengthMenu = [
            [10, 25, 50, 100, -1],
            [10, 25, 50, 100, "All"]
        ],
        dom = '\n'.join([
            "<'row'<'col-sm-12 col-md-5'B><'col-sm-12 col-md-3'l><'col-sm-12 col-md-4'f>>",
            "<'row'<'col-sm-12 col-md-5'i>>",
            "<'row'<'col-sm-12'rt>>",
            "<'row'<'col-sm-12 col-md-7'p>>"
        ]),
        buttons = [
            'copy',
            'excel',
            'csv',
            'colvis',
            'searchPanes'
        ],
        searchPanes = dict(
            layout = f'columns-{len(search_panes_cols)}'
        ),
        columnDefs = [
            dict(
                searchPanes=dict(
                    show = True
                ),
                targets = search_panes_cols_idxs
            ),
            dict(
                searchPanes=dict(
                    show = False
                ),
                targets = search_panes_not_cols_idxs
            ),
            dict(
                targets = hidden_cols_idxs,
                visible = False
            ),
        ]
    )

    data_table_config_json = json.dumps(data_table_config)
    # print(data_table_config_json)

    table_classes_str = ' '.join(table_classes)
    includes_html = '\n'.join([
    f'''{header_include_html}''',
    f'''<script type="text/javascript">''',
    f'''$(document).ready(function() {{''',
    f'''    $('table').addClass("{table_classes_str}");''',
    f'''    $('table').DataTable({data_table_config_json});''',
    f'''}})'''
    f'''</script>'''
    ]);

    head_html = f'<meta charset="UTF-8"><title>{title}</title>{includes_html}'
    html = (
        f'''<!DOCTYPE html><head>{head_html}</head>'''
        f'''<body>{results_df_style.render()}</body></html>'''
    )
    return html

def positional_r_df_to_heatrow_data_uri(
    positional_r_df,
    figure_width = 5,
    figure_height = 1,
    figsize = None,
    dpi = 100,
    format = 'png',
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

    heatrow = plot_heatrow(
        positional_r_df,
        ax = ax,
        figsize = figsize,
        **kwargs
    )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
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

def get_heatmap_df(
    motif_id_to_profile_df,
    ylims = None,
    figure_width = 5,
    figure_height = 1,
    figsize = None,
    dpi = 100,
    format = 'png',
    n_jobs = multiprocessing.cpu_count(),
    progress_wrapper = tqdm,
    imshow_args = dict(
        cmap = 'bwr'
    )
):
    df = pd.DataFrame(
        dict(
            motif_id = list(
                motif_id_to_profile_df.keys()
            )
        )
    )

    if n_jobs > 1:
        data_uris = Parallel(n_jobs=n_jobs)(
            delayed(positional_r_df_to_heatrow_data_uri)(
                positional_r_df,
                figure_width = figure_width,
                figure_height = figure_height,
                dpi = dpi,
                format = format,
                imshow_args = imshow_args
            )
            for positional_r_df
            in progress_wrapper(list(
                df['motif_id']
                .map(motif_id_to_profile_df)
            ))
        )

    else:
        data_uris = [
            positional_r_df_to_heatrow_data_uri(
                positional_r_df,
                figure_width = figure_width,
                figure_height = figure_height,
                dpi = dpi,
                format = format,
                imshow_args = imshow_args
            )
            for positional_r_df
            in progress_wrapper(list(
                df['motif_id']
                .map(motif_id_to_profile_df)
            ))
        ]
    motif_id_to_data_uri = {
        k: v
        for k, v
        in zip(list(df['motif_id']), data_uris)
    }
    df['heatrow'] = (
        df['motif_id']
        .map(motif_id_to_data_uri)
        .map(lambda x: f'<img src="{x}">')
    )
    return df

def get_heatmap_axis_data_uri(
    sequence_length,
    center = None,
    figure_width = 5,
    figure_height = 1,
    figsize = None,
    dpi = 100,
    format = 'png'
):
    if figsize is None:
        figsize = (figure_width, figure_height)
    fig = plt.figure(figsize = figsize, dpi = dpi)
    fig.subplots_adjust(
        left = 0,
        right = 1,
        bottom = 0,
        top = 0.001,
        wspace = 0,
        hspace = 0
    )
    ax = fig.add_subplot()

    for k,v in ax.spines.items():
        v.set_visible(False)

    ax.set_yticks([])
    ax.get_yaxis().set_visible(False)
    ax.xaxis.tick_top()
    if center is None:
        center = sequence_length//2
    if center < 0:
        center = 0
    if center > sequence_length:
        center = sequence_length - 1
    ax.set_xlim(
        left = int(-center),
        right = int(sequence_length - center)
    )

    # ax.set_xticklabels(
    #     [int(tick) for tick in ax.get_xticks()],
    #     rotation = 90
    # )
    plt.xticks(rotation = 90)
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

def get_hover_heatmap_df(
    motif_id_to_profile_df,
    center = None,
    cmap = 'bwr',
    vmin = None,
    vmax = None,
    heatmap_width = None,
    row_height = 0.05,
    hover_row_height = 0.5,
    axis_height = 0.5,
    dpi = 100,
    format = 'png',
    n_jobs = multiprocessing.cpu_count(),
    progress_wrapper = tqdm
):
    sequence_length = motif_id_to_profile_df[list(
        motif_id_to_profile_df.keys()
    )[0]].shape[0]
    if heatmap_width is None:
        heatmap_width = sequence_length / dpi

    if (vmin is None) or (vmax is None):
        vmin_ = np.min([
            df['positional_r'].min()
            for df
            in motif_id_to_profile_df.values()
        ])
        vmax_ = np.max([
            df['positional_r'].max()
            for df
            in motif_id_to_profile_df.values()
        ])

        top = np.max(np.abs([vmin_, vmax_]))
        bottom = -top

    if vmin is None:
        vmin = bottom
    if vmax is None:
        vmax = top

    heatrow_df = get_heatmap_df(
        motif_id_to_profile_df,
        figure_width = heatmap_width,
        figure_height = row_height,
        imshow_args = dict(
            cmap = cmap,
            vmin = vmin,
            vmax = vmax,
            aspect = 'auto'
        ),
        dpi = dpi,
        format = format,
        n_jobs = n_jobs,
        progress_wrapper = progress_wrapper
    )

    hover_heatrow_df = get_heatmap_df(
        motif_id_to_profile_df,
        figure_width = heatmap_width,
        figure_height = hover_row_height,
        imshow_args = dict(
            cmap = 'bwr',
            vmin = vmin,
            vmax = vmax,
            aspect = 'auto'
        ),
        dpi = dpi,
        format = format,
        n_jobs = n_jobs,
        progress_wrapper = progress_wrapper
    )

    heatrow_df = heatrow_df.merge(
        hover_heatrow_df
        .rename(columns={
            'heatrow':'hover_heatrow'
        })
    )

    heatrow_df['heatrow'] = (
        '<img class="heatrow" title="' +
        heatrow_df['motif_id'] +
        '" ' +
        (
            heatrow_df['heatrow']
            .map(
                lambda x: x.replace('<img ', '')
            )
        )
    )
    heatrow_df['hover_heatrow'] = (
        '<img class="hover-heatrow" title="' +
        heatrow_df['motif_id'] +
        '" ' +
        (
            heatrow_df['hover_heatrow']
            .map(
                lambda x: x.replace('<img ', '')
            )
        )
    )
    heatrow_df['heatrow'] = (
        heatrow_df['heatrow'] +
        heatrow_df['hover_heatrow']
    )
    heatrow_df = (
        heatrow_df
        .drop(columns=['hover_heatrow'])
        .rename(columns = {'heatrow':'heatmap'})
    )

    data_uri = get_heatmap_axis_data_uri(
        sequence_length = sequence_length,
        figure_width = heatmap_width,
        figure_height = axis_height,
        dpi = dpi,
        format = format,
        center = center
    )
    img_html = f'<img src="{data_uri}">'
    heatrow_df = (
        heatrow_df
        .rename(columns = {
            'heatmap':f'heatmap<br>{img_html}'
        })
    )
    return heatrow_df

def join_clustermap_df(
    dendrogram_df,
    logo_df,
    heatmap_df
):
    clustermap_df = (
        dendrogram_df
        .merge(logo_df)
        .merge(heatmap_df)
    ).drop(columns=['color'])[[
        'dendrogram',
        'cluster',
        'logo',
        list(heatmap_df.columns)[1],
        'motif_id',
    ]]
    return clustermap_df

def style_clustermap_df(
    clustermap_df,
    font_family = 'Dejavu Sans, Arial, Helvetica, sans-serif',
    link_font_size = '4px',
    hover_link_font_size = '10pt'
):
    clustermap_df_style = (
        clustermap_df
        .style
        .set_table_styles(
            [
                dict(
                    selector = '',
                    props = [
                        ('border', '0'),
                        ('border-spacing', '0'),
                        ('border-collapse', 'collapse'),
                        ('font-family', font_family),
                    ]
                ),
                dict(
                    selector = 'thead',
                    props = [
                        ('position', 'sticky'),
                        ('top', '0px'),
                        ('background-color', 'white')
                    ]
                ),
                dict(
                    selector = 'tr, th, td, a, span, img, div',
                    props = [
                        ('border', '0'),
                        ('border-collapse', 'collapse'),
                        ('padding', '0'),
                        ('margin', '0'),
                        ('text-decoration', 'none'),
                        ('white-space', 'nowrap'),
                        ('text-align', 'center')
                    ]
                ),
                dict(
                    selector = 'tr td a',
                    props = [
                        ('font-size', link_font_size),
                        ('display', 'block'),
                        ('text-align', 'left')
                    ]
                ),
                dict(
                    selector = 'tr th:last-of-type',
                    props = [
                        ('text-align', 'left'),
                    ]
                ),
                dict(
                    selector = 'tr img.heatrow, tr img.dendrogram, tr img.logo',
                    props = [
                        ('display', 'block'),
                        ('padding', '0'),
                        ('margin', '0'),
                    ]
                ),
                dict(
                    selector = 'img.heatrow, img.hover-heatrow',
                    props = [
                        ('image-rendering', 'crisp-edges')
                    ]
                ),
                dict(
                    selector = 'tr:hover img.heatrow, tr:hover img.dendrogram, tr:hover img.logo',
                    props = [
                        ('display', 'none')
                    ]
                ),
                dict(
                    selector = 'tr img.hover-heatrow, tr img.hover-dendrogram, tr img.hover-logo',
                    props = [
                        ('display', 'none'),
                        ('padding', '0'),
                        ('margin', '0'),
                    ]
                ),
                dict(
                    selector = 'tr:hover img.hover-heatrow, tr:hover img.hover-dendrogram, tr:hover img.hover-logo',
                    props = [
                        ('display', 'block')
                    ]
                ),
                dict(
                    selector = 'tr:hover td a',
                    props = [
                        ('font-size', hover_link_font_size)
                    ]
                ),

            ]
        )
        .hide_columns(subset=[
            # 'motif_id',
            'cluster',
        ])
        .hide_index()
    )
    return clustermap_df_style

def get_clustermap_html(
    clustermap_df,
    clustermap_df_style = None,
    title = (
        f'Motif Enrichment Positional Profiling Clustermap '
        f'- MEPP'
    ),
    style_clustermap_df_kwargs = dict(
        font_family = 'Dejavu Sans, Arial, Helvetica, sans-serif',
        link_font_size = '4px',
        hover_link_font_size = '10pt'
    )
):
    if clustermap_df_style is None:
        clustermap_df_style = style_clustermap_df(
            clustermap_df, **style_clustermap_df_kwargs
        )
    head_html = (
        f'<meta charset="UTF-8"><title>{title}</title>'
    )
    clustermap_df_html = (
        f'''<!DOCTYPE html><head>{head_html}</head>'''
        f'''<body>{clustermap_df_style.render()}</body></html>'''
    )
    return clustermap_df_html

def style_clustered_profiles_df(
    clustered_profiles_df,
    motif_id_to_mepp_plot = None,
    dendrogram_level = None,
    colormap_name = 'bwr',
    block_width = 1,
    hover_block_width = 3,
    block_height = 1,
    hover_block_height = 10,
    font_size = '1pt',
    position_font_size = '4pt',
    hover_font_size = '12pt'
):
    df = clustered_profiles_df.copy()
    df_cols = list(df.columns)
    motif_id_col_idx = df_cols.index('motif_id')
    cluster_step_cols = df_cols[0:motif_id_col_idx]
    position_cols = df_cols[motif_id_col_idx+1:]

    # HTML format heatmap dataframe
    df[cluster_step_cols] = (
        df[cluster_step_cols]
        .applymap(lambda x: f'<span class="cluster-step">{x}</span>')
    )
    # f'<span class="cluster-step-dendrogram"><span class="cluster-step">{x}</span></span>'
    if motif_id_to_mepp_plot is None:
        motif_id_to_mepp_plot = {
            k:'#'
            for k
            in list(df['motif_id'])
        }
    motif_id_to_link_html = {
        k: (
            f'<span class="motif-id">'
            f'<a href="{v}">{k}</a>'
            f'</span>'
        )
        for k,v
        in motif_id_to_mepp_plot.items()
    }
    df['motif_id'] = df['motif_id'].map(motif_id_to_link_html)
    if dendrogram_level == None:
        dendrogram_level = max([
            int(col.split('_')[-1])
            for col in
            cluster_step_cols
        ])
    cluster_step_cols_rename = {
        k: f'<span class="cluster-step-header">{i+1}</span>'
        for i,k
        in enumerate(cluster_step_cols)
        if int(k.split('_')[-1]) == dendrogram_level
    }

    motif_id_col_rename = {
        'motif_id': '<span class="motif-id-header">Motif</a>'
    }
    position_cols_rename = {
        k: f'<span class="position-header">{k}</span>'
        for k
        in position_cols
    }

    df = (
        df
        .rename(columns = cluster_step_cols_rename)
        .rename(columns = motif_id_col_rename)
        .rename(columns = position_cols_rename)
        [
            list(cluster_step_cols_rename.values())+
            list(motif_id_col_rename.values())+
            list(position_cols_rename.values())
        ]
        .set_index(
            list(cluster_step_cols_rename.values()) +
            list(motif_id_col_rename.values())
        )
    ).copy()

    # Get heatmap color range
    position_cols = list(df.columns[:])
    # motif_cols = list(df.columns[:1])
    extreme_val = np.max(np.abs([
        np.max(df[position_cols]),
        np.min(df[position_cols])
    ]))

    vmin = -extreme_val
    vmax = extreme_val
    # Style heatmap dataframe
    df_style = df.style
    # df_classes = df.copy()
    # df_classes[motif_cols] = 'motif'
    # df_classes[position_cols] = 'position'
    # df_style = df_style.set_td_classes(df_classes)
    df_style = df_style.format(
        '<span class="heatmap-number">{:.4f}</span>',
        subset = position_cols
    )

    cm = mpl.cm.get_cmap(colormap_name)
    df_style = df_style.background_gradient(
        cmap = cm,
        vmin = vmin,
        vmax = vmax,
        subset = position_cols
    )
    df_style = df_style.set_table_styles(
        [
            dict(
                props = [
                    ('border-spacing', '0'),
                    ('border', '0'),
                    ('border-collapse', 'collapse')
                ]
            ),
            dict(
                props = [
                    ('font-family', (
                        "Consolas, "
                        "'Lucida Console', "
                        "'DejaVu Sans Mono', "
                        "monospace"
                    )),
                    ('font-size', hover_font_size),
                ]
            ),
            dict(
                selector = '.cluster-step-header',
                props = [
                    ('display', 'none')
                ]
            ),
            dict(
                selector = 'tbody th',
                props = [
                    ('background', '#ccc'),
                    ('min-width', '30px'),
                    ('max-width', '30px'),
                    ('overflow', 'hidden'),
                    (
                        'clip-path',
                        (
                            'polygon('
                            '0% 25%, '
                            '50% 0%, '
                            '100% 0%, '
                            '100% 100%, '
                            '50% 100%, '
                            '0% 75%, '
                            '0% 25%'
                            ')'
                        )
                    ),
                    ('position', 'sticky'),
                    ('left', '0px'),
                    ('z-index', '3'),
                    ('white-space', 'nowrap')
                ]
            ),
            dict(
                selector = 'tbody td',
                props = [
                    ('z-index', '1')
                ]
            ),
            dict(
                selector = 'thead tr:first-of-type th',
                props = [
                    ('background', '#fff'),
                    ('position', 'sticky'),
                    ('top', '0px'),
                    ('z-index', '2')
                ]
            ),
            dict(
                selector = 'tbody th:last-of-type',
                props = [
                    ('max-width', 'none'),
                    ('background', '#fff'),
                    ('clip-path', 'none'),
                    ('position', 'sticky'),
                    ('left', '30px'),
                    ('z-index', '3')
                ]
            ),
            dict(
                selector = '.cluster-step',
                props = [
                    ('display', 'none')
                ]
            ),
            dict(
                selector = '.heatmap-number',
                props = [
                    ('display', 'none')
                ]
            ),
            dict(
                selector = '.position-header',
                props = [
                    ('display', 'inline')
                ]
            ),
            dict(
                selector = 'thead tr:first-of-type th',
                props = [
                    ('width', f'{block_width}px'),
                    ('max-width', f'{block_width}px'),
                    ('font-size', position_font_size),
                    ('overflow', 'hidden'),
                    ('writing-mode', 'vertical-rl'),
                    ('text-orientation', 'mixed')
                ]
            ),
            dict(
                selector = 'thead tr:first-of-type th',
                props = [
                    ('width', f'{hover_block_width}px'),
                    ('max-width', f'{hover_block_width}px'),
                    # ('font-size', hover_font_size),
                    ('font-size', position_font_size),
                ]
            ),
            dict(
                selector = 'tbody td',
                props = [
                    ('width', f'{block_width}px'),
                    ('height', f'{block_height}px'),
                    ('font-size', font_size),
                ]
            ),
            dict(
                selector = 'tbody td',
                props = [
                    ('width', f'{hover_block_width}px'),
                    ('height', f'{hover_block_height}px'),
                    ('font-size', hover_font_size),
                ]
            ),
            # dict(
            #     selector = 'tbody td:hover .heatmap-number',
            #     props = [
            #         ('display', 'inline')
            #     ]
            # ),
            dict(
                selector = 'tbody th:last-of-type',
                props = [
                    ('height', f'{block_height}px'),
                    ('font-size', font_size),
                    ('text-align', 'right')
                ]
            ),
            dict(
                selector = 'tbody th:last-of-type',
                props = [
                    ('height', f'{hover_block_height}px'),
                    ('font-size', hover_font_size),
                    ('text-align', 'right')
                ]
            )
        ]
    )

    styled_df_style = df_style
    styled_df = df

    return styled_df_style, styled_df
