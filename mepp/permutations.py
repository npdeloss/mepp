import pandas as pd
import numpy as np
import tensorflow as tf

def add_permuted_scores_to_dataset(
    dataset,
    batch_size = None,
    num_permutations = 1000
):
    if num_permutations <= 0:
        return dataset
    # Get batch size
    if batch_size is None:
        batch_size = [
            data for data in dataset.take(1)
        ][0][0].shape[0]
    # Get scores
    scores = tf.concat(
        [
            score_batch
            for score_batch
            in dataset.map(lambda *data: data[1])
        ],
        axis = 0
    )

    expanded_scores = tf.expand_dims(scores, -1)

    # Permute scores
    # Index 0 is true score
    permuted_scores = tf.concat(
        [expanded_scores]+
        [
            tf.random.shuffle(expanded_scores)
            for i
            in range(num_permutations)
        ],
        axis = -1
    )

    #
    permuted_scores_dataset = (
        tf.data.Dataset.zip((
            dataset.unbatch(),
            tf.data.Dataset.from_tensor_slices(permuted_scores)
        ))
        .map(lambda *data: (
            (data[0][0], data[1]) +
            (
                data[0][2:]
                if len(data[0])>2
                else ()
            )
        ))
        .batch(batch_size)
    )

    return permuted_scores_dataset

# def process_permuted_scores_dataset_with_model(
#     permuted_scores_dataset,
#     motif_conv_model
# ):
#
#     has_gc = len(permuted_scores_dataset.element_spec) > 2
#
#     def process_data(*data):
#
#         onehot_sequences, scores = data[:2]
#         if has_gc:
#             gc_ratios = data[2]
#
#         motif_scores = motif_conv_model(
#             tf.cast(
#                 onehot_sequences,
#                 tf.as_dtype('float')
#             ),
#             training = False
#         )
#         expanded_permuted_scores = tf.expand_dims(scores, 1)
#         if has_gc:
#             return motif_scores, expanded_permuted_scores, gc_ratios
#         else:
#             return motif_scores, expanded_permuted_scores
#
#     processed_permuted_scores_dataset = permuted_scores_dataset.map(process_data)
#     return processed_permuted_scores_dataset

# def get_positional_r_df_with_permutation_stats(
#     r,
#     center = None,
#     confidence_interval_pct = 95
# ):
#
#     # Format positional r to dataframe
#     positional_r_df = pd.DataFrame(
#         dict(positional_r = r[:,0])
#     )
#     positional_r_df['position'] = list(
#         range(positional_r_df.shape[0])
#     )
#     if center is None:
#         center = sequence_length//2
#     positional_r_df['position'] -= center
#
#     # Calculate confidence intervals
#     alpha = 1.0 - (confidence_interval_pct/100.0)
#     percentile_lower = 100.0*alpha/2
#     percentile_upper = 100.0 - 100.0*alpha/2
#     q = (percentile_lower, percentile_upper)
#     r_percentiles = np.percentile(
#         r[:,1:], q, axis=-1
#     )
#
#     # Calculate permutation p-values
#     positional_r_pvals = np.mean(
#         (
#             np.abs(r[:,1:]) >=
#             np.abs(r[:,0:1])
#         ).astype(float),
#         axis = -1
#     )
#
#     # Format confidence intervals and p-values to dataframe
#     positional_r_df[
#         'permutation_positional_r_lower'
#     ] = r_percentiles[0]
#     positional_r_df[
#         'permutation_positional_r_upper'
#     ] = r_percentiles[1]
#     positional_r_df[
#         'permutation_positional_r_pval'
#     ] = positional_r_pvals
#
#     return positional_r_df
