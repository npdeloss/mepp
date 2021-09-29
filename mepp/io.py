import os
import shutil
import pickle as p

import pandas as pd
import numpy as np

from tqdm import tqdm

import tensorflow as tf

import Bio.motifs.jaspar as jaspar
from Bio import SeqIO

from .onehot_dna import alphabet

def motif_matrix_file_to_dicts(
    motifs_file,
    alphabet = alphabet
):
    motifs_bs = jaspar.read(
        motifs_file,
        format = 'jaspar'
    )

    motif_matrix_dict = {
        f'{motif.matrix_id} {motif.name}':
        np.array([
            list(motif.pwm[nuc])
            for nuc
            in alphabet
        ])
        for motif
        in motifs_bs
    }

    motif_consensus_dict = {
        f'{motif.matrix_id} {motif.name}':
        str(motif.consensus)
        for motif
        in motifs_bs
    }

    return motif_matrix_dict, motif_consensus_dict

def motif_matrix_filepath_to_dicts(
    motifs_filepath,
    **kwargs
):
    with open(motifs_filepath) as motifs_file:
        return motif_matrix_file_to_dicts(
            motifs_file,
            **kwargs
        )

def scored_fasta_file_to_dicts(
    fasta_file,
    description_delim = ' '
):
    fasta_records = list(SeqIO.parse(fasta_file, 'fasta'))

    sequence_dict = {
        rec.id:
        str(rec.seq)
        for rec
        in fasta_records
    }

    description_dict = {
        rec.id: tuple(rec.description.split(description_delim))
        for rec
        in fasta_records
        if len(rec.description.split(description_delim)) > 1
    }

    score_dict = {
        key: float(val[1])
        for key, val
        in description_dict.items()
    }

    return sequence_dict, score_dict, description_dict

def scored_fasta_filepath_to_dicts(
    fasta_filepath,
    **kwargs
):
    with open(fasta_filepath) as fasta_file:
        return scored_fasta_file_to_dicts(
            fasta_file,
            **kwargs
        )

def save_dataset(
    dataset,
    path,
    compression = None,
    element_spec_suffix = '.element_spec.p',
    delete_existing = False
):
    normpath = os.path.normpath(path)
    if delete_existing:
        if os.path.exists(normpath):
            shutil.rmtree(normpath)

    tf.data.experimental.save(
        dataset,
        normpath,
        compression
    )

    with open(os.path.normpath(f'{path}{element_spec_suffix}'), 'wb') as f:
        element_spec = dataset.element_spec
        p.dump(element_spec, f)

def load_dataset(
    path,
    compression = None,
    element_spec_suffix = '.element_spec.p'
):
    with open(os.path.normpath(f'{path}{element_spec_suffix}'), 'rb') as f:
        element_spec = p.load(f)

    dataset = tf.data.experimental.load(
        os.path.normpath(path),
        element_spec = element_spec,
        compression = compression
    )

    return dataset
