import sys

import pandas as pd

import click

from Bio import SeqIO

def fasta_file_to_dict(
    fasta_file
):
    fasta_records = list(SeqIO.parse(fasta_file, 'fasta'))

    sequence_dict = {
        rec.id:
        str(rec.seq)
        for rec
        in fasta_records
    }

    return sequence_dict

def fasta_filepath_to_dict(
    fasta_filepath,
    **kwargs
):
    with open(fasta_filepath) as fasta_file:
        return fasta_file_to_dicts(
            fasta_file,
            **kwargs
        )

bed_columns =  [
    'Chromosome',
    'Start',
    'End',
    'Name',
    'Score',
    'Strand'
]
    
def bed_filepath_to_df(bed_filepath, sep = '\t', bed_columns = bed_columns):
    df = pd.read_csv(
        bed_filepath, 
        sep = sep, 
        header = None, 
        names = 
        bed_columns
    )
    return df

def get_safe_substring(s, start, end, pad = ''):
    safe_start = max(start, 0)
    safe_end = min(end, len(s))
    left_padding_length = safe_start - start
    right_padding_length = end - safe_end
    
    left_padding = ''.join([pad]*left_padding_length)
    right_padding = ''.join([pad]*right_padding_length)
    
    substring = s[safe_start:safe_end]
    
    safe_substring = left_padding + substring + right_padding
    
    return safe_substring

revcomp_dict = {
    'A': 'T',
    'C': 'G',
    'G': 'C',
    'T': 'A',
    'N': 'N'
}

revcomp_dict.update({
    k.lower():v.lower()
    for k,v
    in revcomp_dict.copy().items()
})

def reverse_complement(dna, revcomp_dict = revcomp_dict):
    return ''.join([
        revcomp_dict[nuc] 
        for nuc 
        in dna[::-1]
    ])

def get_safe_substring_from_dna(
    dna, start, end, strand,
    revcomp_dict = revcomp_dict,
    pad = 'N'
):
    safe_substring_ = get_safe_substring(
        dna, start, end, pad = pad
    )
    safe_substring = ''.join([
        nuc 
        if nuc in set(revcomp_dict) 
        else 'N' 
        for nuc 
        in safe_substring_
    ])
    if strand is '-':
        return reverse_complement(
            safe_substring,
            revcomp_dict = revcomp_dict
        )
    else:
        return safe_substring

def write_scored_fasta(
    bed_file, 
    fasta_file, 
    out_file, 
    sep = '\t', 
    revcomp_dict = revcomp_dict, 
    pad = 'N',
    print_output = False
):
    col2idx = {colname:idx for idx, colname in enumerate(bed_columns)}
    sequence_dict = fasta_file_to_dict(fasta_file)

    for line in bed_file:
        bed_entry = line.rstrip().split(sep)
        if len(bed_entry) == 2:
            bed_entry.append(bed_entry[-1])
        if len(bed_entry) == 3:
            bed_entry.append(
                f'{bed_entry[0]}:{bed_entry[1]}-{bed_entry[2]}'
            )
        if len(bed_entry) == 4:
            bed_entry.append(0)
        if len(bed_entry) == 5:
            bed_entry.append('+')

        chromosome = str(bed_entry[col2idx['Chromosome']])
        start = int(bed_entry[col2idx['Start']])
        end = int(bed_entry[col2idx['End']])
        name = str(bed_entry[col2idx['Name']])
        score = float(bed_entry[col2idx['Score']])
        strand = str(bed_entry[col2idx['Strand']])
        sequence_header = f'>{name}:{chromosome}:{start}-{end}:{strand} {score}'

        dna = sequence_dict[chromosome]
        sequence = get_safe_substring_from_dna(
            dna,
            start, 
            end, 
            strand,
            revcomp_dict = revcomp_dict,
            pad = pad
        )
        
        scored_fasta_entry = '\n'.join([
            sequence_header,
            sequence
        ])+'\n'
        if print_output:
            print(scored_fasta_entry)
        out_file.write(scored_fasta_entry)

def write_scored_fasta_from_filepaths(
    bed_filepath, 
    fasta_filepath, 
    out_filepath,
    **kwargs
):
    with click.open_file(bed_filepath) as bed_file, \
        click.open_file(fasta_filepath) as fasta_file, \
        click.open_file(out_filepath, 'w') as out_file\
    :
        write_scored_fasta(
            bed_file, 
            fasta_file, 
            out_file, 
            **kwargs
        )

@click.command()
@click.option(
    '-bed',
    'bed_filepath',
    type = str,
    required = True,
    help = (
        'BED file of ranges to extract DNA sequences from -fi. '
        'The fifth column should include the sequence score. '
        'Ranges with a negative strand value will extract reverse-complemented sequence.'
    )
)
@click.option(
    '-fi',
    'fasta_filepath',
    type = str,
    required = True,
    help = (
        'Input FASTA file to extract DNA sequences from.'
    )
)
@click.option(
    '-fo',
    'out_filepath',
    type = str,
    default = '-',
    help = (
        'Output file (opt., default is STDOUT)'
    )
)
def main(
    bed_filepath, 
    fasta_filepath, 
    out_filepath, 
):
    write_scored_fasta_from_filepaths(
        bed_filepath, 
        fasta_filepath, 
        out_filepath, 
        revcomp_dict = revcomp_dict, 
        pad = 'N',
        print_output = False
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
