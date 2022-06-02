# Download and specify bigwig
wget -nc -O GSM4291126_WT_Nanog.bam.bw "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4291126&format=file&file=GSM4291126%5FWT%5FNanog%2Ebam%2Ebw"
BIGWIG_FILEPATH=GSM4291126_WT_Nanog.bam.bw

# Download and specify genome
GENOME_URL="http://hgdownload.cse.ucsc.edu/goldenpath/mm10/bigZips/latest/mm10.fa.masked.gz"
GENOME=mm10
GENOME_FILEPATH=mm10.masked.fa

# Download genome
wget -nc -O "$GENOME_FILEPATH".gz "$GENOME_URL"
zcat "$GENOME_FILEPATH".gz > "$GENOME_FILEPATH"

# Index genome
samtools faidx "$GENOME_FILEPATH"

# Download motifs file
wget -O homer.motifs.txt https://raw.githubusercontent.com/npdeloss/mepp/main/data/homer.motifs.txt

# Specify analysis parameters
MOTIF="nanog"
# Copied from HOMER motifs
MOTIF_FILEPATH=nanog.motif
SLOP=100
BIGWIG=$(basename $BIGWIG_FILEPATH)
MIN_SCORE=0

MOTIF_DB_FILEPATH=homer.motifs.txt

# Specify where files will go
OUTPUT_BASENAME="$GENOME".scan_motif_"$MOTIF".sequence_only_slop_"$SLOP".scored_by_"$BIGWIG".min_score_"$MIN_SCORE".cluster_deduplicated

BED_FILEPATH="$OUTPUT_BASENAME".bed
MEPP_FILEPATH="$OUTPUT_BASENAME".logscored.nanog.periodicity_only.mepp

# Scan for nanog motifs, then score them using the bigwig, output the results to a scored BED FILE
bigWigToWig $BIGWIG_FILEPATH >(wig2bed -x) \
| bedmap --echo --delim '\t' \
--wmean <(scanMotifGenomeWide.pl $MOTIF_FILEPATH $GENOME_FILEPATH -bed -5p ) - \
| awk '$7!="NAN"' \
| awk '{FS=OFS="\t";$5=$7;print $1,$2,$3,$4,$5,$6}' \
|awk -v s=$MIN_SCORE '$5>s' \
| bedtools sort -i - \
| bedtools cluster -s -i - \
| sort -k7,7 -k5,5nr | awk '!a[$7]++' \
| bedtools sort -i - |cut -f1-6 \
> "$BED_FILEPATH"
wc -l "$BED_FILEPATH"

# Convert scores to log-scores, 
# expand interval around motif scans by 100, 
# convert to scored fasta file, 
# then analyze using MEPP
awk '{FS=OFS="\t";$5=log($5+1)/log(2);print $0}' "$BED_FILEPATH" \
| bedtools sort \
|bedtools slop -b $SLOP -g "$GENOME_FILEPATH".fai \
|python -m mepp.get_scored_fasta -fi "$GENOME_FILEPATH" -bed - \
|python -m mepp.cli \
--fa - \
--motifs "$MOTIF_DB_FILEPATH" \
--out "$MEPP_FILEPATH" \
--perms 200 \
--batch 100 \
--dgt 50 \
--jobs 20 \
--gjobs 10 \
--nogpu \
--dpi 100 \
--orientations + \
&> "$MEPP_FILEPATH".log
echo done
