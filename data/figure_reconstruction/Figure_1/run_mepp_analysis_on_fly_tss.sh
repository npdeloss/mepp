# Specify parameters
MIN_COUNT=3
GENOME_URL="https://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.masked.gz"
GENOME_FILEPATH=dm6.masked.fa
TSS_FILEPATH=dm6.embryonic_only.min_$MIN_COUNT.tss.txt
SLOP=200
BED_FILEPATH=dm6.min_$MIN_COUNT.tss.slop_$SLOP.bed
CLUSTERED_DEDUPLICATED_BED_FILEPATH=dm6.embryonic_only.min_$MIN_COUNT.tss.slop_$SLOP.cluster_deduplicated.bed
MOTIF_DB_FILEPATH="ohler_motifs.txt"
MEPP_FILEPATH=dm6.embryonic_only.min_$MIN_COUNT.tss.slop_$SLOP.cluster_deduplicated.logscored.$(basename $MOTIF_DB_FILEPATH).mepp

# Confirm final MEPP filepath
echo $MEPP_FILEPATH

# Download genome
wget -nc -O "$GENOME_FILEPATH".gz "$GENOME_URL"
zcat "$GENOME_FILEPATH".gz > "$GENOME_FILEPATH"

# Index genome
samtools faidx "$GENOME_FILEPATH"

# Format TSS filepath to BED, expand intervals to +/- $SLOP
cat $TSS_FILEPATH \
|awk '{FS=OFS="\t";if($5==1){$5="-"}else{$5="+"};print $2,$3,$4,$1,$6,$5}' \
|bedtools slop -s -b $SLOP -g $GENOME_FILEPATH.fai \
|bedtools sort \
> $BED_FILEPATH

# Perform cluster deduplication
bedtools cluster -s -i $BED_FILEPATH \
|sort -k7,7 -k5,5nr \
|awk '!a[$7]++' \
|bedtools sort \
|cut -f1-6 \
> $CLUSTERED_DEDUPLICATED_BED_FILEPATH

# Convert to scored FASTA, for recordkeeping
cat $CLUSTERED_DEDUPLICATED_BED_FILEPATH \
|awk '{FS=OFS="\t";$5=log($5+1)/log(2);print $0}' \
|python -m mepp.get_scored_fasta -fi "$GENOME_FILEPATH" -bed - \
> $CLUSTERED_DEDUPLICATED_BED_FILEPATH.scored.fa

# Convert to scored FASTA and run MEPP analysis
cat $CLUSTERED_DEDUPLICATED_BED_FILEPATH \
|awk '{FS=OFS="\t";$5=log($5+1)/log(2);print $0}' \
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
--orientations +,- \
&> "$MEPP_FILEPATH".log
