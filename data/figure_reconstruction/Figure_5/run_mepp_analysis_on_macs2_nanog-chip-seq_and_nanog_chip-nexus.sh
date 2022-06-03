# Specify analysis parameters
genome_fa=mm10.fa
slop=200
margin=5
pval=0.0001
motifs_filepath=homer.motifs.txt

# Download genome
GENOME_URL="http://hgdownload.cse.ucsc.edu/goldenpath/mm10/bigZips/mm10.fa.gz"
wget -nc -O "$genome_fa".gz "$GENOME_URL"
zcat "$genome_fa".gz > "$genome_fa"

# Index genome
samtools faidx "$genome_fa"

# Download narrowpeak files
wget -O "GSM4072778_mesc_nanog_nexus.idr-optimal-set.narrowPeak.gz" "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4072778&format=file&file=GSM4072778%5Fmesc%5Fnanog%5Fnexus%2Eidr%2Doptimal%2Dset%2EnarrowPeak%2Egz"
wget -O "GSM4087827_mesc_Nanog_chipseq.idr-optimal-set.narrowPeak.gz" "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSM4087827&format=file&file=GSM4087827%5Fmesc%5FNanog%5Fchipseq%2Eidr%2Doptimal%2Dset%2EnarrowPeak%2Egz"

# Download motifs
wget -O homer.motifs.txt https://raw.githubusercontent.com/npdeloss/mepp/main/data/homer.motifs.txt

# Analyze Nanog ChIP-nexus
narrowpeak_gz=GSM4072778_mesc_nanog_nexus.idr-optimal-set.narrowPeak.gz
mepp_filepath="$narrowpeak_gz".slop_"$slop".margin_"$margin".pval_"$pval".mepp

zcat "$narrowpeak_gz" \
|awk '{OFS="\t";summit=int($2+$10);print $1,summit,summit,$1":"summit,$7,"+"}' \
|bedtools slop -i - -g "$genome_fa".fai -b $slop \
|python -m mepp.get_scored_fasta -fi "$genome_fa" -bed - \
|$(which time) --verbose \
python -m mepp.cli \
--fa - \
--motifs "$motifs_filepath" \
--out "$mepp_filepath" \
--margin "$margin" \
--pval "$pval" \
--perms 100 \
--batch 1000 \
--dgt 50 \
--jobs 15 \
--gjobs 15 \
--nogpu \
--dpi 100 \
--orientations +/- \
&> "$mepp_filepath".log
echo Done for "$mepp_filepath"

# Analyze Nanog ChIP-seq
narrowpeak_gz=GSM4087827_mesc_Nanog_chipseq.idr-optimal-set.narrowPeak.gz
mepp_filepath="$narrowpeak_gz".slop_"$slop".margin_"$margin".pval_"$pval".mepp
zcat "$narrowpeak_gz" \
|awk '{OFS="\t";summit=int($2+$10);print $1,summit,summit,$1":"summit,$7,"+"}' \
|bedtools slop -i - -g "$genome_fa".fai -b $slop \
|python -m mepp.get_scored_fasta -fi "$genome_fa" -bed - \
|$(which time) --verbose \
python -m mepp.cli \
--fa - \
--motifs "$motifs_filepath" \
--out "$mepp_filepath" \
--margin "$margin" \
--pval "$pval" \
--perms 100 \
--batch 1000 \
--dgt 50 \
--jobs 15 \
--gjobs 15 \
--nogpu \
--dpi 100 \
--orientations +/- \
&> "$mepp_filepath".log
echo Done for "$mepp_filepath"
