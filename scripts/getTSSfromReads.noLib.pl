#!/usr/bin/env perl
use warnings;


# Copyright 2009 - 2021 Christopher Benner <cbenner@ucsd.edu>
#
# This file is part of HOMER
#
# HOMER is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HOMER is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#

use POSIX;


my $size = 1;
my $strandOpt = "+";
my $fragLength = 1;
my $prefix = 'out';
my $maxCPUs = 30;
my $genome = 'none';


my $minThreshold = 7;
my $minRawThreshold = -1;
my $pseudo = 2;
my $startIndex = 19;
my $interChrFlag = 1;
my $interNsample = 5000;
my $rlogFlag = '';
my $scoreMode = 'corr';


my $binSize = 1000;
my $maxDistance = 1000000;
my $binSizeLocal = 1;
my $maxDistanceLocal = 2000;

sub printCMD {
	print STDERR "\n\tgetTSSfromReads.pl [options] -d <tagDir1> <tagDir2> ...\n";
	print STDERR "\n\tIf no peak file is given, script will use each single nt position\n";
	print STDERR "\n\tQuantification parameters:\n";
	print STDERR "\t\t-min <#> (minimum reads per 10^7 in at least one experiment, def: $minThreshold)\n";
	print STDERR "\t\t\t-minRaw <#> (minimum raw read count in at least one experiment, def: uses -min)\n";
	print STDERR "\t\t-dinput <inputTagDir1> [inputTagDir2] ... (control directories - if the max from \n";
	print STDERR "\t\t\t\tany of these is higher than the max from the experiments, position is excluded\n";
	print STDERR "\n\n";
	exit;
}


if (@ARGV < 1) {
	printCMD();
}

my $peakFile = '';
my $quantFile = '';
my @tagDirs = ();
my @inputTagDirs = ();
my $tagDirStr = '';
my $inputTagDirStr = '';
my $preMaxN = 0;
my $skipAfterQuant = 0;

for (my $i=0;$i<@ARGV;$i++) {
	if ($ARGV[$i] eq '-d') {
		for (my $j=$i+1;$j<@ARGV;$j++) {
			if ($ARGV[$j] =~ /^-/) {
				last;
			}
			push(@tagDirs, $ARGV[$j]);
			$tagDirStr .= " \"$ARGV[$j]\"";
			print STDERR "\tWill analyze TagDir: $ARGV[$j]\n";
			$i=$j;
		}
	} elsif ($ARGV[$i] eq '-dinput' || $ARGV[$i] eq '-i') {
		for (my $j=$i+1;$j<@ARGV;$j++) {
			if ($ARGV[$j] =~ /^-/) {
				last;
			}
			push(@inputTagDirs, $ARGV[$j]);
			$inputTagDirStr .= " \"$ARGV[$j]\"";
			print STDERR "\tWill analyze inputTagDir: $ARGV[$j]\n";
			$i=$j;
		}
	} elsif ($ARGV[$i] eq '-skipAfterQuant') {
		$skipAfterQuant = 1;
	} elsif ($ARGV[$i] eq '-skipAfterQuant') {
		$skipAfterQuant = 1;
	} elsif ($ARGV[$i] eq '-o') {
		$prefix = $ARGV[++$i];
		print STDERR "\tOutput prefix set to \"$prefix\"\n";
	} elsif ($ARGV[$i] eq '-p') {
		$peakFile = $ARGV[++$i];
		print STDERR "\tUsing peak file \"$peakFile\"\n";
	} elsif ($ARGV[$i] eq '-q') {
		$quantFile = $ARGV[++$i];
		print STDERR "\tUsing quant file \"$quantFile\"\n";
	} elsif ($ARGV[$i] eq '-size') {
		$size = $ARGV[++$i];
		print STDERR "\tSetting peak size to \"$size\"\n";
	} elsif ($ARGV[$i] eq '-min') {
		$minThreshold = $ARGV[++$i];
		print STDERR "\tSetting minimum normalized read threshold to \"$minThreshold\"\n";
	} elsif ($ARGV[$i] eq '-minRaw') {
		$minRawThreshold = $ARGV[++$i];
		$minThreshold = $minRawThreshold;
		print STDERR "\tSetting minimum raw read threshold to \"$minRawThreshold\"\n";
	} elsif ($ARGV[$i] eq '-preMaxN') {
		$preMaxN = $ARGV[++$i];
		print STDERR "\tWill limit analysis to the top $preMaxN peaks before normalization\n";
	} elsif ($ARGV[$i] eq '-cpu') {
		$maxCPUs = $ARGV[++$i];
		print STDERR "\tSetting maxCPUs to \"$maxCPUs\"\n";
	} elsif ($ARGV[$i] eq '-pseudo') {
		$pseudo = $ARGV[++$i];
		print STDERR "\tSetting pseudo count to \"$pseudo\"\n";
	} elsif ($ARGV[$i] eq '-strand') {
		$strandOpt = $ARGV[++$i];
		print STDERR "\tSetting strand to \"$strandOpt\"\n";
	} elsif ($ARGV[$i] eq '-fragLength') {
		$fragLength = $ARGV[++$i];
		print STDERR "\tSetting fragLength to \"$fragLength\"\n";
	} elsif ($ARGV[$i] eq '-rlog') {
		$rlogFlag = " -rlog ";
		print STDERR "\tWill analyze rlog values\n";
	} elsif ($ARGV[$i] eq '-bin') {
		$binSize = $ARGV[++$i];
	} elsif ($ARGV[$i] eq '-binLocal') {
		$binSizeLocal = $ARGV[++$i];
	} elsif ($ARGV[$i] eq '-maxDist') {
		$maxDistance = $ARGV[++$i];
	} elsif ($ARGV[$i] eq '-maxDistLocal') {
		$maxDistanceLocal = $ARGV[++$i];
	} else {
		print STDERR "!!! Error: \"$ARGV[$i]\" not recognized\n";
		printCMD();
	}
}

print STDERR "\tUsing threshold of $minThreshold\n";

my $rand = rand();
my $tmpPeakFile = $rand . ".peaks.tmp";
my $tmpQuantFile = $rand . ".quant.tmp";
my $tmpFile = $rand . ".tmp";
my $tmpFile2 = $rand . ".2tmp";

if ($quantFile ne '') {

} elsif ($peakFile ne '') {
	`adjustPeakFile.pl "$peakFile" -size $size > "$tmpPeakFile"`;
} else {

	my $totalReads = 0;
	my $totalPositions = 0;
	my %tss = ();
	for (my $i=0;$i<@tagDirs;$i++) {
		my $tagDir = $tagDirs[$i];

		my ($dirTotalTags,$dirTotalPos,$dirFragEst,$dirPeakEst) = readTagInfo($tagDir);

		my $scaleFactor = ((1e7)/$dirTotalTags);
	
		print STDERR "\tProcessing $tagDir ($dirTotalTags, scaleFactor=$scaleFactor)\n";
		$tagFiles = `ls $tagDir/*.tags.tsv`;
		chomp $tagFiles;
		my @files = split /\n/,$tagFiles;
		#print STDERR "|@files|\n";
		for (my $j=0;$j<@files;$j++) {
			my $file = $files[$j];
			print STDERR "\t\t$file\n";
			open IN, $file;
			my %data = ();
			while (<IN>) {
				chomp;
				my @line = split /\t/;
				my $chr = $line[1];
				my $pos = $line[2];
				my $strand = $line[3];
				my $v = $line[4];
				if (!exists($data{$chr})) {
					my %a = ();
					$data{$chr} = \%a;
				}
				if (!exists($data{$chr}->{$pos})) {
					my %a = ();
					$data{$chr}->{$pos} = \%a;
				}
				if (!exists($data{$chr}->{$pos}->{$strand})) {
					$data{$chr}->{$pos}->{$strand} = $v;
				} else {
					$data{$chr}->{$pos}->{$strand} += $v;
				}
				$totalReads += $v;
			}
			close IN;

			foreach(keys %data) {
				my $chr = $_;
				foreach(keys %{$data{$chr}}) {
					my $pos = $_;
					foreach (keys %{$data{$chr}->{$pos}}) {
						my $strand = $_;
						my $v = $data{$chr}->{$pos}->{$strand};
						if ($minRawThreshold < 0) {
							$v *= $scaleFactor;
						}
						next if ($v < $minThreshold);
				
						if (!exists($tss{$chr})) {
							my %a = ();
							$tss{$chr} = \%a;
						}
						if (!exists($tss{$chr}->{$pos})) {
							my %a = ();
							$tss{$chr}->{$pos} = \%a;
						}
						if (!exists($tss{$chr}->{$pos}->{$strand})) {
							$tss{$chr}->{$pos}->{$strand} = $v;
							$totalPositions++;
						} else {
							#store max value for filtering later
							if ($v > $tss{$chr}->{$pos}->{$strand}) {
							$tss{$chr}->{$pos}->{$strand} = $v;
							}
						}
						$totalReads += $v;
					}
				}
			}
		}
		print STDERR "\tFinished $tagDir\n\n";
	}
	my $inputFiltered = 0;
	for (my $i=0;$i<@inputTagDirs;$i++) {
		my $tagDir = $inputTagDirs[$i];

		my ($dirTotalTags,$dirTotalPos,$dirFragEst,$dirPeakEst) = readTagInfo($tagDir);
		my $scaleFactor = ((1e7)/$dirTotalTags);

		print STDERR "\t(input) Processing $tagDir ($dirTotalTags, scaleFactor=$scaleFactor)\n";
		$tagFiles = `ls $tagDir/*.tags.tsv`;
		chomp $tagFiles;
		my @files = split /\n/,$tagFiles;
		#print STDERR "|@files|\n";
		for (my $j=0;$j<@files;$j++) {
			my $file = $files[$j];
			print STDERR "\t\t$file\n";
			open IN, $file;
			my %data = ();
			while (<IN>) {
				chomp;
				my @line = split /\t/;
				my $chr = $line[1];
				my $pos = $line[2];
				my $strand = $line[3];
				my $v = $line[4];
				if (!exists($data{$chr})) {
					my %a = ();
					$data{$chr} = \%a;
				}
				if (!exists($data{$chr}->{$pos})) {
					my %a = ();
					$data{$chr}->{$pos} = \%a;
				}
				if (!exists($data{$chr}->{$pos}->{$strand})) {
					$data{$chr}->{$pos}->{$strand} = $v;
				} else {
					$data{$chr}->{$pos}->{$strand} += $v;
				}
				$totalReads += $v;
			}
			close IN;

			foreach(keys %data) {
				my $chr = $_;
				if (!exists($tss{$chr})) {
					next;
				}
				foreach(keys %{$data{$chr}}) {
					my $pos = $_;
					if (!exists($tss{$chr}->{$pos})) {
						next;
					}
					foreach (keys %{$data{$chr}->{$pos}}) {
						my $strand = $_;
						if (!exists($tss{$chr}->{$pos}->{$strand})) {
							next;
						}
						my $v = $data{$chr}->{$pos}->{$strand};
						if ($minRawThreshold < 0) {
							$v *= $scaleFactor;
						}
						#filter if input is higher than max reads
						if ($v > $tss{$chr}->{$pos}->{$strand}) {
							if ($tss{$chr}->{$pos}->{$strand} > 0) {
								$inputFiltered++;
							}
							$tss{$chr}->{$pos}->{$strand} = -1;
						}
					}
				}
			}
		}
		print STDERR "\n\t(input) Finished $tagDir\n";
	}

	print STDERR "\t$inputFiltered of $totalPositions filtered by input overall\n";


	open OUT, ">$prefix.positions.txt";

	my $z = 1;
	foreach(keys %tss) {
		my $chr = $_;
		foreach(keys %{$tss{$chr}}) {
			my $pos = $_;
			foreach(keys %{$tss{$chr}->{$pos}}) {
				my $strand = $_;
				my $v = $tss{$chr}->{$pos}->{$strand};
				next if ($v < $minThreshold);
				print "TSS-$z\t$chr\t$pos\t$pos\t$strand\t$v\n";
				$z++;
			}
		}
	}
	$z--;
	close OUT;
	print STDERR "\tTotalPositions: $totalPositions ($z used after input filtering)\n";
	print STDERR "\tTotalReads: $totalReads\n";
exit;
	if ($preMaxN > 0) {
		print STDERR "\n\tFiltering peaks for the top $preMaxN\n";
		`getTopPeaks.pl "$prefix.positions.txt" $preMaxN > "$tmpFile"`;
		`mv "$tmpFile" "$prefix.positions.txt"`;
	}
	`cp $prefix.positions.txt $tmpPeakFile`;

}
exit;
if ($quantFile ne '') {
	`cp "$quantFile" "$tmpQuantFile"`;
} else {
	`annotatePeaks.pl "$tmpPeakFile" $genome -fragLength $fragLength -strand $strandOpt -d $tagDirStr -cpu $maxCPUs $rlogFlag > "$tmpQuantFile"`;
	`cp "$tmpQuantFile" "$prefix.quant.txt"`;
	`rm "$tmpPeakFile"`;
	if ($skipAfterQuant) {
		`rm "$tmpQuantFile"`;
		exit;
	}
}



my $totalTSS = 0;
my $totalInput = 0;
my $totalKept = 0;
my $numExps = 0;

print STDERR "\tReading in peak quantification data (file=$tmpQuantFile)\n";
%tss = ();
open IN, $tmpQuantFile;
my $c = 0;
while (<IN>) {
	$c++;
	next if ($c < 2);
	chomp;
	s/\r//g;
	my @line = split /\t/;
	my $chr = $line[1];
	my $pos = $line[2];
	my $strand = $line[4];
	$strand = '+' if ($strand eq '0');
	$strand = '-' if ($strand eq '1');
	my $v = $line[5];
	$totalTSS++;

	my @v = ();
	my $maxV=-1e10;
	my $avgV = 0;
	my $inputV = 0;
	my $avgN = 0;
	for (my $i=$startIndex;$i<@line;$i++) {
		my $v = $line[$i];
		if ($lastDirInputFlag && $i==scalar(@line)-1) {
			$inputV = $v;
			last;
		}
		$avgV += $v;
		$avgN++;
		$maxV = $v if ($v > $maxV);
		if ($rlogFlag eq '') {
			$v = log($v+$pseudo)/log(2.0);
		}
		push(@v, $v);
	}
	$avgV /= $avgN if ($avgN > 0);
	$numExps = $avgN if ($avgN > $numExps);
	if ($lastDirInputFlag && $avgV < $inputV) {
		$totalInput++;
		next;
	}
	next if ($maxV < $minThreshold);


	$totalKept++;
	if (!exists($tss{$chr})) {
		my %a = ();
		$tss{$chr} = \%a;
	}
	if (!exists($tss{$chr}->{$pos})) {
		my %a = ();
		$tss{$chr}->{$pos} = \%a;
	}
	my $t = {v=>\@v};
	$tss{$chr}->{$pos}->{$strand} = $t;
}

print STDERR "\tTotalTSS: $totalTSS\n";
print STDERR "\tTotalInput: $totalInput\n";
print STDERR "\tValidTSS: $totalKept\n";
print STDERR "\tNum Exps/TagDirectories: $numExps\n";

`rm "$tmpQuantFile"`;

my @same = ();
my @sameN = ();
my @conv = ();
my @convN = ();
my @div = ();
my @divN = ();
for (my $i=0;$i<$maxDistance/$binSize;$i++) {
	push(@same,0);
	push(@sameN,0);
	push(@conv,0);
	push(@convN,0);
	push(@div,0);
	push(@divN,0);
}


my @samelocal = ();
my @sameNlocal = ();
my @convlocal = ();
my @convNlocal = ();
my @divlocal = ();
my @divNlocal = ();
for (my $i=0;$i<$maxDistanceLocal/$binSizeLocal;$i++) {
	push(@samelocal,0);
	push(@sameNlocal,0);
	push(@convlocal,0);
	push(@convNlocal,0);
	push(@divlocal,0);
	push(@divNlocal,0);
}


my %chrR = ();
my %chrN = ();
my $interR = 0;
my $interN = 0;



my $mmax = $maxDistance;
$mmax = $maxDistanceLocal if ($mmax < $maxDistanceLocal);

my @chrs = sort {$a cmp $b} keys %tss;
print STDERR "\n\tCalculating cross-correlations\n";
foreach(keys %tss) {
	my $chr=$_;
	my @pos = sort {$a <=> $b} keys %{$tss{$chr}};

	my $totalPos = scalar(@pos);
	print STDERR "\t\t$chr ($totalPos)\n";

	for (my $i=0;$i<@pos;$i++) {
		print STDERR "\t\t\t$i\n" if ($i % 100000 == 0);
		$p = $pos[$i];

		if (exists($tss{$chr}->{$p}->{'+'})) {
			$d = '+';
			my $v1 = $tss{$chr}->{$p}->{$d}->{'v'};

			#first check if current location makes transcription on other strand
			if (exists($tss{$chr}->{$p}->{'-'})) {
				my $v2 = $tss{$chr}->{$p}->{'-'}->{'v'};
				my ($r,$pv) = (0,0);
				if ($numExps > 2) {
					($r,$pv) = correlation($v1,$v2);
				} elsif ($numExps == 2) {
					$r = abs(($v2->[1]-$v2->[0]) - ($v1->[1]-$v1->[0]));
				} else {
					$r = abs($v2->[0] - $v1->[0]);
				}
				$div[0]+=$r;
				$divN[0]++;
				$divlocal[0]+=$r;
				$divNlocal[0]++;
			}

			for (my $j=$i+1;$j<@pos;$j++) {
				my $pp = $pos[$j];
				my $diffbp = $pp-$p;
				last if ($diffbp >= $mmax);

				my $diff = floor($diffbp/$binSize);
				my $difflocal = floor($diffbp/$binSizeLocal);

				if (exists($tss{$chr}->{$pp}->{'+'})) {
					my $v2 = $tss{$chr}->{$pp}->{'+'}->{'v'};
					my ($r,$pv) = (0,0);
					if ($numExps > 2) {
						($r,$pv) = correlation($v1,$v2);
					} elsif ($numExps == 2) {
						$r = abs(($v2->[1]-$v2->[0]) - ($v1->[1]-$v1->[0]));
					} else {
						$r = abs($v2->[0] - $v1->[0]);
					}

					if ($diffbp < $maxDistance) {
						$same[$diff]+=$r;
						$sameN[$diff]++;
					}
					if ($diffbp < $maxDistanceLocal) {
						$samelocal[$difflocal]+=$r;
						$sameNlocal[$difflocal]++;
					}
				}
				if (exists($tss{$chr}->{$pp}->{'-'})) {
					my $v2 = $tss{$chr}->{$pp}->{'-'}->{'v'};
					my ($r,$pv) = (0,0);
					if ($numExps > 2) {
						($r,$pv) = correlation($v1,$v2);
					} elsif ($numExps == 2) {
						$r = abs(($v2->[1]-$v2->[0]) - ($v1->[1]-$v1->[0]));
					} else {
						$r = abs($v2->[0] - $v1->[0]);
					}
					if ($diffbp < $maxDistance) {
						$conv[$diff]+=$r;
						$convN[$diff]++;
					}
					if ($diffbp < $maxDistanceLocal) {
						$convlocal[$difflocal]+=$r;
						$convNlocal[$difflocal]++;
					}
				}
			}
		}
		if (exists($tss{$chr}->{$p}->{'-'})) {
			$d = '-';
			my $v1 = $tss{$chr}->{$p}->{$d}->{'v'};
			for (my $j=$i+1;$j<@pos;$j++) {
				my $pp = $pos[$j];
				my $diffbp = $pp-$p;
				last if ($diffbp >= $mmax);

				my $diff = floor($diffbp/$binSize);
				my $difflocal = floor($diffbp/$binSizeLocal);
				if (exists($tss{$chr}->{$pp}->{'+'})) {
					my $v2 = $tss{$chr}->{$pp}->{'+'}->{'v'};
					my ($r,$pv) = (0,0);
					if ($numExps > 2) {
						($r,$pv) = correlation($v1,$v2);
					} elsif ($numExps == 2) {
						$r = abs(($v2->[1]-$v2->[0]) - ($v1->[1]-$v1->[0]));
					} else {
						$r = abs($v2->[0] - $v1->[0]);
					}
					if ($diffbp < $maxDistance) {
						$div[$diff]+=$r;
						$divN[$diff]++;
					}
					if ($diffbp < $maxDistanceLocal) {
						$divlocal[$difflocal]+=$r;
						$divNlocal[$difflocal]++;
					}
				}
				if (exists($tss{$chr}->{$pp}->{'-'})) {
					my $v2 = $tss{$chr}->{$pp}->{'-'}->{'v'};
					my ($r,$pv) = (0,0);
					if ($numExps > 2) {
						($r,$pv) = correlation($v1,$v2);
					} elsif ($numExps == 2) {
						$r = abs(($v2->[1]-$v2->[0]) - ($v1->[1]-$v1->[0]));
					} else {
						$r = abs($v2->[0] - $v1->[0]);
					}
					if ($diffbp < $maxDistance) {
						$same[$diff]+=$r;
						$sameN[$diff]++;
					}
					if ($diffbp < $maxDistanceLocal) {
						$samelocal[$difflocal]+=$r;
						$sameNlocal[$difflocal]++;
					}
				}
			}
		}
	}
	if ($interChrFlag) {
		my $chrR = 0;
		my $chrN = 0;
		foreach(@chrs) {
			my $chr2 = $_;
			my @pos2 = sort {$a <=> $b} keys %{$tss{$chr2}};
			my $totalPos2 = scalar(@pos2);
			for (my $k=0;$k<$interNsample;$k++) {
				my $r1 = floor(rand()*$totalPos);
				my $r2 = floor(rand()*$totalPos2);
				my $d1 = "+";
				my $d2 = "+";
				$d1 = "-" if (rand() > 0.5);
				$d2 = "-" if (rand() > 0.5);
				my $v1 = 0;
				if (exists($tss{$chr}->{$pos[$r1]}->{$d1})) {
					$v1 = $tss{$chr}->{$pos[$r1]}->{$d1}->{'v'};
				} else {
					$k--;
					next;
				}
				my $v2 = 0;
				if (exists($tss{$chr2}->{$pos2[$r2]}->{$d2})) {
					$v2 = $tss{$chr2}->{$pos2[$r2]}->{$d2}->{'v'};
				} else {
					$k--;
					next;
				}
				my ($r,$pv) = (0,0);
				if ($numExps > 2) {
					($r,$pv) = correlation($v1,$v2);
				} elsif ($numExps == 2) {
					$r = abs(($v2->[1]-$v2->[0]) - ($v1->[1]-$v1->[0]));
				} else {
					$r = abs($v2->[0] - $v1->[0]);
				}
				if ($chr2 eq $chr) {
					$chrR += $r;
					$chrN++;
				} else {
					$interR += $r;
					$interN++;
				}
			}
		}
		$chrR /= $chrN if ($chrN > 0);
		$chrR{$chr} = $chrR;
		$chrN{$chr} = $chrN;
	}
}

open OUT, ">$prefix.distal.txt";
print OUT "binIndex\tDistance\tAll\tAllN\tSameStrand\tSameN\tConvergent\tConvN\tDivergent\tDivN\n";
for (my $i=0;$i<$maxDistance/$binSize;$i++) {
	my $dd = $i*$binSize;
	print OUT "$i\t$dd";
	$r = $same[$i]+$conv[$i]+$div[$i];
	$n = $sameN[$i]+$convN[$i]+$divN[$i];
	$r /= $n if ($n>0);
	print OUT "\t$r\t$n";
	$r = $same[$i];
	$n = $sameN[$i];
	$r /= $n if ($n>0);
	print OUT "\t$r\t$n";
	$r = $conv[$i];
	$n = $convN[$i];
	$r /= $n if ($n>0);
	print OUT "\t$r\t$n";
	$r = $div[$i];
	$n = $divN[$i];
	$r /= $n if ($n>0);
	print OUT "\t$r\t$n";
	print OUT "\n";
}
close OUT;

open OUT, ">$prefix.local.txt";
print OUT "binIndex\tDistance\tAll\tAllN\tSameStrand\tSameN\tConvergent\tConvN\tDivergent\tDivN\n";
for (my $i=0;$i<$maxDistanceLocal/$binSizeLocal;$i++) {
	my $dd = $i*$binSizeLocal;
	print OUT "$i\t$dd";
	$r = $samelocal[$i]+$convlocal[$i]+$divlocal[$i];
	$n = $sameNlocal[$i]+$convNlocal[$i]+$divNlocal[$i];
	$r /= $n if ($n>0);
	print OUT "\t$r\t$n";
	$r = $samelocal[$i];
	$n = $sameNlocal[$i];
	$r /= $n if ($n>0);
	print OUT "\t$r\t$n";
	$r = $convlocal[$i];
	$n = $convNlocal[$i];
	$r /= $n if ($n>0);
	print OUT "\t$r\t$n";
	$r = $divlocal[$i];
	$n = $divNlocal[$i];
	$r /= $n if ($n>0);
	print OUT "\t$r\t$n";
	print OUT "\n";
}
close OUT;

if ($interChrFlag) {
	open OUT, ">$prefix.chromosomes.txt";
	print OUT "Comparison\tAvgCorr\tCount\n";
	$r = $interR;
	$n = $interN;
	$r /= $n if ($n>0);
	print OUT "InterChr\t$r\t$n\n";
	foreach(keys %chrR) {
		my $c = $_;
		print OUT "$c\t$chrR{$c}\t$chrN{$c}\n";
	}
	close OUT;
}


sub correlation {
	my ($x,$y) = @_;

	my $n = scalar(@$x);
	return 0 if ($n==0 || scalar(@$y) == 0);
	my $xysum = 0;
	my $xsum = 0;
	my $ysum = 0;
	my $x2sum = 0;
	my $y2sum = 0;
	for (my $i=0;$i<$n;$i++) {
		$xysum += $x->[$i]*$y->[$i];
		$xsum += $x->[$i];
		$ysum += $y->[$i];
		$x2sum += $x->[$i]*$x->[$i];
		$y2sum += $y->[$i]*$y->[$i];
	}
	my $r = "NA";
	my $numerator = $xysum - $xsum*$ysum/$n;
	my $denomerator = ($x2sum - $xsum*$xsum/$n)*($y2sum-$ysum*$ysum/$n);
	return ("NA","NA") if ($denomerator <= 0);
	$denomerator = sqrt($denomerator);
	if ($denomerator > 0) {
		$r = $numerator / $denomerator;
	}
	my $df = $n-2;
	my $den = ((1-$r)*(1+$r));
	if ($den < 1e-10) {
		$den = 1e-10;
	}
	my $t = $r*sqrt($df/($den));
	my $logp = 1;
	if (abs($t) > 1e-5) {
		$logp = logbetai(0.5*$df,0.5,$df/($df+$t*$t));
	}

	return ($r,$logp);
}
sub logbetai {
	my ($a, $b, $x) = @_;
	if ($x <= 0.0 || $x >= 1.0) {
		print STDERR "variable x has invalid value in function logbetai (x=$x)\n";
	}
	my $bt = gammln($a+$b)-gammln($a)-gammln($b)+$a*log($x)+$b*log(1.0-$x);
	if ($x < ($a+1.0)/($a+$b+2.0)) {
		return $bt+log(betacf($a,$b,$x))-log($a);
	} else {
		return log(1.0-exp($bt)*betacf($b,$a,1.0-$x)/$b);
	}

}

sub betacf {
	my ($a,$b,$x) = @_;
	my ($m,$m2,$aa,$c,$d,$del,$h,$qab,$qam,$qap) = (0,0,0,0,0,0,0,0,0,0);
	my ($MAXIT,$EPS,$FPMIN) = (100,3.0e-7,1.0e-30);

	$qab=$a+$b;
	$qap=$a+1.0;
	$qam=$a-1.0;
	$c=1.0;
	$d=1.0-$qab*$x/$qap;
	$d=$FPMIN if (abs($d) < $FPMIN);
	$d=1.0/$d;
	$h=$d;

	for (my $m=1;$m<=$MAXIT;$m++) {
		$m2=2*$m;
		$aa=$m*($b-$m)*$x/(($qam+$m2)*($a+$m2));
		$d=1.0+$aa*$d;
		$d=$FPMIN if (abs($d) < $FPMIN);
		$c=1.0+$aa/$c;
		$c=$FPMIN if (abs($c) < $FPMIN);
		$d=1.0/$d;
		$h *= $d*$c;
		$aa = -($a+$m)*($qab+$m)*$x/(($a+$m2)*($qap+$m2));
		$d=1.0+$aa*$d;
		$d=$FPMIN if (abs($d) < $FPMIN);
		$c=1.0+$aa/$c;
		$c=$FPMIN if (abs($c) < $FPMIN);
		$d=1.0/$d;
		$del=$d*$c;
		$h*=$del;
		last if (abs($del-1.0)<$EPS);
	}
    if ($m > $MAXIT) {
        print STDERR "Error calculating betacf\n";
        exit;
    }
    return $h;
}
sub gammln {
	my ($xx) = @_;
	
    my @cof=(76.18009172947146,-86.50532032941677,
            24.01409824083091,-1.231739572450155,
            0.1208650973866179e-2,-0.5396239384953e-5);
	my ($x,$y,$tmp,$ser)=(0,0,0,0);
	$y=$xx;
	$x=$xx;
	$tmp=$x+5.5;
	$tmp -= ($x+0.5)*log($tmp);
	$ser=1.000000000190015;
	for (my $j=0;$j<=5;$j++) {
		$ser += $cof[$j]/(++$y);
	}
	return -1*$tmp+log(2.5066282746310005*$ser/$x);
}

sub readTagInfo {
    my ($dir,$pc) = @_;

	my $file = "$dir/" . "/tagInfo.txt";
    my $p=0;
    my $t = 0;
    my $fragEst = "NA";
    my $peakEst = "NA";
	if (!defined($pc)) {
		$pc = 0;
	}
    open IN, $file or die "Could not open $file\n";
    my $count = 0;
    while (<IN>) {
        $count++;
        if ($count == 1) {
            if (/Estimated Fragment Length=(\d+?)[,\t]/) {
                $fragEst = $1;
            }
            if (/Estimated Peak Width=(\d+?)[,\t]/) {
                $peakEst = $1;
            }

        }
        next if ($count < 2);
        chomp;
        s/\r//g;
        my @line= split /\t/;
		if ($count == 2 || $line[0] eq 'genome') {
        	$p = $line[1];
        	$t = $line[2];
		} else {
			if ($line[0] =~ /fragmentLengthEstimate=(.+)$/) {
				$fragEst = $1;
			}
			if ($line[0] =~ /peakSizeEstimate=(.+)$/) {
				$peakEst = $1;
			}
		}
    }
    close IN;
	if ($pc == 1) {	
		$t = $p;
	} elsif ($pc > 1) {
		$file = "$dir/tagCountDistribution.txt";
		if (-e $file) {
			open IN, $file;	
			my $total = 0;
			my $count = 0;
			while (<IN>) {
				$count++;
				chomp;
				s/\r//g;
				my @line = split /\t/;
				next if ($count==1);
				$total += $line[1];
				if ($line[0] eq $pc) {
					last;
				}
			}
			close IN;
			#print STDERR "total=$total\n";
			$t *= $total;
		}
	}

    #print STDERR "$t, $p, $fragEst, $peakEst, $pc\n";
    return ($t, $p, $fragEst, $peakEst);
}
