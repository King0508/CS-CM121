# CS CM121/221 – Project 1 (Part 1): SNP Caller

This repo contains my Part 1 solution for building a simple **SNP caller**.  
A **SNP (Single Nucleotide Polymorphism)** is a position in the genome where an individual’s base
(A/C/G/T) may differ from the **reference** genome.

## What my script does

Given:

- `chr1_1e6_2e6.fasta` — reference sequence for a window on chr1
- `out_sort.bam` (+ `.bai`) — aligned reads for this window
- `putative_snps.tsv` — candidate SNP sites with columns: `chr, pos, ref, alt, maf`

The script:

1. Reads each candidate site from the TSV.
2. Verifies the reference base at that position (catches off-by-one mistakes).
3. Pulls all reads that overlap the position.
4. For each read that truly covers the base (not a deletion/ref-skip), records:
   - `read_name`
   - `position` (1-based, as in the TSV)
   - `observation` (A/C/G/T from the read)
   - `phred` (the base quality score)
5. Writes a table `result.csv` with one row per (site × read) observation.

This is the required output for Part 1.
