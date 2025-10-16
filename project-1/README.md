# CS CM121/221 – Project 1 (Part 1 and Part 2): SNP Caller

This directory contains my solution for building a simple **SNP caller**.  
A **SNP (Single Nucleotide Polymorphism)** is a position in the genome where an individual’s base
(A/C/G/T) may differ from the **reference** genome.

Part 1: For each putative SNP, scan aligned reads and report what each read shows at that exact position.

Part 2: For the same SNPs, combine all read-level evidence and estimate posterior genotype probabilities: p(AA|data), p(AB|data), p(BB|data) where A = major allele and B = minor allele (from the TSV MAF).

I used Python + pandas + pysam. The script is found in solution.py.
