import pandas as pd
import pysam as ps

if __name__ == "__main__":
    bam = ps.AlignmentFile("out_sort.bam", "rb")
    ref = ps.FastaFile("chr1_1e6_2e6.fasta")
    snps = pd.read_csv("putative_snps.tsv", sep="\t")

    fasta_contig = ref.references[0]
    window_mode = fasta_contig.startswith("chr1:1000000-2000000")
    window_start = 1_000_000

    rows = []

    for _, r in snps.iterrows():
        pos = int(r["pos"])
        if window_mode:
            chrom = fasta_contig
            pos1 = pos
        else:
            chrom = "chr1"
            pos1 = window_start + pos - 1

        ref_base = ref.fetch(chrom, pos1 - 1, pos1).upper()
        if ref_base != str(r["ref"]).upper():
            raise ValueError(
                f"Reference mismatch at {chrom}:{pos1} (FASTA={ref_base}, TSV.ref={r['ref']})"
            )

        for aln in bam.fetch(chrom, pos1 - 1, pos1):
            for qpos, rpos in aln.get_aligned_pairs(matches_only=False):
                if rpos == pos1 - 1:
                    if qpos is not None:
                        rows.append(
                            {
                                "read_name": aln.query_name,
                                "position": pos1,
                                "observation": aln.query_sequence[qpos],
                                "phred": aln.query_qualities[qpos],
                            }
                        )
                    break

    final_df = pd.DataFrame(
        rows, columns=["read_name", "position", "observation", "phred"]
    )
    final_df.to_csv("result.csv", index=False)
