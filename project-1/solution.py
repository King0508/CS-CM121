import pandas as pd
import pysam as ps
import numpy as np
import math


def logsumexp(log_val):
    a = max(log_val)
    return a + math.log(sum(math.exp(v - a) for v in log_val))


def base_at_pos0(read, pos0):
    for readpos, refpos in read.get_aligned_pairs(matches_only=False):
        if refpos == pos0:  # if we are at the part of the read that is pos0
            if readpos is None:  # if the
                return None, None
            # locates
            base = None if read.query_sequence[readpos] is None else read.query_sequence[readpos]
            q = None if read.query_qualities[readpos] is None else read.query_qualities[readpos]
            return base, q
    return None, None


if __name__ == "__main__":
    snps = pd.read_csv("putative_snps.tsv", sep="\t")
    ref = ps.FastaFile("chr1_1e6_2e6.fasta")  # database of genes
    bam = ps.AlignmentFile("out_sort.bam", "rb")  # all the reads

    out_rows = []
    for _, row in snps.iterrows():
        contig = str(row["chr"])
        pos1 = int(row["pos"])
        ref_base_tsv = str(row["ref"])
        alt_base_tsv = str(row["alt"])
        maf = float(row["maf"])
        pos0 = pos1 - 1

        A = ref_base_tsv
        B = alt_base_tsv
        # Using Hardy-Weinberg
        p_AA = (1 - maf) * (1 - maf)
        p_AB = 2 * (1 - maf) * maf
        p_BB = maf * maf

        log_prior = {
            "AA": math.log(p_AA) if p_AA > 0 else -math.inf,
            "AB": math.log(p_AB) if p_AB > 0 else -math.inf,
            "BB": math.log(p_BB) if p_BB > 0 else -math.inf,
        }
        log_lik = {"AA": 0.0, "AB": 0.0, "BB": 0.0}

        for read in bam.fetch(contig, pos0, pos1):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            # Need to get the base and the phred score of this base read at pos0
            base, Q = base_at_pos0(read, pos0)
            if base is None or Q is None:
                continue
            p_err = 10.0 ** (-Q / 10.0)

            def logp_given_true_base(true_base):
                if base == true_base:
                    return math.log(1 - p_err)
                else:
                    return math.log(p_err / 3.0)
            lp_A = logp_given_true_base(A)
            lp_B = logp_given_true_base(B)
            lp_AB = math.log(0.5) + logsumexp([lp_A, lp_B])

            log_lik["AA"] += lp_A
            log_lik["AB"] += lp_AB
            log_lik["BB"] += lp_B

        # Posterior ∝ prior * likelihood
        # Log posterior ∝ log prior + log likelihood
        log_post = {
            g: (log_prior[g] + log_lik[g]) for g in ("AA", "AB", "BB")
        }

        norm = logsumexp([log_post["AA"], log_post["AB"], log_post["BB"]])
        post_AA = math.exp(log_post["AA"] - norm)
        post_AB = math.exp(log_post["AB"] - norm)
        post_BB = math.exp(log_post["BB"] - norm)

        out_rows.append({
            "chrom": contig,
            "position": pos1,
            "AA": round(post_AA, 6),
            "AB": round(post_AB, 6),
            "BB": round(post_BB, 6),
        })

    pd.DataFrame(out_rows, columns=["chrom", "position", "AA",
                 "AB", "BB"]).to_csv("result.csv", index=False)
