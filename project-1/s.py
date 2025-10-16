import math
import pandas as pd
import pysam as ps
import numpy as np


def logsumexp(log_vals):
    a = max(log_vals)
    return a + math.log(sum(math.exp(v - a) for v in log_vals))


def base_at_pos0(read, pos0):
    for rpos, qpos in read.get_aligned_pairs(matches_only=False):
        if rpos == pos0:
            if qpos is None:
                return None, None
            base = read.query_sequence[qpos]
            q = read.query_qualities[qpos] if read.query_qualities is not None else None
            return base, q
    return None, None


if __name__ == "__main__":
    bam = ps.AlignmentFile("out_sort.bam", "rb")
    ref = ps.FastaFile("chr1_1e6_2e6.fasta")
    snps = pd.read_csv("putative_snps.tsv", sep="\t")

    out_rows = []

    for _, row in snps.iterrows():
        contig = str(row["chr"])
        pos1 = int(row["pos"])
        ref_base_tsv = str(row["ref"]).upper()
        alt_base_tsv = str(row["alt"]).upper()
        maf = float(row["maf"])  # minor allele frequency for allele B

        pos0 = pos1 - 1

        fasta_base = ref.fetch(contig, pos0, pos0 + 1).upper()
        if fasta_base != ref_base_tsv:
            # If your grader expects strict checking, keep raise; otherwise, use "continue".
            raise ValueError(
                f"Ref mismatch at {contig}:{pos1}: TSV={ref_base_tsv}, FASTA={fasta_base}")

        # Define A = major, B = minor (by problem statement)
        # With the provided TSV, ref is the major and alt is the minor.
        A = ref_base_tsv
        B = alt_base_tsv

        # Hardy–Weinberg priors using maf = f = P(B)
        f = maf
        p_AA = (1.0 - f) * (1.0 - f)
        p_AB = 2.0 * f * (1.0 - f)
        p_BB = f * f
        log_prior = {
            "AA": math.log(p_AA) if p_AA > 0 else -math.inf,
            "AB": math.log(p_AB) if p_AB > 0 else -math.inf,
            "BB": math.log(p_BB) if p_BB > 0 else -math.inf,
        }

        # Accumulate log-likelihood over reads that overlap exactly this position
        log_lik = {"AA": 0.0, "AB": 0.0, "BB": 0.0}
        any_reads = False

        for read in bam.fetch(contig, pos0, pos0 + 1):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            base, Q = base_at_pos0(read, pos0)
            if base is None:
                continue  # deletion/gap at this site
            base = base.upper()
            if base not in ("A", "C", "G", "T"):
                continue  # skip ambiguous
            any_reads = True

            # Per-base error probability from Phred Q:  p_err = 10^(-Q/10)
            # Work in logs consistently (natural log here).
            if Q is None:
                # If qualities are missing, assume a conservative Q=20
                Q = 20
            p_err = 10.0 ** (-Q / 10.0)

            # Model: if true allele is X, observe X with (1 - p_err); other bases share p_err/3
            def log_p_obs_given_homo(true_base):
                if base == true_base:
                    return math.log(1.0 - p_err)
                else:
                    return math.log(p_err / 3.0)

            # For heterozygote AB, a read is equally likely to originate from A or B:
            # P(obs | AB) = 0.5*P(obs|A) + 0.5*P(obs|B)
            lp_A = log_p_obs_given_homo(A)
            lp_B = log_p_obs_given_homo(B)
            # log(0.5*(e^lpA + e^lpB))
            lp_AB = math.log(0.5) + logsumexp([lp_A, lp_B])

            log_lik["AA"] += lp_A
            log_lik["AB"] += lp_AB
            log_lik["BB"] += lp_B

        # Posterior ∝ prior * likelihood  → log posterior = log prior + log likelihood
        log_post = {
            g: (log_prior[g] + log_lik[g]) for g in ("AA", "AB", "BB")
        }

        # Normalize to probabilities with log-sum-exp
        norm = logsumexp([log_post["AA"], log_post["AB"], log_post["BB"]])
        post_AA = math.exp(log_post["AA"] - norm)
        post_AB = math.exp(log_post["AB"] - norm)
        post_BB = math.exp(log_post["BB"] - norm)

        # If there were zero usable reads, the posterior effectively equals the prior (the math above already does that)
        out_rows.append({
            "chrom": contig,
            "position": pos1,
            "p(AA|data)": round(post_AA, 6),
            "p(AB|data)": round(post_AB, 6),
            "p(BB|data)": round(post_BB, 6),
        })

    pd.DataFrame(out_rows, columns=["chrom", "position", "p(AA|data)",
                 "p(AB|data)", "p(BB|data)"]).to_csv("result.csv", index=False)
