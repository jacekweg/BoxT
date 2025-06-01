import numpy as np

def mean_first_relevant(ranks: np.ndarray) -> float:
    """
    Mean First Relevant (MFR):
    Arithmetic mean of the rank of the first relevant item.
    """
    return ranks.mean()


def adjusted_mean_rank_index(ranks: np.ndarray, candidate_sizes: np.ndarray) -> float:
    """
    Adjusted Mean Rank Index (AMRI):
    AMRI = 1 - (2 * sum(r_i - 1)) / sum(|S_i|)
    """
    if len(ranks) != len(candidate_sizes):
        raise ValueError("`ranks` and `candidate_sizes` must have the same length.")
    numerator = 2 * np.sum(ranks - 1)
    denominator = np.sum(candidate_sizes)
    return 1.0 - numerator / denominator


def rank_biased_precision(ranks: np.ndarray, p: float = 0.85) -> float:
    """
    Rank-Biased Precision (RBP):
    RBP = (1-p) * sum(r_k * p^(k-1)) where r_k is the relevance at rank k.
    """
    ranks = np.maximum(ranks, 1.0)

    relevance_scores = 1.0 / ranks

    # Calculate p^(k-1) for each rank
    rank_discounts = np.power(p, ranks - 1)

    individual_rbp = (1 - p) * relevance_scores * rank_discounts

    return individual_rbp.mean()


def expected_first_relevant(ranks: np.ndarray,
                            stop_distribution: np.ndarray = None,
                            p: float = 0.9) -> float:
    max_rank = int(ranks.max())

    if stop_distribution is not None:
        if len(stop_distribution) < max_rank:
            raise ValueError("`stop_distribution` shorter than max rank in `ranks`.")
        pk = stop_distribution[:max_rank]
        pk = pk / pk.sum()
    else:
        k = np.arange(1, max_rank + 1)
        pk = (1 - p) * (p ** (k - 1))
        pk = pk / pk.sum()

    cumulative_p = np.cumsum(pk[::-1])[::-1]
    expected_effort = np.cumsum((k * pk)[::-1])[::-1]
    
    # Fix divide by zero warning
    epsilon = np.finfo(float).eps
    safe_cumulative_p = np.maximum(cumulative_p, epsilon)
    efr_lookup = expected_effort / safe_cumulative_p

    efr_lookup = np.nan_to_num(efr_lookup, nan=max_rank)

    efr_values = efr_lookup[ranks.astype(int) - 1]

    return efr_values.mean()