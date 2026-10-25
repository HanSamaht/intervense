"""Structure discovery backends from the causal-learn library.

The intervention graph in causal_graph.py answers "how strong is this
edge". This module answers a different question: "which edges should
exist at all". It runs constraint-based or functional discovery on the
background data augmented with the model prediction column, so the
prediction node is a first-class citizen of the discovered structure.

Supported methods:
  - "pc":     PC constraint-based algorithm (conditional independence)
  - "ges":    Greedy Equivalence Search (BIC score)
  - "lingam": DirectLiNGAM (linear non-Gaussian, gives a full DAG)

Robustness contract for real data:
  - Rows containing NaN or Inf are dropped once, before any backend
    runs, and the count is reported in every edge detail line.
  - Every discovered edge carries an estimated effect size, not a
    placeholder: for each edge i -> j the module fits an OLS of j on
    i plus the other discovered parents of j, and reports the
    standardized partial coefficient of i. PC and GES decide which
    edges exist; the regression measures how strong they are.
  - Backends run under a dimension guard: features beyond
    max_features are pre-filtered by absolute correlation with the
    prediction column, and the reduced set is recorded.
"""
from __future__ import annotations

import numpy as np

from ..core.types import CausalEdge

try:  # optional dependency, soft fallback
    from causallearn.search.ConstraintBased.PC import pc as _pc
    from causallearn.search.FCMBased.lingam import DirectLiNGAM
    from causallearn.search.ScoreBased.GES import ges as _ges

    HAS_CAUSAL_LEARN = True
except Exception:  # pragma: no cover - exercised only without the library
    HAS_CAUSAL_LEARN = False

DISCOVERY_METHODS = ("pc", "ges", "lingam")


def _augment_with_predictions(driver, data: np.ndarray) -> np.ndarray:
    """Append the model prediction column to the background matrix."""
    preds = driver.predict(data).reshape(-1, 1)
    return np.hstack([np.asarray(data, dtype=float), preds])


def _drop_bad_rows(data: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove rows with any non-finite value. Real panel data arrives
    with NaN cells; discovery backends refuse the whole matrix."""
    finite = np.isfinite(data).all(axis=1)
    return data[finite], int((~finite).sum())


def _dimension_guard(data: np.ndarray, names: list[str],
                     max_features: int, target_idx: int):
    """Keep the features most correlated with the target when the
    column count exceeds the guard. The prediction column always stays.
    Returns reduced (data, names, target_idx, dropped_names)."""
    if data.shape[1] - 1 <= max_features:
        return data, names, target_idx, []
    y = data[:, target_idx]
    feature_cols = [j for j in range(data.shape[1]) if j != target_idx]
    corr = np.array([abs(np.corrcoef(data[:, j], y)[0, 1])
                     if np.std(data[:, j]) > 1e-12 else 0.0
                     for j in feature_cols])
    order = np.argsort(-corr)[:max_features]
    keep_cols = sorted([feature_cols[j] for j in order] + [target_idx])
    dropped = [names[j] for j in feature_cols if j not in keep_cols]
    return (data[:, keep_cols],
            [names[j] for j in keep_cols],
            keep_cols.index(target_idx),
            dropped)


def _cpdag_edges(graph: np.ndarray, names: list[str]):
    """Extract edges from a causal-learn CPDAG matrix.

    Encoding: graph[i, j] == -1 and graph[j, i] == 1 means i -> j.
    graph[i, j] == -1 and graph[j, i] == -1 is an undirected edge whose
    orientation the algorithm cannot determine from data alone.

    Returns (i, j, directed) tuples; effect sizes are attached later by
    the regression pass so PC and GES edges report measured strengths.
    """
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if graph[i, j] == -1 and graph[j, i] == 1:
                out.append((i, j, True))
            elif graph[i, j] == 1 and graph[j, i] == -1:
                out.append((j, i, True))
            elif graph[i, j] == -1 and graph[j, i] == -1:
                out.append((i, j, False))
    return out


def _partial_effects(data: np.ndarray, triples):
    """OLS effect for each edge i -> j given the other parents that the
    same graph proposes for j. Coefficients are standardized (unit
    standard deviation inputs and output) so magnitudes compare across
    edges. Undirected edges are treated as i -> j with a note."""
    parents: dict[int, list[int]] = {}
    for i, j, directed in triples:
        if directed:
            parents.setdefault(j, []).append(i)
    out = []
    for i, j, directed in triples:
        others = [p for p in parents.get(j, []) if p != i]
        cols = [i] + others
        X = data[:, cols]
        y = data[:, j]
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            sx = X[:, 0].std()
            sy = y.std()
            eff = float(beta[0] * sx / sy) if sx > 1e-12 and sy > 1e-12 else 0.0
        except np.linalg.LinAlgError:
            eff = 0.0
        out.append((i, j, eff, directed))
    return out


def discovery_edges(
    driver,
    method: str = "pc",
    alpha: float = 0.05,
    max_samples: int = 400,
    max_edges: int = 15,
    max_features: int = 60,
) -> list[CausalEdge]:
    """Run structure discovery over features plus the prediction node.

    Returns edges of kind "discovered". Feature-to-feature edges are
    structural hypotheses. Feature-to-prediction edges mark features the
    discovery algorithm places as direct causes of the model output.
    Prediction-to-feature edges, if any, are diagnostics: the discovery
    backend sees the output as upstream of an input, which usually means
    the feature is a deterministic function of other inputs.

    Every edge carries a standardized partial effect estimated on the
    same data the backend scored, plus a note naming the method and the
    data reduction steps that were applied.
    """
    if not HAS_CAUSAL_LEARN:
        raise ImportError(
            "causal-learn is not installed. Install it with: pip install causal-learn"
        )
    if method not in DISCOVERY_METHODS:
        raise ValueError(f"unknown discovery method {method!r}, expected one of {DISCOVERY_METHODS}")

    # NaN rows must go before predict: sklearn estimators refuse
    # non-finite inputs outright
    bg, n_bad_rows = _drop_bad_rows(
        np.asarray(driver.background[:max_samples], dtype=float))
    data = _augment_with_predictions(driver, bg)
    names = list(driver.feature_names) + ["prediction"]
    target_idx = len(names) - 1

    # discovery backends choke on degenerate matrices: a constant one-hot
    # column makes the correlation matrix singular and PC never terminates.
    stds = data.std(axis=0)
    keep_idx = np.where(stds > 1e-12)[0]
    dropped_const = [names[j] for j in np.where(stds <= 1e-12)[0]]
    if len(keep_idx) < 3:
        raise ValueError("fewer than 3 usable columns after NaN and constant filtering, nothing to discover")
    if dropped_const:
        data = data[:, keep_idx]
        names = [names[j] for j in keep_idx]
        target_idx = names.index("prediction") if "prediction" in names else -1

    data, names, target_idx, dropped_dim = _dimension_guard(
        data, names, max_features, target_idx)

    reduction = []
    if n_bad_rows:
        reduction.append(f"{n_bad_rows} rows dropped for non-finite values")
    if dropped_const:
        reduction.append(f"constant columns dropped: {', '.join(dropped_const)}")
    if dropped_dim:
        reduction.append(
            f"dimension guard kept top {max_features} features by |corr| with prediction; dropped: "
            + ", ".join(dropped_dim))
    note_tail = ("; " + "; ".join(reduction)) if reduction else ""

    if method == "pc":
        # max_cond_size caps the conditioning set size: PC is exponential
        # in it and dense one-hot backgrounds push it to order 6 and beyond
        res = _pc(data, alpha=alpha, indep_test="fisherz",
                  max_cond_size=3, show_progress=False)
        triples = _cpdag_edges(res.G.graph, names)
    elif method == "ges":
        res = _ges(data, score_func="local_score_BIC")
        triples = _cpdag_edges(res["G"].graph, names)
    else:  # lingam
        model = DirectLiNGAM()
        model.fit(data)
        adj = model.adjacency_matrix_
        weighted = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if i != j and abs(adj[i, j]) > 1e-6:
                    weighted.append((i, j, float(adj[i, j])))
        weighted.sort(key=lambda t: abs(t[2]), reverse=True)
        # LiNGAM carries its own standardized weights; keep them
        directed = [(i, j, adj[i, j], True) for i, j, _ in weighted]
    if method != "lingam":
        directed = _partial_effects(data, triples)

    # Prediction-related edges always survive the cap; the rest compete.
    pred_edges = [e for e in directed if target_idx in (e[0], e[1])]
    struct_edges = [e for e in directed if target_idx not in (e[0], e[1])]
    keep = pred_edges + struct_edges[: max(0, max_edges - len(pred_edges))]

    edges: list[CausalEdge] = []
    for i, j, w, is_directed in keep:
        src, dst = names[i], names[j]
        arrow = "->" if is_directed else "-"
        if j == target_idx:
            detail = f"{method} discovery: {src} {arrow} prediction direct cause, standardized effect {w:+.3f}{note_tail}"
        elif i == target_idx:
            detail = (
                f"{method} discovery: prediction placed upstream of {dst}; "
                "the feature is likely a deterministic function of other inputs"
                f"{note_tail}"
            )
        else:
            note = "" if is_directed else " (orientation undetermined, CPDAG)"
            detail = f"{method} discovery: {src} {arrow} {dst} structural hypothesis, standardized effect {w:+.3f}{note}{note_tail}"
        edges.append(CausalEdge(source=src, target=dst, effect=float(w), kind="discovered", detail=detail))
    return edges
