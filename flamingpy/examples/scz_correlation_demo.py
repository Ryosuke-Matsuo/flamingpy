"""Visualize how the SCZ transform introduces correlations between CV modes."""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from flamingpy.codes.surface_code import SurfaceCode
from flamingpy.cv.ops import SCZ_apply
from flamingpy.noise import CVLayer


def sample_pairs(layer, coords, *, num_samples=4000, seed=1234):
    """Return homodyne samples with and without SCZ entanglement."""
    rng = np.random.default_rng(seed)
    layer.populate_states(rng=rng)
    # Ensure downstream calls see a dense ndarray (helps static analysers, too).
    adj = layer.egraph.adj_generator(sparse=False)
    if hasattr(adj, "toarray"):
        adj = adj.toarray()
    else:
        adj = np.asarray(adj)
    indices = [layer.egraph.to_indices[coord] for coord in coords]
    num_modes = layer._N
    raw = np.zeros((num_samples, len(indices)), dtype=np.float64)
    scz = np.zeros_like(raw)
    for i in range(num_samples):
        means = layer._means_sampler(rng=rng)
        covs = layer._covs_sampler()
        sample = rng.normal(means, covs)
        raw[i] = sample[num_modes:][indices]
        entangled = np.asarray(SCZ_apply(adj, sample.copy()))
        scz[i] = entangled[num_modes:][indices]
    return raw, scz


def corr_matrix(data):
    """Compute the column-wise correlation matrix."""
    if data.shape[1] == 1:
        return np.array([[1.0]])
    return np.corrcoef(data, rowvar=False)


def plot_correlations(raw_corr, scz_corr, labels):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    matrices = [(raw_corr, "Without SCZ"), (scz_corr, "With SCZ")]
    im = None
    for ax, (matrix, title) in zip(axes, matrices):
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
    if im is None:
        raise RuntimeError("No correlation matrices were plotted; check input data.")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Correlation")
    fig.suptitle("Syndrome-mode p-homodyne correlations")
    fig.tight_layout()
    return fig


def main(*, distance=2, delta=0.2, num_modes=6, num_samples=4000, seed=1234):
    code = SurfaceCode(distance=distance, ec="primal", boundaries="periodic")
    layer = CVLayer(code, delta=delta, sampling_order="initial")
    coords = list(code.all_syndrome_coords)
    if num_modes is not None:
        if num_modes > len(coords):
            raise ValueError(
                f"Requested {num_modes} modes but only {len(coords)} are available for distance={distance}."
            )
        coords = coords[:num_modes]
    raw, scz = sample_pairs(layer, coords, num_samples=num_samples, seed=seed)
    raw_std = raw.std(axis=0)
    scz_std = scz.std(axis=0)
    keep = (raw_std > 1e-12) & (scz_std > 1e-12)
    if not np.all(keep):
        dropped = len(keep) - int(np.count_nonzero(keep))
        print(f"Dropping {dropped} deterministic mode(s) with zero variance.")
        coords = [coord for coord, flag in zip(coords, keep) if flag]
        raw = raw[:, keep]
        scz = scz[:, keep]
    raw_corr = corr_matrix(raw)
    scz_corr = corr_matrix(scz)
    labels = [str(coord) for coord in coords]
    plot_correlations(raw_corr, scz_corr, labels)
    print("Correlation matrix without SCZ:\n", raw_corr)
    print("\nCorrelation matrix with SCZ:\n", scz_corr)
    print("\nAverage absolute correlation increase:", np.mean(np.abs(scz_corr - raw_corr)))
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, default=2, help="Surface code distance (default: 2)")
    parser.add_argument("--delta", type=float, default=0.2, help="Squeezing/noise parameter delta")
    parser.add_argument(
        "--num-modes",
        type=int,
        default=6,
        help="Number of syndrome modes to visualise (default: 6, set negative to use all)",
    )
    parser.add_argument("--num-samples", type=int, default=4000, help="Number of Monte Carlo samples")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility")
    args = parser.parse_args()
    num_modes = None if args.num_modes is not None and args.num_modes < 0 else args.num_modes
    main(
        distance=args.distance,
        delta=args.delta,
        num_modes=num_modes,
        num_samples=args.num_samples,
        seed=args.seed,
    )
