import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
from scipy.stats import linregress
import numpy as np


def combine_variables(variables, sample, as_var=False):
    """Stack variables from sample along new axis."""
    data = torch.cat([sample[i] for i in variables], dim=1)
    if as_var:
        return Variable(data)
    else:
        return data


def plot_samples(graph, samples, legend=None, **kwargs):
    """Plot all relevant dependencies in a graph from a/multiple sample(s)."""
    # If we did not already receive a list of samples, make one element list
    if not isinstance(samples, list):
        samples = [samples]
    # Get non root variables
    non_roots = graph.non_roots()
    # Get maximum number of input variables
    max_deps = max([len(graph.parents(var)) for var in non_roots])

    fig, axs = plt.subplots(len(non_roots), max_deps,
                            figsize=(5 * max_deps, 5 * len(non_roots)))

    # Go through all dependencies and plot them as 2D scatter plots
    for i, y_var in enumerate(non_roots):
        for j, x_var in enumerate(graph.parents(y_var)):
            for sample in samples:
                axs[i, j].plot(sample[x_var].numpy(),
                               sample[y_var].numpy(), '.', **kwargs)
                axs[i, j].set_xlabel(x_var)
                axs[i, j].set_ylabel(y_var)
                if legend:
                    axs[i, j].legend(legend)
    plt.tight_layout()
    plt.show()


def correlations(sample, sem=None, sources=None, targets=None):
    """Compute the correlations of all specified dependencies."""
    if not sources:
        sources = sem.vertices()
    if not targets:
        targets = sem.vertices()

    values = ['slope', 'rvalue', 'pvalue', 'intercept', 'stderr']
    corr = {}
    for v in values:
        corr[v] = np.zeros((len(targets), len(sources)))

    for i, t in enumerate(targets):
        for j, s in enumerate(sources):
            sdata = sample[s].numpy().squeeze()
            tdata = sample[t].numpy().squeeze()
            model = linregress(sdata, tdata)
            for v in values:
                corr[v][i, j] = getattr(model, v)
    return corr


def print_correlations(sample, sem=None, sources=None, targets=None):
    """Print the correlations of all specified dependencies."""
    if not sources:
        sources = sem.vertices()
    if not targets:
        targets = sem.vertices()

    corr = correlations(sample, sem=sem, sources=sources, targets=targets)

    for i, t in enumerate(targets):
        for j, s in enumerate(sources):
            print("\n\nRelation between {} and {}:".format(s, t))
            [print("{}: {:.4f}, ".format(name, value[i, j]), end='')
             for name, value in corr.items()]


def plot_correlations(sample, sem=None, sources=None, targets=None):
    """Plot the correlations of all specified dependencies."""
    if not sources:
        sources = sem.vertices()
    if not targets:
        targets = sem.vertices()

    corr = correlations(sample, sem=sem, sources=sources, targets=targets)

    for label, data in corr.items():
        plt.figure(figsize=(5, 5))
        plt.imshow(data)
        plt.title(label)
        plt.xticks(range(len(sources)), sources)
        plt.yticks(range(len(targets)), targets)
        plt.colorbar()
        plt.show()


def evaluate_on_new_sample(sem, target, corrected, n_sample=8192, plot=True):
    """Evaluate the learned and corrected versions on a new sample."""
    base = sem.sample(n_sample)
    orig = sem.predict_from_sample(base)
    fair = sem.predict_from_sample(base, replace={target: corrected})
    fair_target = target + 'fair'
    orig[fair_target] = fair[target]
    if plot:
        plot_samples(sem, [base, orig, fair],
                     legend=['base', 'learned', 'fair'], alpha=0.3)
    return base, orig, fair
