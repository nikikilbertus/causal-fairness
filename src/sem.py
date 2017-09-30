import torch

from graph import Graph
from mlp import MLP, train
import utils


class SEM(Graph):
    """A representation of a structural eqaution model."""

    def __init__(self, graph):
        """
        Initialize a structural equation model.

        Arguments:

            graph: A dictionary, where the vertices of the graph are the keys
            and each value is a lists of parents(!) of the key vertex. Setting
            the value to `None` means that the vertex is a root of the graph.
        """
        super().__init__(graph)
        self.equations = {}
        self.learned = {}

    def _can_sample(self):
        """Check whether graph and equations are consistent."""
        vertices = set(self.vertices())
        eqs = set(self.equations.keys())
        assert vertices == eqs, \
            "Vertices: {}, equations for: {}".format(vertices, eqs)

    def attach_equation(self, vertex, equation):
        """Attach an equation or distribution to a vertex."""
        assert vertex in self.vertices(), "{} non-existent".format(vertex)
        print("Attaching equation to vertex {}...".format(vertex), end=' ')
        self.equations[vertex] = equation
        print("DONE")

    def sample(self, n_samples):
        """If possible, sample from the structural equation model."""
        self._can_sample()
        sample = {}
        for v in self.topological_sort():
            print("Sample vertex {}...".format(v), end=' ')
            if v in self.roots():
                sample[v] = self.equations[v](n_samples)
            else:
                sample[v] = self.equations[v](sample)
            print("DONE")
        return sample

    def learn_from_sample(self, sample=None, hidden_sizes=(), binarize=[],
                          **kwargs):
        """Learn the structural equations from data."""
        if sample is None:
            n_samples = 8192
            print("There was no sample provided to learn from.")
            print("Generate sample with {} examples.".format(n_samples))
            sample = self.sample(n_samples)
            learned_sample = True
        else:
            learned_sample = False

        for v in self.non_roots():
            parents = self.parents(v)
            print("Training {} -> {}...".format(parents, v), end=' ')
            data = utils.combine_variables(parents, sample)
            if v in binarize:
                final = torch.nn.Sigmoid()
            else:
                final = None
            net = MLP([data.size(-1), *hidden_sizes, 1], final=final)
            self.learned[v] = train(net, data, sample[v], **kwargs)
            print("DONE")

        if learned_sample:
            return sample

    def predict_from_sample(self, sample, update=None, mutate=False,
                            replace=[]):
        """Predict non-root variables in a sample for updated sample."""
        assert self.learned, "Must learn all SEMs before prediction."

        if mutate:
            new_sample = sample
        else:
            import copy
            new_sample = copy.deepcopy(sample)

        if update is None:
            update = [v for v in self.topological_sort()
                      if v not in self.roots()]

        print("Updating the vertices {}...".format(update), end=' ')
        for v in update:
            args = utils.combine_variables(self.parents(v),
                                           new_sample,
                                           as_var=True)
            if v in replace:
                new_sample[v] = replace[v](args).data
            else:
                new_sample[v] = self.learned[v](args).data
        print("DONE")
        if not mutate:
            return new_sample

    def print_learned_parameters(self, show=None, weights=True, biases=True):
        """Print the learned parameters."""
        if show is None:
            show = self.non_roots()

        for target, model in self.learned.items():
            if target in show:
                print("Parameters for ", target)
                for name, param in model.named_parameters():
                    if weights and 'weight' in name:
                        print("weights:")
                        print(param.data.numpy())
                    if biases and 'bias' in name:
                        print("biases:")
                        print(param.data.numpy())
