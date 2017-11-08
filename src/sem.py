import torch

from graph import Graph
from mlp import MLP, train
import utils


class SEM(Graph):
    """A representation of a structural eqaution model."""

    def __init__(self, graph):
        """
        Initialize a structural equation model.

        A structural equation model entails a causal graph. Here, we start from
        the graph and then attach the equations/distributions for each vertex
        in the graph. Hence, `SEM` inherits from `Graph` and is initialized
        with the same argument as a `Graph` object.

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

    def _get_hidden(self, hidden_sizes, vertex):
        """
        Extract the sizes of hidden layers for a vertex.

        For flexibility, there is a bit of logic and fallbacks to default
        values to extract the sizes of the hidden layers.
        """
        if isinstance(hidden_sizes, dict):
            if vertex in hidden_sizes:
                hidden = hidden_sizes[vertex]
            elif None in hidden_sizes:
                hidden = hidden_sizes[None]
            else:
                hidden = ()
        else:
            hidden = hidden_sizes
        return hidden

    def attach_equation(self, vertex, equation):
        """
        Attach an equation or distribution to a vertex.

        In an SEM each vertex is determined by a function of its parents (and
        independent noise), except for root vertices, which follow some
        specified distribution.

        Arguments:

            vertex: The vertex for which we attach the equation.

            equation: A callable with a single argument. For a root vertex,
            this is the number of samples to draw. For non-root vertices the
            argument is a dictionary, where the keys are the parent vertices
            and the values are torch tensors containing the data.

        Examples:

            For a root vertex 'X' following a standard normal distribution:

            >>> sem.attach_equation('X', lambda n: torch.randn(n, 1))

            To attach a standard normal to all root vertices, we can run:

            >>> for v in sem.roots():
            >>>     sem.attach_equation(v, lambda n: torch.randn(n, 1))

            For a non-root vertex 'Z' that is the sum of its two parents 'X'
            and 'Y', we call:

            >>> sem.attach_equation('Z', lambda data: data['X'] +  data['Y'])
        """
        assert vertex in self.vertices(), "{} non-existent".format(vertex)
        print("Attaching equation to vertex {}...".format(vertex), end=' ')
        self.equations[vertex] = equation
        print("DONE")

    def sample(self, n_samples):
        """
        If possible, sample from the structural equation model.

        We can only sample from the SEM if each vertex has an equation
        attached, the graph is an acyclic DAG and the attached equations are
        consistent with the graph structure.

        Arguments:

            n_samples: The size of the sample to draw.

        Returns:

            The sample as a dictionary with the vertices as keys and torch
            tensors as values.
        """
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
        """
        Learn the structural equations from data.

        Given observed data for all vertices in the graph, we can learn the
        structural equations, i.e., how each vertex depends on its parents by
        supervised training.

        Arguments:

            sample: Either the observed data as a dictionary or an integer
            indicating the sample size, or none. For the data dictionary, the
            keys are the vertices and the values are torch tensors. For an
            integer > 0, we try to draw a sample of that size from analytically
            attached equations (see `attach_equation`). Default is `None`,
            which is equivalent to `sample=8192`.

            hidden_sizes: Either a list/tuple or a dictionary. If list or
            tuple, it contains the numbers of neurons in each layer.
            `len(hidden_sizes)-1` is the number of layers. First and last
            entries are input and output dimension. If dictionary, we can
            choose a different architecture for each vertex. The keys are (a
            subset) of vertices and the values are lists or tuples as before.

            binarize: A list or tuple of variables that take binary values
            (always 0/1). For those variables, a final sigmoid layer is
            attached to the network.

            **kwargs: Further named variables that are passed on to the `train`
            function in the `mlp` module (where they are passed to
            `torch.optim.Adam`). `**kwargs` can also contain a `dropout`
            argument.
        """
        # Was a sample provided or do we need to generate one
        if sample is None or isinstance(sample, int):
            n_samples = 8192 if sample is None else sample
            print("There was no sample provided to learn from.")
            print("Generate sample with {} examples.".format(n_samples))
            sample = self.sample(n_samples)
            learned_sample = True
        else:
            learned_sample = False

        dropout = kwargs.pop('dropout', 0.0)

        # Build and train network for all non-roots
        for v in self.non_roots():
            parents = self.parents(v)
            print("Training {} -> {}...".format(parents, v), end=' ')
            data = utils.combine_variables(parents, sample)
            if v in binarize:
                final = torch.nn.Sigmoid()
            else:
                final = None
            hidden = self._get_hidden(hidden_sizes, v)
            net = MLP([data.size(-1), *hidden, sample[v].size(-1)],
                      final=final, dropout=dropout)
            self.learned[v] = train(net, data, sample[v], **kwargs)
            print("DONE")

        if learned_sample:
            return sample

    def predict_from_sample(self, sample, update=None, mutate=False,
                            replace={}):
        """
        Predict non-root variables in a sample for an updated sample.

        Given an input sample, update (some of) the non-root vertices in
        topological order using the learned equations using the values in the
        sample for each vertex that is not updated.

        Arguments:

            sample: A sample from the sem, i.e., a dictionary where the keys
            are the vertices and the values are torch tensors.

            update: A list or tuple of non-root vertices to be updated. Default
            is `None`, in which case all non-root vertices are updated.

            mutate: A boolean indicating whether the input sample should be
            overwritten with the updated values, or a copy should be returned.
            (default `False`, i.e. a copy is returned).

            replace: A dictionary, where the keys are vertices for which
            another equation should be used than the learned one in
            `sem.learned`. The values are networks that represent a structural
            equation for the corresponding vertex. (default {}, i.e. use the
            internal learned equations for all vertices.)

        Examples:

            Draw update all non-roots with the learned equations from the base
            sample. This is useful to compare whether the learned functions
            indeed replicate the original data generating process:

            >>> predicted_sample = sem.predict_from_sample(base_sample)

            Once we have specified interventions and retrained the equations to
            get a corrected net `corrected` for the target variable `Y`, we can
            also update observed data with this discrimination free predictor:

            >>> fair_sample = sem.predict_from_sample(base_sample,
                                                      update=['Y'],
                                                      replace={'Y': corrected})
        """
        assert self.learned, "Must learn all SEMs before prediction."

        if mutate:
            new_sample = sample
        else:
            import copy
            new_sample = copy.deepcopy(sample)

        if update is None:
            update = [v for v in self.topological_sort()
                      if v not in self.roots()]
        else:
            assert not any(x in self.roots() for x in update), \
                "Cannot update root vertex."

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
        """Print all learned parameters for vertices in `show`."""
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
