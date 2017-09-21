import copy
from inspect import signature

from graph import Graph
import utils


class SEM(Graph):
    """A representation of a structural eqaution model."""

    def __init__(self, graph):
        """Initialize a structural equation model."""
        super().__init__(graph)
        self.equations = {}

    def _check_signature(self, vertex, equation):
        """Check that the number of arguments matches the number of parents."""
        assert vertex in self.vertices(), "{} non-existent".format(vertex)

        n_args = len(signature(equation))
        if vertex in self.roots():
            assert n_args == 0, \
                "{} arguments for root {}".format(n_args, vertex)
        else:
            n_parents = len(self.parents(vertex))
            assert n_parents == n_args, \
                "{} arguments for {} parents of {}".format(n_args, )

    def _can_sample(self):
        """Check whether graph and equations are consistent."""
        vertices = set(self.vertices())
        eqs = (self.equations.keys())
        assert vertices == eqs, \
            "Vertices: {}, equations for: {}".format(vertices, eqs)

        for v in vertices:
            self._check_signature(v, self.equations[v])

    def attach_equation(self, vertex, equation):
        """Attach an equation or distribution to a vertex."""
        self._check_signature(vertex, equation)
        print("Attaching equation to vertex {}...".format(vertex), end=' ')
        self.equations[vertex] = equation
        print("DONE")

    def sample(self):
        sample = {}
        for v in self.topological_sort():
            print("Sample vertex {}...".format(v), end=' ')
            if v in self.roots():
                sample[v] = self.equations[v]()
            else:
                args = tuple([sample[p] for p in self.parents(v)])
                sample[v] = self.equations[v](*args)
            print("DONE")
        return sample

    def learn_from_sample(self, sample, learned):
        from torch.autograd import Variable
        new_sample = copy.deepcopy(sample)
        need_update = [v for v in self.topological_sort()
                       if v not in self.roots()]
        print("Updating the nodes {}.".format(need_update))
        for update in need_update:
            print("Updating node {}...".format(update), end=' ')
            argument = Variable(utils.combine_variables(self.parents(update),
                                                        new_sample))
            new_sample[update] = learned[update](argument).data
        print("DONE")
        return new_sample
