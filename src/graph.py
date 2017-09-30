import collections
import copy
import networkx as nx


class Graph:
    """A light weight, self made graph representation."""

    def __init__(self, graph):
        """
        Initialize a Graph object.

        Arguments:

            graph: A dictionary, where the vertices of the graph are the keys
            and the values are the parents(!) of the key vertex. Setting the
            value to `None` means that the vertex is a root of the graph.
        """
        # If the input is a dict, assume it is already in the correct format
        if isinstance(graph, dict):
            self.graph = graph
        else:
            print(("Could not process the input {} as a graph."
                   " Initialized empty graph.").format(graph))
            self.graph = None

    def __repr__(self):
        """Define the representation."""
        import pprint
        return pprint.pformat(self.graph)

    def __str__(self):
        """Define the string format."""
        import pprint
        return pprint.pformat(self.graph)

    def __iter__(self):
        """Make the graph iterable."""
        return iter(self.graph)

    def __getitem__(self, item):
        """Make the graph usable as the internal dictionary representation."""
        return self.graph[item]

    def _try_add_vertex(self, vertex):
        """Try to add a new vertex to the graph."""
        if vertex in self.graph:
            print("Vertex already exists.")
        else:
            self.graph[vertex] = None
            print("Added vertex ", vertex)

    def _try_add_edge(self, source, target):
        """Try to add a new edge to the graph or abort."""
        if source in self.graph:
            if target not in self.graph[source]:
                self.graph[source].append(target)
            else:
                print("Edge already exists.")
        else:
            self.graph[source] = [target]

    def add_vertices(self, vertices):
        """Add one or multiple vertices to the graph."""
        if isinstance(vertices, collections.Iterable):
            for v in vertices:
                self._try_add_vertex(v)
        else:
            self._try_add_vertex(v)

    def add_edge(self, source, target):
        """Add a single edge from source to target."""
        self._try_add_edge(source, target)

    def vertices(self):
        """Find all vertices."""
        return list(self.graph.keys())

    def edges(self):
        """Find all edges."""
        edges = []
        for node, parents in self.graph.items():
            if parents is not None:
                for p in parents:
                    edges.append({p: node})
        return edges

    def roots(self):
        """Find all root vertices."""
        return [node for node in self.graph if self.graph[node] is None]

    def non_roots(self):
        return [node for node in self.graph if self.graph[node] is not None]

    def leafs(self):
        """Find all leaf vertices."""
        return list(set(self.vertices()).difference(self.non_leafs()))

    def non_leafs(self):
        """Find all non-leaf vertices."""
        return list(set(sum([p for p in self.graph.values()
                             if p is not None], [])))

    def parents(self, vertex):
        """Find the parents of a vertex."""
        return self.graph[vertex]

    def children(self, vertex):
        """Find the children of a vertex."""
        children = []
        for node, parents in self.graph.items():
            if parents is not None and vertex in parents:
                children.append(node)
        return children

    def descendents(self, vertex):
        """Find all descendents of a vertex."""
        descendents = []
        current_children = self.children(vertex)
        if not current_children:
            return descendents

        descendents += current_children

        for child in current_children:
            new_descendents = self.descendents(child)
            descendents += new_descendents

        return list(set(descendents))

    def get_intervened_graph(self, interventions):
        """Return the intervened graph as a new graph."""
        intervened_graph = copy.deepcopy(self.graph)
        if isinstance(interventions, collections.Iterable):
            for i in interventions:
                intervened_graph[i] = None
        else:
            intervened_graph[interventions] = None
        return Graph(intervened_graph)

    def summary(self):
        """Print a detailed summary of the graph."""
        print("Vertices in graph", self.vertices())
        print("Roots in graph", self.roots())
        print("Non-roots in graph", self.non_roots())
        print("Leafs in graph", self.leafs())
        print("Non-leafs in graph", self.non_leafs())
        print("Edges in the graph", self.edges())

        for v in self.vertices():
            print("Children of {} are {}".format(v, self.children(v)))
            print("Parents of {} are {}".format(v, self.parents(v)))
            print("Descendents of {} are {}".format(v, self.descendents(v)))

    def _convert_to_nx(self):
        """Convert the graph to a networkx `DiGraph`."""
        G = nx.DiGraph()
        for edge in self.edges():
            edge = next(iter(edge.items()))
            G.add_edge(*edge)
        return G

    def topological_sort(self):
        """Compute a topological sort of the graph."""
        G = self._convert_to_nx()
        return list(nx.topological_sort(G))

    def draw(self):
        """Draw the graph."""
        from nxpd import draw
        try:
            get_ipython
            from nxpd import nxpdParams
            nxpdParams['show'] = 'ipynb'
        except NameError:
            pass
        G = self._convert_to_nx()
        G.graph['dpi'] = 80
        return draw(G)
