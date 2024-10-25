import networkx as nx
import matplotlib.pyplot as plt
import json

class ArticleGraph:
    def __init__(self, dataframe, relationships):
        self.G = nx.Graph()
        self.dataframe = dataframe
        self.relationships = relationships  # Directly accept the relationships as a dict or JSON
        self.node_types = self._extract_node_types()
        self.node_frequencies = self._calculate_node_frequencies()  # Calculate frequencies of nodes
        self._build_graph()

    def _extract_node_types(self):
        node_types = set()
        for relation in self.relationships:
            node_types.add(relation["source"])
            node_types.add(relation["target"])
        return node_types

    def _calculate_node_frequencies(self):
        """Calculate the frequency of each node based on its appearance in the dataframe."""
        node_frequencies = {}

        # Loop through the dataframe and count occurrences of both source and target nodes
        for _, row in self.dataframe.iterrows():
            for relation in self.relationships:
                source = row[relation["source"]]
                target = row[relation["target"]]

                # Increment frequency for source
                if source in node_frequencies:
                    node_frequencies[source] += 1
                else:
                    node_frequencies[source] = 1

                # Increment frequency for target
                if target in node_frequencies:
                    node_frequencies[target] += 1
                else:
                    node_frequencies[target] = 1

        return node_frequencies

    def _build_graph(self):
        """
        Build the graph based on the dataframe and relationships.
        """
        for _, row in self.dataframe.iterrows():
            for relation in self.relationships:
                source = row[relation["source"]]
                target = row[relation["target"]]
                relationship = relation["relationship"]

                # Add nodes with types
                self.G.add_node(source, type=relation["source"])
                self.G.add_node(target, type=relation["target"])

                # Add edges
                self.G.add_edge(source, target, relationship=relationship)

    def plot_graph(self,
                   title:str = None,
                   layout: str = "random_layout",
                   figsize: tuple = (14,14),
                   show_axis: str = "on"):
        """
                Plot the graph with node sizes based on frequency of appearance.

        :param title:
        :param layout: ["random_layout", "spring_layout", "layered"]
        :param figsize: tuple
        :param show_axis: ["off", "on"]
        :return:
        """
        plt.figure(figsize=figsize)
        plt.grid()

        # Position the nodes according to specified layout
        pos = nx.random_layout(self.G)

        if layout == "layered":
            y = {}
            for i, node_type in enumerate(list(self.node_types)):
                y[node_type] = (i + 1) * 1.5

            for i, node in dict(self.G.nodes).items():
                pos[i] += [0,y[node["type"]]]
        elif layout == "spring_layered":
            pos = nx.spring_layout(self.G)


        # Define default colors and shapes
        default_color = "skyblue"
        default_shape = "o"

        # Generate a unique color and shape for each node type
        unique_colors = plt.cm.get_cmap("tab20", len(self.node_types))
        node_colors = {
            node_type: unique_colors(i) for i, node_type in enumerate(self.node_types)
        }
        node_shapes = {node_type: default_shape for node_type in self.node_types}

        # Calculate node sizes based on frequency (number of occurrences)
        max_frequency = (
            max(self.node_frequencies.values()) if self.node_frequencies else 1
        )

        # Scale node sizes, ensuring a minimum size of 300
        node_sizes = {
            node: 200 + (frequency / max_frequency) * 2000
            for node, frequency in self.node_frequencies.items()
        }

        # Draw nodes with different colors and shapes for different types
        for node_type in self.node_types:
            nodelist = [
                node
                for node, attr in self.G.nodes(data=True)
                if attr["type"] == node_type
            ]
            color = node_colors.get(node_type, default_color)
            shape = node_shapes.get(node_type, default_shape)

            # Get the sizes for the corresponding nodes
            sizes = [
                node_sizes.get(node, 200) for node in nodelist
            ]  # Default to 300 if no size

            nx.draw_networkx_nodes(
                self.G,
                pos,
                nodelist=nodelist,
                node_color=color,
                node_shape=shape,
                node_size=sizes,
                alpha=0.9,
            )

        # Draw the edges
        nx.draw_networkx_edges(
            self.G, pos, width=2, alpha=0.6, edge_color="gray", style="dashed"
        )

        # Draw the labels
        nx.draw_networkx_labels(
            self.G, pos, font_size=6, font_color="black", font_weight="bold"
        )

        # Draw edge labels (relationships)
        edge_labels = {(u, v): d["relationship"] for u, v, d in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            self.G,
            pos,
            edge_labels=edge_labels,
            font_color="red",
            font_size=6,
            font_weight="bold",
        )

        # Show the plot
        graph_title = "Newspaper Articles Graph" if not title else title
        plt.title(graph_title, fontsize=20)
        plt.axis(show_axis)
        plt.show()