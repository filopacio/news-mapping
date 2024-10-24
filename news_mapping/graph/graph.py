import networkx as nx
import matplotlib.pyplot as plt
import json


class ArticleGraph:
    def __init__(self, dataframe, config_file):
        self.G = nx.Graph()
        self.dataframe = dataframe
        self.config_file = config_file
        self.relationships = self._load_config()
        self.node_types = self._extract_node_types()
        self.node_frequencies = (
            self._calculate_node_frequencies()
        )  # Calculate frequencies of nodes
        self._build_graph()

    def _load_config(self):
        with open(self.config_file, "r") as file:
            config = json.load(file)
        return config["relationships"]

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
        """Build the graph based on the dataframe and relationships from the config."""
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

    def plot_graph(self):
        """Plot the graph with node sizes based on frequency of appearance."""
        plt.figure(figsize=(14, 14))

        # Position the nodes using the spring layout
        pos = nx.spring_layout(self.G, seed=42)

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
            node: 300 + (frequency / max_frequency) * 2000
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
                node_sizes.get(node, 300) for node in nodelist
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
            self.G, pos, font_size=12, font_color="black", font_weight="bold"
        )

        # Draw edge labels (relationships)
        edge_labels = {(u, v): d["relationship"] for u, v, d in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            self.G,
            pos,
            edge_labels=edge_labels,
            font_color="red",
            font_size=10,
            font_weight="bold",
        )

        # Show the plot
        plt.title("Newspaper Articles Graph", fontsize=20)
        plt.axis("off")
        plt.show()


# class ArticleGraph:
#     def __init__(self, dataframe, config_file):
#         self.G = nx.Graph()
#         self.dataframe = dataframe
#         self.config_file = config_file
#         self.relationships = self._load_config()
#         self.node_types = self._extract_node_types()
#         self._build_graph()
#
#     def _load_config(self):
#         with open(self.config_file, 'r') as file:
#             config = json.load(file)
#         return config['relationships']
#
#     def _extract_node_types(self):
#         node_types = set()
#         for relation in self.relationships:
#             node_types.add(relation['source'])
#             node_types.add(relation['target'])
#         return node_types
#
#     def _build_graph(self):
#         for _, row in self.dataframe.iterrows():
#             for relation in self.relationships:
#                 source = row[relation['source']]
#                 target = row[relation['target']]
#                 relationship = relation['relationship']
#
#                 # Add nodes
#                 self.G.add_node(source, type=relation['source'])
#                 self.G.add_node(target, type=relation['target'])
#
#                 # Add edges
#                 self.G.add_edge(source, target, relationship=relationship)
#
#     def plot_graph(self):
#         plt.figure(figsize=(14, 14))
#
#         # Position the nodes using the spring layout
#         pos = nx.spring_layout(self.G, seed=42)
#
#         # Define default colors and shapes
#         default_color = 'skyblue'
#         default_shape = 'o'
#
#         # Generate a unique color and shape for each node type
#         unique_colors = plt.cm.get_cmap('tab20', len(self.node_types))
#         node_colors = {node_type: unique_colors(i) for i, node_type in enumerate(self.node_types)}
#         node_shapes = {node_type: default_shape for node_type in self.node_types}
#
#         # Draw nodes with different colors and shapes for different types
#         for node_type in self.node_types:
#             nodelist = [node for node, attr in self.G.nodes(data=True) if attr['type'] == node_type]
#             color = node_colors.get(node_type, default_color)
#             shape = node_shapes.get(node_type, default_shape)
#             nx.draw_networkx_nodes(
#                 self.G, pos, nodelist=nodelist, node_color=color,
#                 node_shape=shape, node_size=500, alpha=0.9
#             )
#
#         # Draw the edges
#         nx.draw_networkx_edges(self.G, pos, width=2, alpha=0.6, edge_color='gray', style='dashed')
#
#         # Draw the labels
#         nx.draw_networkx_labels(self.G, pos, font_size=12, font_color='black', font_weight='bold')
#
#         # Draw edge labels (relationships)
#         edge_labels = {(u, v): d['relationship'] for u, v, d in self.G.edges(data=True)}
#         nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_color='red', font_size=10, font_weight='bold')
#
#         # Show the plot
#         plt.title("Newspaper Articles Graph", fontsize=20)
#         plt.axis('off')
#         plt.show()
