import networkx as nx
import matplotlib.pyplot as plt

# Define the structures and their edges
structures = {
    "chain_structure": [("A", "B"), ("B", "C")],
    "complex_structure": [("A", "B"), ("B", "C"), ("A", "C")],
    "ritesh_s_structure": [
        ("A", "D"),
        ("A", "F"),
        ("B", "D"),
        ("C", "D"),
        ("C", "E"),
        ("E", "G"),
        ("D", "G"),
        ("D", "F"),
        ("F", "H"),
        ("G", "H"),
        ("H", "I"),
        ("F", "I"),
    ],
    "vusani_s_structure_1": [
        ("A", "E"),
        ("B", "E"),
        ("C", "E"),
        ("D", "E"),
        ("E", "F"),
        ("E", "G"),
        ("E", "H"),
        ("F", "H"),
        ("G", "H"),
    ],
    "vusani_s_structure_2": [
        ("A", "F"),
        ("B", "F"),
        ("C", "F"),
        ("D", "F"),
        ("F", "G"),
        ("F", "H"),
        ("F", "I"),
        ("F", "J"),
        ("G", "J"),
        ("H", "J"),
        ("I", "J"),
    ],
    "complex_20_node_structure": [
        ('A', 'D'),
        ('A', 'E'),
        ('A', 'F'),
        ('B', 'D'),
        ('B', 'G'),
        ('B', 'H'),
        ('C', 'E'),
        ('C', 'G'),
        ('C', 'I'),
        ('D', 'J'),
        ('D', 'K'),
        ('E', 'J'),
        ('E', 'L'),
        ('F', 'K'),
        ('F', 'M'),
        ('G', 'L'),
        ('G', 'N'),
        ('H', 'M'),
        ('H', 'O'),
        ('I', 'N'),
        ('I', 'P'),
        ('J', 'Q'),
        ('K', 'Q'),
        ('L', 'R'),
        ('M', 'R'),
        ('N', 'S'),
        ('O', 'S'),
        ('P', 'T'),
        ('Q', 'T'),
        ('R', 'T'),
        ('S', 'T'),
    ],
}

def plot_structure(edges, title):
    # Initialize directed graph
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    # Set up plot size and layout
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # Use spring layout for visually appealing spacing
    
    # Draw the graph with labels
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=15)
    plt.title(title)
    plt.show()

# Plot each structure
for name, edges in structures.items():
    plot_structure(edges, title=name)
