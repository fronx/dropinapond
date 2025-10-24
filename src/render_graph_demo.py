# import networkx for graph generation
import networkx as nx

# import matplotlib library
import matplotlib.pyplot as plt

# generation of a sample graph
G = nx.Graph()
G.add_edges_from([('A', 'B'), ('A', 'C'), 
                  ('B', 'C'), ('E', 'F'),
                  ('D', 'E'), ('A', 'D'), 
                  ('D', 'G'), ('C', 'F'),
                  ('D', 'F'), ('E', 'H')])

# Defining ego as large and red
# while alters are in lavender
# Let 'A' be the ego
ego = 'A'
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color = "lavender", 
        node_size = 800, with_labels = True)

options = {"node_size": 1200, "node_color": "r"}
nx.draw_networkx_nodes(G, pos, nodelist=[ego], **options)
plt.show()