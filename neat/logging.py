import os
import json
import networkx as nx
import matplotlib.pyplot as plt

from neat import Individual


def load_population(output_dir: str, max_generation: int):
    """
    Load all generations of populations and build mappings:
      - id2ind: individual id -> Individual instance
      - parent2children: parent id -> list of child ids
    """
    id2ind = {}
    parent2children = {}

    for gen in range(max_generation + 1):
        path = os.path.join(output_dir, f"population_{gen}.json")
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            raw_list = json.load(f)
        for raw in raw_list:
            # reconstruct Individual
            ind = Individual.from_json(json.dumps(raw))
            id2ind[ind.id] = ind
            for pid in raw.get('parents_id', []):
                parent2children.setdefault(pid, []).append(ind.id)

    return id2ind, parent2children


def trace_back(target_id: int,
               id2ind: dict,
               parent2children: dict) -> dict:
    """
    Build a nested lineage dict for target_id:
      {
        'individual': Individual,
        'children': [ same structure... ]
      }
    """
    if target_id not in id2ind:
        raise ValueError(f"Individual id={target_id} not found.")

    def build_node(cur_id):
        node = {
            'individual': id2ind[cur_id],
            'children': []
        }
        for cid in parent2children.get(cur_id, []):
            node['children'].append(build_node(cid))
        return node

    return build_node(target_id)


def visualize_network(ind: Individual, show_weights: bool = True):
    """
    Draw a single neural network structure using networkx.
    """
    G = nx.DiGraph()
    # add nodes
    for nid, ntype in zip(ind.nodes[0, :].tolist(), ind.nodes[1, :].tolist()):
        G.add_node(int(nid), type=int(ntype))

    # add edges
    mask = ind.genes[4, :] == 1
    src = ind.genes[1, mask].tolist()
    dst = ind.genes[2, mask].tolist()
    wts = ind.genes[3, mask].tolist()
    for s, d, w in zip(src, dst, wts):
        G.add_edge(int(s), int(d), weight=float(w))

    # layout & draw
    pos = nx.spring_layout(G)
    color_map = {1: 'skyblue', 2: 'lightgreen', 3: 'orange', 4: 'grey'}
    node_colors = [color_map[G.nodes[n]['type']] for n in G.nodes]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=500)
    if show_weights:
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()


def visualize_lineage_tree(root_node: dict):
    """
    Draw the lineage tree from a nested lineage dict:
      {
        'individual': Individual,
        'children': [ ... ]
      }
    """
    G = nx.DiGraph()

    def add_edges(node):
        pid = node['individual'].id
        for child in node.get('children', []):
            cid = child['individual'].id
            G.add_edge(pid, cid)
            add_edges(child)

    add_edges(root_node)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=300, arrowsize=12)
    plt.show()
