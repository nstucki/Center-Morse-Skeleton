import numpy as np
import networkx as nx
from itertools import product
import pyvista as pv

def get_neighbors_26(x, y, z):
    neighbors = [(x+i, y+j, z+k) for i, j, k in product([-1, 0, 1], repeat=3) if (i, j, k) != (0, 0, 0)]
    return neighbors

def create_skeleton_graph(skeleton):
    graph = nx.Graph()
    shape = skeleton.shape

    # Add vertices to the graph
    for x, y, z in product(range(shape[0]), range(shape[1]), range(shape[2])):
        if skeleton[x, y, z] == 1:
            graph.add_node((x, y, z))

    # Add edges based on 26-connectivity
    for x, y, z in graph.nodes():
        neighbors = get_neighbors_26(x, y, z)
        for neighbor in neighbors:
            if neighbor in graph.nodes():
                graph.add_edge((x, y, z), neighbor)

    return graph

def get_neighbors_18(x, y, z):
    neighbors = [
        (x + i, y + j, z + k)
        for i, j, k in product([-1, 0, 1], repeat=3)
        if (i, j, k) != (0, 0, 0) and abs(i) + abs(j) + abs(k) <= 2
    ]
    return neighbors

def create_skeleton_graph_18(skeleton):
    graph = nx.Graph()
    shape = skeleton.shape

    # Add vertices to the graph
    for x, y, z in product(range(shape[0]), range(shape[1]), range(shape[2])):
        if skeleton[x, y, z] == 1:
            graph.add_node((x, y, z))

    # Add edges based on 18-connectivity
    for x, y, z in graph.nodes():
        neighbors = get_neighbors_18(x, y, z)
        for neighbor in neighbors:
            if neighbor in graph.nodes():
                graph.add_edge((x, y, z), neighbor)

    return graph
    
def save_skeleton_graph(skeleton_graph, save_path):
    # Now you can access the vertices and edges of the skeleton graph
    vertices = list(skeleton_graph.nodes())
    edges = list(skeleton_graph.edges())
    vertices = np.array(vertices)
    dict = [{tuple(coord): idx} for idx, coord in enumerate(vertices)]
    # merge a list of dictionaries into a single dictionary not two dictionary
    merged_dict = {k: v for d in dict for k, v in d.items()}
    for i in range(len(edges)):
        edges[i] = (merged_dict[edges[i][0]], merged_dict[edges[i][1]])
    edges = np.array(edges)
    patch_edge = np.concatenate((np.int32(2 * np.ones((edges.shape[0], 1))), edges), 1)
    # mesh = pyvista.PolyData(patch_coord)
    # print(patch_edge.shape)
    # mesh.lines = patch_edge.flatten()
    mesh = pv.UnstructuredGrid(patch_edge.flatten(), np.array([4] * len(edges)), vertices)
    mesh_structured = mesh.extract_surface()
    # mesh.save(save_path + 'vtp/sample_' + str(idx).zfill(6) + '_graph.vtp')
    mesh_structured.save(save_path)