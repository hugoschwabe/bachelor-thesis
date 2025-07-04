#Complexity
import geopandas as gpd
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx
from networkx.algorithms.community.quality import modularity
import numpy as np
from scipy.stats import wasserstein_distance

# What happens when avg_deg of simplified is higher than original?
def complexity(original:nx.Graph, simplified:nx.Graph, verbose:bool=False) -> float:
    """
    Calculate and return the complexity part of the scoring function
    """
    # calculate node count score
    nodes_score = 1 - (simplified.number_of_nodes() / original.number_of_nodes())
    
    # calculate edge count score
    edges_score = 1 - (simplified.number_of_edges() / original.number_of_edges())
    
    # calculate cyclomatic number
    pre_cyclo = original.number_of_edges() - original.number_of_nodes() + nx.number_connected_components(original)
    post_cyclo = simplified.number_of_edges() - simplified.number_of_nodes() + nx.number_connected_components(simplified)
    cyclo_score = 1 - (post_cyclo / pre_cyclo)

    # calculating score
    complexity_score = (nodes_score + edges_score + cyclo_score) / 3

    if verbose:
        print(f"nodes_score: {nodes_score}")
        print(f"edges_score: {edges_score}")
        print(f"pre_cyclo: {pre_cyclo}")
        print(f"post_cyclo: {post_cyclo}")
        print(f"cyclomatic_score: {cyclo_score}")
        print(f"complexity_score: {complexity_score}\n")

    # average the three complexity scores and return result
    return complexity_score

def structure(original:nx.Graph, simplified:nx.Graph, verbose:bool=False) -> float:
    """
    Calculate and return the structure part of the scoring function
    """
    # Algebraic Connectivity: The second-smallest eigenvalue
    ac_simplified = nx.linalg.algebraic_connectivity(simplified)
    ac_original = nx.linalg.algebraic_connectivity(original)
    # Symmetric Difference for score calculation, taking into account increases and decreases after simplification
    ac_score = 1 - (abs(ac_simplified - ac_original) / ac_simplified + ac_original)

    # Distribution of Betweenness Centrality: Earth Mover's Distance (EMD)
    centrality_before = nx.betweenness_centrality(original)
    centrality_after_raw = nx.betweenness_centrality(simplified)
    # centrality dictionary for the 'after' case, including nodes that were removed (their centrality is now 0).
    centrality_after = {node: centrality_after_raw.get(node, 0.0) for node in original.nodes()}
    dist_before = np.array([centrality_before[node] for node in original.nodes()])
    print(dist_before)
    dist_after = np.array([centrality_after[node] for node in original.nodes()])
    print(dist_after)
    emd_score = wasserstein_distance(dist_before, dist_after)

    # Spectral Distance: Capturing global structural changes
    nodelist = list(original.nodes())
    # Get the Laplacian matrices as dense NumPy arrays
    laplacian_before = nx.laplacian_matrix(original, nodelist=nodelist).toarray()
    laplacian_after = np.zeros_like(laplacian_before)
    nodes_in_after = [node for node in nodelist if node in simplified]
    if nodes_in_after:
        # Calculate the smaller Laplacian for only the nodes that still exist
        sub_laplacian = nx.laplacian_matrix(simplified, nodelist=nodes_in_after).toarray()
        # Create a mapping from node name to its index in the original full nodelist
        node_to_idx = {node: i for i, node in enumerate(nodelist)}
        # Get the indices where the sub-matrix should be placed
        after_indices = [node_to_idx[node] for node in nodes_in_after]
        # Use NumPy's advanced indexing to place the smaller sub_laplacian
        # into the correct locations within the full-sized zero matrix.
        laplacian_after[np.ix_(after_indices, after_indices)] = sub_laplacian
    # Calculate the raw spectral distance (Frobenius norm of the difference)
    dist_raw = np.linalg.norm(laplacian_before - laplacian_after, 'fro')
    print(dist_raw)
    # Calculate the maximum possible distance (norm of the original Laplacian)
    dist_max = np.linalg.norm(laplacian_before, 'fro')
    print(dist_max)
    # Normalize the score
    spectral_score = 1.0 - (dist_raw / dist_max)

    # calculating score
    structure_score = (ac_score + emd_score + spectral_score) / 3

    if verbose:
        print(f"ac_simplified: {ac_simplified}")
        print(f"ac_original: {ac_original}")
        print(f"ac_score: {ac_score}")
        print(f"emd_score: {emd_score}")
        print(f"spectral_dist_score: {spectral_score}")
        print(f"structure_score: {structure_score}\n")

    return structure_score

def count_regions(G:nx.Graph, regions:gpd.GeoDataFrame) -> int:
    # create geodataframe from nodes
    df = gpd.GeoDataFrame(graph_to_nodes_df(G))
    df = df.set_geometry(gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:3035").to_crs("EPSG:4326")

    # merge nuts regions file with id-point dataframe
    df = df.sjoin(regions, how="left", predicate='within')

    # reduce merged dataframe to unique regions with number of nodes per region
    df = df.value_counts("NUTS_ID")

    # filter for rows, that contain regions with nodes
    df = df[df > 0]

    # count number of lines
    return df.size

def regionality(original:nx.Graph, simplified:nx.Graph, regions:gpd.GeoDataFrame, verbose:bool=False) -> float:
    """
    Calculate and return the regionality part of the scoring function
    """
    original_regions = count_regions(original, regions)
    simplified_regions = count_regions(simplified, regions)
    regionality_score = simplified_regions / original_regions

    if verbose:
        print(f"original_regions: {original_regions}")
        print(f"simplified_regions: {simplified_regions}")
        print(f"regionality_score: {regionality_score}\n")

    return regionality_score

def properties(original:nx.Graph, simplified:nx.Graph, verbose:bool=False) -> float:
    """
    Calculate and return the properties part of the scoring function
    """
    original_properties = properties_df(graph_node_names(original))
    simplified_properties = properties_df(graph_node_names(simplified))
    properties_score = simplified_properties / original_properties

    if verbose:
        print(f"original_properties: {original_properties}")
        print(f"simplified_properties: {simplified_properties}")
        print(f"properties_score: {properties_score}")
    
    return properties_score
    
def properties_df(l:list) -> float:
    score = 0
    for elem in l:
        if "DSO" in elem: score += 1
        if "IND" in elem: score += 1
        if "TPP" in elem: score += 0.75
        if "BIO" in elem: score += 0.25
        if "GPR" in elem: score += 0.25
        if "CS" in elem: score += 1
        if "CV" in elem: score += 0
        if "LNG" in elem: score += 0.25
        if "IC" in elem: score += 1
        if "ST" in elem: score += 0.75
        if "X" in elem: score += 0.5
    return score

def score(original:nx.Graph, simplified:nx.Graph, regions:nx.Graph, verbose:bool=False) -> float:
    """
    Scoring function assessing the quality of the gas network simplification 
    """
    return (complexity(original, simplified, verbose) + structure(original, simplified, verbose) + regionality(original, simplified, regions, verbose) + properties(original, simplified, verbose)) / 4

def graph_node_names(G:nx.Graph) -> list[str]:
    nodes = graph_to_nodes_df(G)["nodes"]
    if type(nodes[0]) == str:
        return nodes.to_list()
    else:	
        return [item for sublist in nodes for item in sublist]

def graph_to_nodes_df(G:nx.Graph) -> pd.DataFrame:
    """
    Converts graph's nodes into a dataframe while preserving all attributes
    """
    return pd.DataFrame(dict(G.nodes(data=True))).T.reset_index().rename(columns={"index": "nodes"})

def run_algo(original:nx.Graph, func) -> list[frozenset]:
    """
    Run a NetworkX algorithm and return the results
    """
    #results = greedy_modularity_communities(graph)
    results = func(original)

    # Show results
    print("Gefundene Communities: " + str(len(results)))

    # Calculate modularity
    mod_value = modularity(original, results)
    print(f"Modularität Q = {mod_value}")
    return results

# Mistake: Directions are not taken into account
def build_results_graph(original:nx.Graph, results:list[frozenset]) -> tuple[nx.Graph, tuple[float, float]]:
    """
    Build the graph from simplification results list
    """
    nodes = []
    group = []
    for i in range(len(results)):
        nodes += list(results[i])
        group += [i] * len(results[i]) 
    df = pd.DataFrame({"group": group, "nodes": nodes})
    df = df.merge(graph_to_nodes_df(original), how="left", on="nodes")
    df["x"] = df["coord"].map(lambda x: x[0])
    df["y"] = df["coord"].map(lambda x: x[1])
    #print(df.head(10))

    edges_converter = pd.DataFrame(np.array(original.edges), columns=["nodes_left", "nodes_right"])
    edges_converter = edges_converter.merge(df.rename(columns={"nodes":"nodes_left", "group":"index_left"}), on="nodes_left", how="left")
    edges_converter = edges_converter.merge(df.rename(columns={"nodes":"nodes_right", "group":"index_right"}), on="nodes_right", how="left")
    edges_converter = edges_converter[["index_left", "index_right"]].drop_duplicates()
    edges_converter = edges_converter[edges_converter["index_left"] != edges_converter["index_right"]]
    edges_converter = list(edges_converter[["index_left", "index_right"]].itertuples(index=False, name=None))
    #print(edges_converter)
    
    nodes_simpliefied = df.groupby("group")
    nodes_simpliefied = nodes_simpliefied.agg({"x":"mean", "y":"mean", "nodes": list, "node_type": set})
    nodes_simpliefied = nodes_simpliefied.rename(columns={"nodes":"index"})
    #print(nodes_simpliefied)

    simplified = nx.Graph()
    simplified.add_nodes_from(nodes_simpliefied.T.to_dict().items())
    simplified.add_edges_from(list(edges_converter))

    return simplified

def plot_network(graph:nx.Graph, gdf:gpd.GeoDataFrame=None, clusters:list=None, node_size:int=20, title:str="Title", padding_ratio:float=0.15, nodes:bool=True, edges:bool=True) -> None:
    """
    Plot any NetworkX graph with optional NUTS regions background from a GeoDataFrame.
    """
    fig, ax = plt.subplots(figsize=(8, 10))  # Use explicit axis

    # Plot GeoDataFrame (e.g. NUTS background)
    if gdf is not None:
        gdf.plot(ax=ax, color='lightgrey', edgecolor='white', alpha=0.5, linewidth=2, zorder=0)

    # Get node positions
    try:
        pos = {node: (data["coord"][0], data["coord"][1]) for node, data in graph.nodes(data=True)}
    except:
        pos = {node: (data["x"], data["y"]) for node, data in graph.nodes(data=True)}

    if nodes:
        # Color palette
        colors = [
            'red', 'blue', 'green', 'orange', 'purple',
            'cyan', 'magenta', 'yellow', 'lime', 'pink',
            'teal', 'gold', 'navy', 'brown', 'olive'
        ]

        # Handle clusters
        if clusters is None:
            clusters = np.array(graph.nodes).reshape(-1, 1)
        
        # Draw nodes
        
        for i, res in enumerate(clusters):
            nx.draw_networkx_nodes(
                graph, pos,
                nodelist=res,
                node_size=node_size,
                node_color=colors[i % len(colors)],
                ax=ax
            )

    if edges:
        # Draw edges
        nx.draw_networkx_edges(graph, pos, alpha=1, arrows=False, ax=ax)
    
    # Set axis extent from node coordinates with padding
    x_vals = [coord[0] for coord in pos.values()]
    y_vals = [coord[1] for coord in pos.values()]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    x_pad = (x_max - x_min) * padding_ratio
    y_pad = (y_max - y_min) * padding_ratio

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_title(title)
    ax.set_axis_on()  
    plt.tight_layout()
    plt.show()