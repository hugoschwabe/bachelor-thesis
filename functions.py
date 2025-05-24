#Complexity
import geopandas as gpd
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx
from networkx.algorithms.community.quality import modularity
import numpy as np

# What happens when avg_deg of simplified is higher than original?
def complexity(original:nx.Graph, simplified:nx.Graph, verbose:bool=False):
    """
    Calculate and return the complexity part of the scoring function
    """
    # calculate node count score
    nodes_score = simplified.number_of_nodes() / original.number_of_nodes()
    
    # calculate edge count score
    edges_score = simplified.number_of_edges() / original.number_of_edges()
    
    # calculate average degree score
    original_avg_degree = sum([x[1] for x in nx.degree(original)]) / len(nx.degree(original))
    simplified_avg_degree = sum([x[1] for x in nx.degree(simplified)]) / len(nx.degree(simplified))
    avg_degree_score = simplified_avg_degree / original_avg_degree

    if verbose:
        print(f"nodes_score: {nodes_score}")
        print(f"edges_score: {edges_score}")
        print(f"avg_degree_score: {avg_degree_score}")

    # average the three complexity scores and return result
    return 1 - ((nodes_score + edges_score + avg_degree_score) / 3)

#Structure
def structure():
    """
    Calculate and return the structure part of the scoring function
    """
    return 0

def count_regions(G:nx.Graph, regions:gpd.GeoDataFrame):
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

def regionality(original:nx.Graph, simplified:nx.Graph, regions:gpd.GeoDataFrame, verbose:bool=False):
    """
    Calculate and return the regionality part of the scoring function
    """
    if verbose:
        print(f"original regions: {count_regions(original, regions)}")
        print(f"simplified regions: {count_regions(simplified, regions)}")

    # calculate score from difference in number of regions
    return count_regions(simplified, regions) / count_regions(original, regions)

def properties(original:nx.Graph, simplified:nx.Graph, verbose:bool=False):
    """
    Calculate and return the properties part of the scoring function
    """
    if verbose:
        print(f"original properties: {properties_df(graph_node_names(original))}")
        print(f"simplified properties: {properties_df(graph_node_names(simplified))}")
    return properties_df(graph_node_names(simplified)) / properties_df(graph_node_names(original))
    
def properties_df(l:list):
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

def score(original:nx.Graph, simplified:nx.Graph, regions:nx.Graph, verbose:bool=False):
    """
    Scoring function assessing the quality of the gas network simplification 
    """
    if verbose:
        print(f"complexity: {complexity(original, simplified, verbose=True)}\n")
        print(f"structure: {structure()}\n")
        print(f"regionality: {regionality(original, simplified, regions, verbose=True)}\n")
        print(f"properties: {properties(original, simplified, verbose=True)}\n")
    return (complexity(original, simplified) + structure() + regionality(original, simplified, regions) + properties(original, simplified)) / 4

def graph_node_names(G:nx.Graph):
    nodes = graph_to_nodes_df(G)["nodes"]
    if type(nodes[0]) == str:
        return nodes.to_list()
    else:	
        return [item for sublist in nodes for item in sublist]

def graph_to_nodes_df(G:nx.Graph):
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

def plot_network(graph:nx.Graph, gdf:gpd.GeoDataFrame=None, clusters:list=None, node_size:int=20, title:str="Title", padding_ratio:float=0.15, nodes:bool=True, edges:bool=True):
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