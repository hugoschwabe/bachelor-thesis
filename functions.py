#Complexity
import geopandas as gpd
from matplotlib import pyplot as plt
import pandas as pd
import networkx as nx
from networkx.algorithms.community.quality import modularity
import numpy as np

def complexity(original:nx.Graph, simplified:nx.Graph):
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

    # average the three complexity scores and return result
    return 1 - ((nodes_score + edges_score + avg_degree_score) / 3)

#Structure
def structure():
    """
    Calculate and return the structure part of the scoring function
    """
    return 0

def count_regions(df:pd.DataFrame, regions:gpd.GeoDataFrame):
    
    # create dataframe from nodes["id"], nodes["x"], nodes["y"]
    df = gpd.GeoDataFrame(df["id"].copy())
    df = df.set_geometry(gpd.points_from_xy(regions["x"], regions["y"]), crs="EPSG:3035").to_crs("EPSG:4326")

    # merge nuts regions file with id-point dataframe
    df = df.sjoin(regions, how="left", predicate='within')

    # reduce merged dataframe to unique regions with number of nodes per region
    df = df.value_counts("NUTS_ID")

    # filter for rows, that contain regions with nodes
    df = df[df > 0]

    # count number of lines
    return df.size

def regionality(original:nx.Graph, simplified:nx.Graph, regions:gpd.GeoDataFrame):
    """
    Calculate and return the regionality part of the scoring function
    """
    # calculate score from difference in number of regions
    return count_regions(simplified) / count_regions(original)

#compare to pre-processed
def properties(original:nx.Graph, simplified:nx.Graph):
    """
    Calculate and return the properties part of the scoring function
    """
    # calculate properties score from 
    return properties_df(simplified) / properties_df(original)
    
def properties_df(df:pd.DataFrame):
    # count how often each unique element occurs  
    count = df["type"].value_counts().reset_index()
    # count node types, apply weights, add weighted count
    score = 0
    for i in range(0, count.shape[1]):
        match count["type"].iloc[i]:
            case "DSO": score += count["count"].iloc[i]
            case "IND": score += count["count"].iloc[i]
            case "TPP": score += count["count"].iloc[i]*0.75
            case "BIO": score += count["count"].iloc[i]*0.25
            case "GPR": score += count["count"].iloc[i]*0.25
            case "GS": score += count["count"].iloc[i]
            case "FUEL": score += count["count"].iloc[i]*0
            case "LNG": score += count["count"].iloc[i]*0.25
            case "IC": score += count["count"].iloc[i]
            case "UGS": score += count["count"].iloc[i]*0.75
            case "N": score += count["count"].iloc[i]*0.5
    return score

def score(original:nx.Graph, simplified:nx.Graph, regions:nx.Graph):
    """
    Scoring function assessing the quality of the gas network simplification 
    """
    return (3 * complexity(original, simplified) + structure() + regionality(original, simplified, regions) + properties(original, simplified)) / 6

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

    # Ergebnisse anzeigen
    print("Gefundene Communities: " + str(len(results)))

    # Modularität berechnen
    mod_value = modularity(original, results)
    print(f"Modularität Q = {mod_value}")
    return results

def build_results_graph(original:nx.Graph, results:list[frozenset]) -> tuple[nx.Graph, tuple[float, float]]:
    """
    Build the graph from simplification results list
    """
    index = []
    nodes = []
    for i in range(len(results)):
        nodes += list(results[i])
        index += [i] * len(results[i]) 
    df = pd.DataFrame({"index": index, "nodes": nodes})
    df = df.merge(graph_to_nodes_df(original), how="left", on="nodes")

    edges_converter = pd.DataFrame(np.array(original.edges), columns=["nodes_left", "nodes_right"])
    edges_converter = edges_converter.merge(df.rename(columns={"nodes":"nodes_left", "index":"index_left"}), on="nodes_left", how="left")
    edges_converter = edges_converter.merge(df.rename(columns={"nodes":"nodes_right", "index":"index_right"}), on="nodes_right", how="left")
    edges_converter = edges_converter[["index_left", "index_right"]].drop_duplicates()
    edges_converter = edges_converter[edges_converter["index_left"] != edges_converter["index_right"]]

    nodes_simpliefied = df.groupby("index").agg({"x":"mean", "y":"mean"})
    nodes_simpliefied.head(10)

    simplified_pos = {i: (nodes_simpliefied.iloc[i].x, nodes_simpliefied.iloc[i].y) for i in range(nodes_simpliefied["x"].size)}

    simplified = nx.Graph()
    simplified.add_nodes_from(list(nodes_simpliefied.index))
    simplified.add_edges_from(edges_converter.to_numpy())

    return simplified, simplified_pos

def plot_network(graph:nx.Graph, pos:dict[float, float], clusters:list=None, node_size:int=20, title:str="Title"):
    """
    Plot any NetworkX graph with correct position for nodes
    """
    plt.figure(figsize=(8, 10))

    colors = [
        'red', 'blue', 'green', 'orange', 'purple',
        'cyan', 'magenta', 'yellow', 'lime', 'pink',
        'teal', 'gold', 'navy', 'brown', 'olive'
    ]
    if clusters is None:
        clusters = []
        for elem in list(graph.nodes):
            clusters += [[elem]]

    for i, res in enumerate(clusters):
        nx.draw_networkx_nodes(graph, pos, 
                            nodelist=res, 
                            node_size=node_size, 
                            node_color=colors[i % len(colors)]
        )

    nx.draw_networkx_edges(graph, pos, alpha=1)

    plt.title(title)
    plt.show()
