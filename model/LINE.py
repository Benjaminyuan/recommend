import networkx as nx
import pandas as pd
from progress.bar import Bar
node2poi_list = []
poi2node_map = {}

def get_graph(path):
    node_idx = 0
    G = nx.DiGraph()
    file_data = pd.read_csv(path, header=0, sep=',')

    bar = Bar('building graph', max=file_data.shape[0], fill='@', suffix='%(percent)d%%')
    print("\n---parsing----\n")
    for idx, row in file_data.iterrows():
        venue_id1 = row["venue_id1"]
        venue_id2 = row["venue_id2"]
        if venue_id1 not in poi2node_map:
            node2poi_list.append(venue_id1)
            poi2node_map[venue_id1] = node_idx
            node_idx += 1
        if venue_id2 not in poi2node_map:
            node2poi_list.append(venue_id2)
            poi2node_map[venue_id2] = node_idx
            node_idx +=1
        G.add_edges_from([(poi2node_map[venue_id1], poi2node_map[venue_id2],{"weight":row["weight"]}),
            (poi2node_map[venue_id2], poi2node_map[venue_id1],{"weight":row["weight"]})])
        bar.next()
    bar.finish()
    print("graph node size: {}".format(len(list(G.nodes))))
    print("graph edge size: {}".format(len(list(G.edges))))
