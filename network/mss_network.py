import numpy as np
import pymysql.cursors
import jenkspy as jnb
import networkx as nx
from itertools import count
import matplotlib as mpl
import matplotlib.pyplot as plt



class dictBT:

    def __init__(self, id, data):
        self._id = id
        self._data = data
        self.lst = None
        self.rst = None
    
    def id(self):
        return self._id
    
    def data(self):
        return self._data
    
    #def lst(self):
    #    return self._lst
    
    #def rst(self):
    #    return self._rst
    
    def insert_left(self, id, data):
        if self.lst == None:
            self.lst = dictBT(id, data)
        else:
            new_node = dictBT(id, data)
            new_node.lst = self.lst
            self.lst = new_node
    
    def insert_right(self, id, data):
        if self.rst == None:
            self.rst = dictBT(id, data)
        else:
            new_node = dictBT(id, data)
            new_node.rst = self.rst
            self.rst = new_node
    
    def print_node(self):
        if self:
            print('id:', self.id(), 'data:', self.data())




def nb_nodes(t):
    '''
    dictBT -> int
    Returns the number of nodes of t
    '''
    if not t:
        return 0
    if not t.lst:
        if not t.rst:
            return 1
        else:
            return 1 + nb_nodes(t.rst)
    else:
        if not t.rst:
            return 1 + nb_nodes(t.lst)
        else:
            return 1 + nb_nodes(t.lst) + nb_nodes(t.rst)


def add_node_dictBT(t, id, data):
    '''
    dictBT * int * Dict[str, T]
    Returns the dictBT formed with the addition of
    node (id, data) to t
    '''
    if not t:
        return dictBT(id, data)
    if nb_nodes(t.lst) <= nb_nodes(t.rst):
        t.lst = add_node_dictBT(t.lst, id, data)
    else:
        t.rst = add_node_dictBT(t.rst, id, data)
    return t


def print_BT(t):
    '''
    dictBT -> None
    Print the content of a dictAB
    '''
    if t:
        t.print_node()
        print_BT(t.lst)
        print_BT(t.rst)


def connexion_bd():
    '''
    None -> pypmysql.cursor
    Returns the cursors object for the connection
    to the relationnal data base
    '''
    myUser = 'user'
    myPassword = 'pw'
    Host = 'host'
    database = 'data_bnf'

    connection = pymysql.connect(host=Host,
                                 user=myUser,
                                 password=myPassword,
                                 db=database,
                                 cursorclass=pymysql.cursors.DictCursor)

    return connection.cursor()


def constuct_dictBT(query):
    '''
    str -> dictBT
    Returns a dictAB given a SQL query
    '''
    cursor = connexion_bd()
    cursor.execute(query)
    id = 0
    dict_tree = None
    for row in cursor:
        dico_temp = row
        dict_tree = add_node_dictBT(dict_tree, id, dico_temp)
        id += 1
    return dict_tree


def comp_dict(data1, data2):
    '''
    Dict[str, T] * Dict[str, T] -> int
    Returns the number of equal modality for each
    variable of data1 and data2
    '''
    cpt = 0
    for (k1, v1), (k2, v2) in zip(data1.items(), data2.items()):
        if k1 == k2:
            if v1 == v2:
                cpt += 1
    return cpt


def fusion_dict(data1, data2):
    '''
    Dict[str, T] * Dict[str, T] -> Dict[str, T]
    Returns the fusion of data1 and data2
    '''
    if data1 == dict():
        if data2 == dict():
            return dict()
        else:
            return data2
    if data2 == dict():
        return data1
    res = data1
    for k, v in data2.items():
        if k not in res:
            res[k] = v
    return res


def variable_series(t, var):
    '''
    dictBT * str -> List[float]
    Returns a list of float given a variable var
    and a binary tree dictionnary t
    '''
    if not t:
        return []
    return [t.data()[var]] + variable_series(t.lst, var) + variable_series(t.rst, var)


def discretization(t, var, classes):
    '''
    dictBT * str * List[str] -> None
    Modifies the modalities of a variable var in
    a binary tree dictionnary t given a list of
    discrete classes
    '''
    if t:
        i = 1
        while t.data()[var] > classes[i]:
            i += 1
        t.data()[var] = '[' + str(classes[i-1]) + ',' + str(classes[i]) + ']'
        discretization(t.lst, var, classes)
        discretization(t.rst, var, classes)



def jenks_natural_breaks(t, var, classes_nb):
    '''
    dictBT * str * int -> None
    Processes to the Jenks natural breaks
    discretization given a binary tree
    dictionnary t a variable var and a
    number of classes whished
    '''
    values_list = sorted(variable_series(t, var))
    classes = jnb.jenks_breaks(values_list, n_classes=classes_nb)
    discretization(t, var, classes)


def jnb_for_variables_list(t, var_list, classes_nb):
    '''
    dictBT * List[str] * int -> None
    Processes to JNB given a binary tree dictionnary t
    a list of variable and a numbre of classes
    '''
    for var in var_list:
        jenks_natural_breaks(t, var, classes_nb)


def construct_edges(id, data, t):
    '''
    dictBT -> Dict[Tuple[int, int], int]
    Returns the dictionnary with all the edges and its
    valuation given id and data
    '''
    if not t or id == t.id():
        return dict()
    data_edges = fusion_dict(construct_edges(id, data, t.lst),
                             construct_edges(id, data, t.rst))
    if (t.id(), id) not in data_edges:
        if data_edges == dict():
            data_edges = {(id, t.id()): comp_dict(data, t.data())}
        else:
            data_edges[(id, t.id())] = comp_dict(data, t.data())
    return data_edges


def construct_all_edges(t, t_ref):
    '''
    dictBT -> Dict[Tuple[int, int], int]
    Returns the dictionnary of all the edges
    given t and t_ref
    '''
    if not t:
        return dict()
    return fusion_dict(construct_edges(t.id(), t.data(), t_ref),
                       fusion_dict(construct_all_edges(t.lst, t_ref),
                                   construct_all_edges(t.rst, t_ref)))


def decile(L):
    '''
    List[float] -> List[float]
    Precondition : len(L) >= 10
    Returns the list of the decile of L
    '''
    LD = [L[0]]
    for i in range(1, 11):
        LD.append(L[int(((len(L)-1)*i)/10)])
    L[len(L) - 1]
    return LD


def filter_main_edges(edges):
    '''
    Dict[Tuple[int, int], int] -> Dict[Tuple[int, int], int]
    Returns a dictionnary of edges associated to their weight
    '''
    edges_dec = decile(sorted([wgt for _, wgt in edges.items()]))
    return {edge: wgt for edge, wgt in edges.items() if wgt > edges_dec[-3]}


def convert_dictAB(t):
    '''
    dictBT -> List[Tuple[int, Dict[str, T]]]
    Retruns the convertion of a dictAB in a List
    of Tuple for the construction of the network
    '''
    if not t:
        return []
    return [(t.id(), t.data())] + convert_dictAB(t.lst) + convert_dictAB(t.rst)


def create_network(nodes, edges):
    '''
    List[int, Dict[str, T]] * Dict[Tuple[int, int], int] -> nx.Graph
    Returns a network graph given a list of nodes and a dictionnary
    of edges which contains the weight of each edge
    '''
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for (src, trg), wgt in filter_main_edges(edges).items():
        G.add_edge(src, trg)
        G[src][trg]['weight'] = wgt
    return G


def plot_network(G):
    '''
    nx.Graph -> None
    Plot G in a matplotlib window
    '''
    # Fruchterman-Reingold spatialization of nodes
    pos = nx.fruchterman_reingold_layout(G, dim=2)
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    plt.axis('off')
    plt.show()


def louvain_clustering(G):
    '''
    nx.Graph -> Dict[int, Set[int]]
    Processes to louvain clustering and returns a dictionnary
    which associates a number to each set of node ids
    resulting from louvain clustering
    '''
    clusters_list = nx.community.louvain_communities(G)
    clusters_dict = dict()
    i = 1
    for cluster in clusters_list:
        clusters_dict[i] = cluster
        i += 1
    for node in G.nodes():
        for cluster, nodes_set in clusters_dict.items():
            if node in nodes_set:
                G.nodes[node]['cluster'] = cluster
    return clusters_dict


def plot_network_with_clusters(G, clusters_dict):
    '''
    nx.Graph * Dict[int, int] -> None
    Plot G in a matplotlib window with the clusters legend
    given a network G and a cluster dictionnary
    '''
    nb_clusters = len(clusters_dict)

    bounds = []
    ticks = []
    for i in range(nb_clusters+1):
        bounds.append(i+0.5)
        if(i != 0):
            ticks.append(i)

    cmap = plt.get_cmap('tab10', nb_clusters)
    groups = set(nx.get_node_attributes(G, 'cluster').values())
    mapping = dict(zip(sorted(groups), count()))
    colors = [mapping[G.nodes[n]['cluster']] for n in G.nodes()]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    pos = nx.fruchterman_reingold_layout(G, dim=2)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(G, pos=pos, node_size=20, node_color=colors, cmap=cmap)

    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=ticks)
    plt.axis('off')
    plt.show()


def write_variables(file, t):
    '''
    FILE * dictBT -> None
    Writes the headers line in a file given
    a binary tree dictionnary t
    '''
    file.write('ID')
    for var in t.data():
        file.write(';' + var)
    file.write('\n')


def export_data_clustering(G, t, data_file_path):
    '''
    nx.Grpah * dictBT * str -> None
    Writes in data_file_path the data about the network G
    '''
    file = open(data_file_path, 'w', encoding='utf-8')
    write_variables(file, t)
    for id, data in G.nodes(data=True):
        file.write(str(id))
        for _, val in data.items():
            file.write(';' + str(val))
        file.write('\n')

def export_adjacency_matrix(G, matrix_file_path):
    '''
    nx.Graph * str -> None 
    Writes in matrix_file_path the adjacency matrix
    corresponding to the network G
    '''
    file = open(matrix_file_path, 'w', encoding='utf-8')
    file.write(',')
    am = nx.to_numpy_array(G)
    x_max, y_max = am.shape
    for x in range(x_max):
        if x < x_max -1:
            file.write(str(x + 1) + ',')
        else:
            file.write(str(x + 1) + '\n')
    for y in range(y_max):
        file.write(str(y + 1) + ',')
        for x in range(x_max):
            if x < x_max -1:
                file.write(str(am[x,y]) + ',')
            else:
                file.write(str(am[x,y]) + '\n')
    file.close()


def export_graph_indicators(G, indicators_file_path):
    '''
    nx.Graph * str -> None
    Computes the centrality indicators and write the result
    for each node of the network G in indicators_file_path
    '''
    # opening of the exit file and write headers (1)
    file = open(indicators_file_path, 'w')
    file.write('id;degree_centrality;betweenness_centrality;pagerank;closeness_centrality;eigenvector_centrality')
    # computation of the indicators
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    pagerank = nx.pagerank(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)
    # store data in exit file
    first_lap = True
    for id, data in G.nodes(data=True):
        #print(id, data)
        # write headers (2)
        if first_lap:
            for k in data:
                file.write(';' + str(k))
            file.write('\n')
            first_lap = False
        #write indicators' values
        file.write(str(id) + ';' +
                   str(degree_centrality[id]) + ';' +
                   str(betweenness_centrality[id]) + ';' +
                   str(pagerank[id]) + ';' +
                   str(closeness_centrality[id]) + ';' +
                   str(eigenvector_centrality[id]))
        # write data about node
        for k, v in data.items():
            file.write(';' + str(v))
        file.write('\n')
    # close exit file
    file.close()



def main_network(query, numerical_variables, files_path):
    '''
    str * List[str] -> None
    Computes the network analysis
    '''
    data_file_path = files_path + 'resultats2.csv'
    matrix_file_path = files_path + 'adjacency_matrix.csv'
    indicators_file_path = files_path + 'graph_indicators.csv'
    t = constuct_dictBT(query)
    jnb_for_variables_list(t, numerical_variables, 5)
    nodes = convert_dictAB(t)
    edges = construct_all_edges(t, t)
    G = create_network(nodes, edges)
    dc = louvain_clustering(G)
    plot_network_with_clusters(G, dc)
    export_data_clustering(G, t, data_file_path)
    export_adjacency_matrix(G, matrix_file_path)
    export_graph_indicators(G, indicators_file_path)



query = """
SELECT pressmark, order_in_ms, AVG(prop_up_space) AS 'prop_up_space', AVG(prop_down_space) AS 'prop_down_space', AVG(prop_ext_space) AS 'prop_ext_space', AVG(prop_int_space) AS 'prop_int_space', AVG(prop_black_space) AS 'prop_black_space', ROUND(AVG(nbr_columns), 0) AS 'nbr_columns', ROUND(AVG(nbr_lines), 0) AS 'nbr_lines', MAX(presence_annotations) AS 'presence_annotations', MAX(city) AS 'city', MAX(library) AS 'library', MAX(theme) AS 'theme', MAX(title) AS 'title'
FROM pages
GROUP BY pressmark, order_in_ms
ORDER BY pressmark, order_in_ms
"""

lnv = ['prop_up_space', 'prop_down_space', 'prop_ext_space', 'prop_int_space', 'prop_black_space']

file_path = './alto_version/resultats_clustering/'

main_network(query, lnv, file_path)
