from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
from typing import Optional, Any
from utils.my_dataloader import Temporal_Dataloader, data_load, Temporal_Splitting
import torch 
from numpy import ndarray
import copy

class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray, \
                hash_table: dict[int, int] = None, node_feat: Optional[ndarray|None] = None, edge_feat: Optional[ndarray|None] = None):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
        self.node_feat = node_feat
        self.edge_feat = edge_feat
        self.hash_table = hash_table
        
        self.target_node: Optional[set|None] = None
        self.seen_nodes: np.ndarray = None

    def set_up_seen_nodes(self, seen_nodes: np.ndarray, indcutive_seen_nodes: np.ndarray):
        """
        set up the seen nodes
        :param seen_nodes: np.ndarray, seen nodes
        :param indcutive_seen_nodes: np.ndarray, inductive seen nodes
        """
        self.seen_nodes = seen_nodes
        self.target_node = indcutive_seen_nodes

def to_TPPR_Data(graph: Temporal_Dataloader) -> Data:
    nodes = graph.x
    edge_idx = np.arange(graph.edge_index.shape[1])
    timestamp = graph.edge_attr
    src, dest = graph.edge_index[0, :], graph.edge_index[1, :]
    labels = graph.y

    hash_dataframe = copy.deepcopy(graph.my_n_id.node.loc[:, ["index", "node"]].values.T)
    
    """
    :param hash_table, should be a matching list, now here it is refreshed idx : origin idx,
    """
    hash_table: dict[int, int] = {idx: node for idx, node in zip(*hash_dataframe)}

    edge_feat, node_feat = graph.edge_pos, graph.node_pos
    TPPR_data = Data(src_node_ids= src, dst_node_ids=dest, node_interact_times=timestamp, edge_ids = edge_idx, \
                     labels=labels, hash_table=hash_table, node_feat=node_feat, edge_feat=edge_feat)

    return TPPR_data

def span_time_quantile(threshold: float, tsp: np.ndarray, dataset: str):
    val_time = np.quantile(tsp, threshold)
    
    if dataset in ["dblp", "tmall"]:
        spans, span_freq = np.unique(tsp, return_counts=True)
        if val_time == spans[-1]: val_time = spans[int(spans.shape[0]*threshold)]
    return val_time

def get_link_prediction_data(dataset_name: str, snapshot: int, val_ratio: float = 0.8, test_ratio: float=0.0):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    # graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    # edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    # node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))
    view = snapshot - 2

    graph, idx_list = data_load(dataset=dataset_name, emb_size = 64)
    node_raw_features, edge_raw_features = graph.pos

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)
    
    graph.pos = node_raw_features, edge_raw_features

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'
    graph_list = Temporal_Splitting(graph).temporal_splitting(time_mode="view", snapshot=snapshot, views = view)
    
    assert len(graph_list) == view, f"the number of views should be {view}, but got {len(graph_list)}"
    Data_list = list()
    for idx in range(view-1):
        temporal_graph = graph_list[idx]
        # get the timestamp of validate and test set
        full_data = to_TPPR_Data(temporal_graph)
        src_node_ids = full_data.src_node_ids.astype(np.longlong)
        dst_node_ids = full_data.dst_node_ids.astype(np.longlong)
        node_interact_times = full_data.node_interact_times.astype(np.float64)
        edge_ids = full_data.edge_ids.astype(np.longlong)
        labels = full_data.labels
        
        val_time = span_time_quantile(threshold=0.8, tsp=node_interact_times, dataset=dataset_name)


        # the setting of seed follows previous works
        random.seed(2020)

        # union to get node set
        node_set = set(src_node_ids) | set(dst_node_ids)
        num_total_unique_node_ids = len(node_set)

        # compute nodes which appear at test time
        t1_temporal: Temporal_Dataloader = graph_list[idx+1]
        t1_full_data: Data = to_TPPR_Data(t1_temporal)
        t1_node_set = set(t1_full_data.src_node_ids) | set(t1_full_data.dst_node_ids)
        t1_num_unique_node_ids = len(t1_node_set)
        t_hash_table, t1_hash_table = full_data.hash_table, t1_full_data.hash_table
        
        """
        Basically, test_node_set should be valid node set consdiering its temporal features, lets make a row:
        new val test set = set(random.sample(val_ndoe_set, int(0.1 * val_node_set))
        """
        val_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
        new_val_node_set = set(random.sample(sorted(val_node_set), int(0.05 * num_total_unique_node_ids)))
        
        new_val_source_mask = np.isin(src_node_ids, sorted(new_val_node_set))
        new_val_destination_mask = np.isin(dst_node_ids, sorted(new_val_node_set))
        
        observed_edge_mask = np.logical_and(~new_val_source_mask, ~new_val_destination_mask)
        # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
        train_mask = np.logical_and(node_interact_times <= val_time, observed_edge_mask)
        # train_mask = node_interact_times<=val_time

        train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                        node_interact_times=node_interact_times[train_mask],
                        edge_ids=edge_ids[train_mask], labels=labels)

        # define the new nodes sets for testing inductiveness of the model
        train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
        # assert len(train_node_set & new_val_node_set) == 0
        # new nodes that are not in the training set
        new_node_set = node_set - train_node_set # key points 1

        val_mask = node_interact_times > val_time

        # new edges with new nodes in the val and test set (for inductive evaluation)
        edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                                for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)

        # validation and test data
        val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                        node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels)

        test_data = t1_full_data

        # validation and test with edges that at least has one new node (not in training set)
        new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                                node_interact_times=node_interact_times[new_node_val_mask],
                                edge_ids=edge_ids[new_node_val_mask], labels=labels)
        """
        try to not resort the new node set, see whether it works
        """
        # t_node_match, t1_node_match = np.vectorize(t1_hash_table.get), np.vectorize(t_hash_table.get)
        # t_node_original, t1_node_original = t_node_match(train_node_set), t1_node_match(t1_node_match)
        
        # t1_new_node_set = t1_node_original - t_node_original # key points 2
        # reverse_hash_table = {v: k for k, v in t1_hash_table.items()}
        # reverse_t1_new_node_set = np.vectorize(reverse_hash_table.get)(t1_new_node_set)
        t1_new_node_set = t1_node_set - train_node_set # key points 2.1
        t1_edge_contains_new_node_mask = np.array([(src_node_id in t1_new_node_set or dst_node_id in t1_new_node_set)
                                                    for src_node_id, dst_node_id in zip(t1_full_data.src_node_ids, t1_full_data.dst_node_ids)])
        new_node_test_mask = np.logical_and(t1_full_data.node_interact_times, t1_edge_contains_new_node_mask)  
        
        new_node_test_data = Data(src_node_ids=test_data.src_node_ids[new_node_test_mask], dst_node_ids=test_data.dst_node_ids[new_node_test_mask],
                                node_interact_times=test_data.node_interact_times[new_node_test_mask],
                                edge_ids=test_data.edge_ids[new_node_test_mask], labels=test_data.labels)
        
        Data_list.append([full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data])

        print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
        print("The training dataset has {} interactions, involving {} different nodes, with ratio of {:.4f}".format(
            train_data.num_interactions, train_data.num_unique_nodes, train_data.num_interactions/full_data.num_interactions))
        print("The validation dataset has {} interactions, involving {} different nodes, with ratio of {:.4f}".format(
            val_data.num_interactions, val_data.num_unique_nodes, val_data.num_interactions / full_data.num_interactions))
        print("The test dataset has {} interactions, involving {} different nodes".format(
            test_data.num_interactions, test_data.num_unique_nodes))
        print("The new node validation dataset has {} interactions, involving {} different nodes".format(
            new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes))
        print("The new node test dataset has {} interactions, involving {} different nodes".format(
            new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes))
        print("{} nodes were used for the inductive testing, i.e. are never seen during training\n\n".format(len(t1_new_node_set)))

    return node_raw_features, edge_raw_features, Data_list # full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data


def get_node_classification_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for node classification task
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values

    # The setting of seed follows previous works
    random.seed(2020)

    train_mask = node_interact_times <= val_time
    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data

def quantile_(threshold: float, timestamps: torch.Tensor) -> tuple[torch.Tensor]:
    full_length = timestamps.shape[0]
    val_idx = int(threshold*full_length)

    if not isinstance(timestamps, torch.Tensor):
        timestamps = torch.from_numpy(timestamps)
    train_mask = torch.zeros_like(timestamps, dtype=bool)
    train_mask[:val_idx] = True

    val_mask = torch.zeros_like(timestamps, dtype=bool)
    val_mask[val_idx:] = True

    return train_mask, val_mask