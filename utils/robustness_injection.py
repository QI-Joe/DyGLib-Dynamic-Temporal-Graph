from utils.my_dataloader import Temporal_Dataloader, NodeIdxMatching, Dynamic_Dataloader, data_load
import numpy as np
from numpy import ndarray
import copy
from typing import List, Tuple, Optional, Any
import torch

class Imbalance(object): 
    def __init__(self, ratio: float, val_time: float, src_label, val_ratio: Optional[float | None] = None, *args, **kwargs):
        super(Imbalance, self).__init__(*args, **kwargs)
        self.imbalance_ratio = ratio
        self.val_time = val_time
        self.seen_node: ndarray = None
        self.src_label: Optional[np.ndarray | Any] = src_label
        self.val_ratio = val_ratio
        
    def __call__(self, data: Temporal_Dataloader, *args, **kwds):
        """
        Imbalance Data Evaluation. In the node classification task, given a dataset G = (𝐺𝑖 , 𝑦𝑖 ), we simulate class imbalance by setting \n
        the proportions of training samples per class as {1, 1/2^𝛽 , 1/3^𝛽 , . . . , 1/|Y|^𝛽 }, where 𝛽 ∈ {0, 0.5, 1, 1.5, 2} controls the imbalance ratio. \n
        The num-ber of samples in the first class is fixed under all 𝛽 values.
        """
        nodes = data.x.cpu().numpy() if isinstance(data.x, torch.Tensor) else data.x
        node_interact_times = data.edge_attr.cpu().numpy() if isinstance(data.edge_attr, torch.Tensor) else data.edge_attr
        edge_index = data.edge_index
        if isinstance(data.edge_index, torch.Tensor):
            edge_index = data.edge_index.cpu().numpy()
        
        src, dst = edge_index[0], edge_index[1]
        
        # All for implmentation
        # available_node_list: list[np.ndarray] = self.pooling_check(seen_node, node_match_list)
        
        uniqclass, uniquenum = np.unique(self.src_label, return_counts=True)
        fixed_sample = uniquenum[0]
        sample_per_classes = [int(fixed_sample/((i+1)**self.imbalance_ratio)) for i in range(len(uniqclass))]
        
        selected_idx, outside_select = [], []
        for class_label, num_samples in zip(uniqclass, sample_per_classes):
            class_idx = np.where(self.src_label == class_label)[0]
            if num_samples>0 and len(class_idx)>0:
                selected = np.random.choice(class_idx, min(num_samples, len(class_idx)), replace=False)
                not_selected = list(set(class_idx.tolist()) - set(selected.tolist()))
                selected_idx.extend(selected)
                outside_select.extend(not_selected)
        
        """
        Attention! There should be a assert to evaluate one thing:
        (np.array(selected_idx.extend(outside_select)) == seen_node).all() == True
        """
        
        train_seen_node_mask = np.zeros(src.shape, dtype=bool)
        train_seen_node_mask[selected_idx] = True
        assert train_seen_node_mask.sum()>=len(selected_idx), f"train_seen_node_mask: {train_seen_node_mask.sum()}, selected_idx: {len(selected_idx)}"
        
        train_mask = train_seen_node_mask & (node_interact_times<=self.val_time)
        val_mask = node_interact_times>self.val_time
        nn_val_mask = ~train_seen_node_mask
        
        self.seen_node = src[train_mask]
        print(f"Imbalance ratio: {self.imbalance_ratio}, train seen node / random selected node: {round(train_mask.sum()/train_seen_node_mask.sum(), 4)}")
        
        return train_mask, val_mask, nn_val_mask
    
    def test_processing(self, t1_data: Temporal_Dataloader, new_test_msak: Optional[ndarray | None] = None):
        """
        :return: inductive test mask for unseen nodes in src edges
        """
        t1_edge_index = t1_data.edge_index
        if isinstance(t1_data.edge_index, torch.Tensor):
            t1_edge_index = t1_data.edge_index.cpu().numpy()
        t1_src = t1_edge_index[0]
        nn_test_mask = ~np.isin(t1_src, self.seen_node)
        return nn_test_mask
        
        
class Few_Shot_Learning(object):
    def __init__(self, fsl_num: int, val_time: Optional[float|None] = None, src_label: Optional[np.ndarray|Any] = None, *args, **kwargs):
        super(Few_Shot_Learning, self).__init__(*args, **kwargs)
        self.fsl_num = fsl_num
        self.seen_node: ndarray = None
        self.val_time = val_time # val_ratio here equals val_ratio + train_ratio
        self.src_label: Optional[np.ndarray | Any] = src_label
        
    def __call__(self, data: Temporal_Dataloader, *args, **kwds):
        """
        Few-shot Evaluation. Specifically, For graph classification \n
        tasks, given a training graph dataset G = {(𝐺𝑖 , 𝑦𝑖 )}, we set the \n
        number of training graphs per class as 𝛾 ∈ {10, 20, 30, 40, 50}. \n
        
        Note that few shot learning will no longer consider the train_val split, \n
        considering limited training data
        """
        # node_num, label = data.num_nodes, data.y.cpu().numpy() if isinstance(data.y, torch.Tensor) else data.y
        # uniquclss, uniqunum = np.unique(label, return_counts=True)
        src_unique_label, src_unique_num_label = np.unique(self.src_label, return_counts=True)
        
        training_data, node_match_list = [], data.my_n_id.node.values # "index", "original node idx", "label"
        src = data.edge_index[0].cpu().numpy() if isinstance(data.edge_index, torch.Tensor) else data.edge_index[0]
        data_interact_times = data.edge_attr.cpu().numpy() if isinstance(data.edge_attr, torch.Tensor) else data.edge_attr
        
        for cls in src_unique_label:
            class_indices = np.where(self.src_label == cls)[0]
            np.random.shuffle(class_indices)
            
            num_samples = min(self.fsl_num, len(class_indices))
            current_cls_selelcted_indices = class_indices[: num_samples]
            training_data.extend(current_cls_selelcted_indices)
        
        print(f"Few shot learning: {self.fsl_num}, training data: {len(training_data)}, uniqu classes: {src_unique_label.shape[0]}")
        train_mask = np.zeros(src.shape, dtype=bool)
        train_mask[training_data] = True
        
        val_mask = data_interact_times>self.val_time
        nn_val_mask = ~train_mask
        
        self.seen_node = src[training_data]
        return train_mask, val_mask, nn_val_mask
    
    def test_processing(self, t1_data: Temporal_Dataloader, new_test_msak: Optional[ndarray | None] = None):
        t1_edge_index = t1_data.edge_index
        if isinstance(t1_data.edge_index, torch.Tensor):
            t1_edge_index = t1_data.edge_index.cpu().numpy()
        t1_src = t1_edge_index[0]
        nn_test_mask = ~np.isin(t1_src, self.seen_node)
        return nn_test_mask
    
class Edge_Distrub(object):
    def __init__(self, ratio: float, *args, **kwargs):
        super(Edge_Distrub, self).__init__(*args, **kwargs)
        self.ratio = ratio
        self.seen_node: ndarray = None
        self.node_match_list: Optional[NodeIdxMatching | Any] = None
        
    def __call__(self, data: Temporal_Dataloader):
        if self.ratio>=0.9:
            raise ValueError("The ratio should be less than 0.9")
        num_edges = data.num_edges
        num_edge2delete = int(num_edges * self.ratio)
        
        edge_index: np.ndarray = data.edge_index.cpu().numpy() if isinstance(data.edge_index, torch.Tensor) else data.edge_index
        # guarantee all the node has been seen once
        """
        :param unique_freq: provided by return_index=True !! not return_counts=True \n
        thus unique_freq record the first appearance of each node in entire graph in format of (n,) \n
        [1, 18, 45, 17, ..., 386_560] as index match of each node [0, 2, 5, 1, ..., 28578] since it has been flatten 
        """
        unique_nodes, first_appear_idx = np.unique(edge_index.flatten(), return_index=True)
        non_flatten_mask = first_appear_idx >= num_edges
        first_appear_idx[non_flatten_mask] = first_appear_idx[non_flatten_mask] - num_edges
        selected_edge_mask = np.ones(num_edges, dtype=bool)
        selected_edge_mask[first_appear_idx] = False
        edge_able2remove = np.arange(num_edges)[selected_edge_mask]
        
        edge_idx2delete = np.random.choice(edge_able2remove, num_edge2delete, replace=False)
        mask = np.ones(num_edges, dtype=bool)
        mask[edge_idx2delete] = False
        
        retained_edges = edge_index[:, mask]
        retained_tmp = data.edge_attr[mask]
        if isinstance(data.edge_attr, torch.Tensor):
            retained_tmp = retained_tmp.cpu().numpy()
        
        return retained_edges, retained_tmp
    
    def test_processing(self, t1_data: Temporal_Dataloader, new_test_msak: Optional[ndarray | None] = None):
        """
        :funtionality: No specific utility meaning, only to align with structure with that of Imbalance and FSL 
        :return: unmodified t1_data
        """
        if (new_test_msak).all() != None:
            return new_test_msak
        raise ValueError("The new_test_mask should not be None")