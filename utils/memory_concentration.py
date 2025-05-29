import torch
import numpy as np
from typing import List, Tuple, Optional, Any

class MemoryConcentration(torch.nn.Module):
    def __init__(self):
        super(MemoryConcentration, self).__init__()
    
    def concentrate(self, edges_idx, edge_embeddings):
        """
        Concentrate the same node embeddings in a src_edge_embeddings
        the idea is use different method to concentrate the same node embeddings
        :param edges_idx: Tensor, shape (num_edges, ), the index of the edges only for src node
        :param edge_embeddings: Tensor, shape (num_edges, dim), the embeddings of the edges
        :return: Tensor, shape (num_nodes, dim), the concentrated node embeddings
        """
        unique_nodes, unique_indices = np.unique(edges_idx, return_index=True)
        num_length, concrete_list = unique_nodes.shape[0], list()
        for i in range(num_length):
            node_idx = unique_nodes[i]
            node_embeddings = edge_embeddings[edges_idx == node_idx]
            if node_embeddings.shape[0]>1:
                node_embeddings = torch.mean(node_embeddings, dim=0)
            else: node_embeddings = node_embeddings.squeeze(0)
            concrete_list.append(node_embeddings)
        return torch.stack(concrete_list, dim=0), unique_indices       

if __name__ == "__main__":
    memory_concentration = MemoryConcentration()

    # Define test inputs
    edges_idx = torch.tensor([0, 1, 0, 2, 1, 3, 3, 3])
    edge_embeddings = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
        [13.0, 14.0],
        [15.0, 16.0]
    ])

    # Expected outputs
    expected_concentrated_embeddings = torch.tensor([
        [3.0, 4.0],  # Mean of embeddings for node 0
        [6.0, 7.0],  # Mean of embeddings for node 1
        [7.0, 8.0],  # Embedding for node 2
        [13.0, 14.0] # Mean of embeddings for node 3
    ])
    expected_unique_indices = np.array([0, 1, 3, 5])

    # Run the concentrate method
    concentrated_embeddings, unique_indices = memory_concentration.concentrate(edges_idx, edge_embeddings)

    # Assertions
    assert torch.allclose(concentrated_embeddings, expected_concentrated_embeddings), \
        f"Expected {expected_concentrated_embeddings}, but got {concentrated_embeddings}"
    assert np.array_equal(unique_indices, expected_unique_indices), \
        f"Expected {expected_unique_indices}, but got {unique_indices}"
