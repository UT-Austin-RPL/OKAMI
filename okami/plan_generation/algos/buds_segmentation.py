
import os
import cv2
import torch

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from PIL import Image
from easydict import EasyDict
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from orion.algos.temporal_segments import TemporalSegments
from orion.algos.dino_features import rescale_feature_map, DinoV2ImageProcessor
from orion.utils.misc_utils import resize_image_to_same_shape

class Node():
    def __init__(self, start_idx, end_idx, level, idx):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.level = level
        self.idx = idx

        self.parent_idx = None
        self.children_node_indices = []
        self.cluster_label = None

    @property
    def centroid_idx(self):
        return (self.start_idx + self.end_idx) // 2

    @property
    def len(self):
        return self.end_idx - self.start_idx
    
class HierarchicalAgglomerativeTree():
    """Construct a hierarchical tree for clusters

    """
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.node_indices = []
        self.level_indices = {}
        self.graph = nx.Graph()
        self.root_node = None

    def add_nodes(self, nodes):
        for node in nodes:
            if node.idx in self.node_indices:
                continue
            self.add_node(node)

    def add_node(self, node):
        self.nodes.append(node)
        if node.children_node_indices != []:
            for child_idx in node.children_node_indices:
                self.edges.append([node.idx, child_idx])
        self.node_indices.append(node.idx)
        if node.level not in self.level_indices.keys():
            self.level_indices[node.level] = []

        self.level_indices[node.level].append(node.idx)

    def to_nx_graph(self):
        for node in self.nodes:
            self.graph.add_node(node.idx)
            
        for edge in self.edges:
            self.graph.add_edge(edge[0], edge[1])

        return self.graph

    def find_parent(self, node_idx):
        while self.nodes[node_idx].parent_idx is not None:
            node_idx = self.nodes[node_idx].parent_idx
        return self.nodes[node_idx]

    def create_root_node(self):
        root_node = Node(start_idx=0, end_idx=0, level=-1, idx=len(self.nodes))
        for node in self.nodes:
            if node.parent_idx is None:

                root_node.children_node_indices.append(node.idx)
                root_node.level = max(node.level + 1, root_node.level)
                node.parent_idx = root_node.idx
                self.edges.append([root_node.idx, node.idx])                

        self.nodes.append(root_node)
        self.root_node = root_node
        self.level_indices[self.root_node.level] = [self.root_node.idx]

    @property
    def max_depth(self):
        return max(self.level_indices.keys()) + 1

    
    def find_children_nodes(self, parent_node_idx, depth=0, no_leaf=False, min_len=20):
        # Return when depth = 0
        node_list = []
        for node_idx in self.nodes[parent_node_idx].children_node_indices:
            if depth == 0 or self.nodes[node_idx].len < min_len:
                node_list.append(node_idx)
            else:

                if no_leaf:
                    if self.nodes[node_idx].level == 1:
                        node_list.append(node_idx)
                    else:
                        node_list += self.find_children_nodes(node_idx, depth-1)
                else:
                    if self.nodes[node_idx].level == 0 or self.nodes[node_idx].len < min_len:
                        node_list.append(node_idx)
                    else:
                        node_list += self.find_children_nodes(node_idx, depth-1)

        return node_list

    def find_midlevel_abstraction(self, parent_node_idx, depth=0, no_leaf=False, min_len=40, sorted=False):
        # Return when depth = 0
        node_list = []
        for node_idx in self.nodes[parent_node_idx].children_node_indices:
            if depth == 0:
                node_list.append(node_idx)
            else:
                if no_leaf:
                    if self.nodes[node_idx].level == 1:
                        node_list.append(node_idx)
                    else:
                        node_list += self.find_children_nodes(node_idx, depth-1, min_len=min_len)
                else:
                    if self.nodes[node_idx].level == 0:
                        node_list.append(node_idx)
                    else:
                        node_list += self.find_children_nodes(node_idx, depth-1, min_len=min_len)
        if sorted:
            node_list = self.sort_node_list(node_list)
        return node_list

    def sort_node_list(self, unsorted_node_list):
        sorted_node_list = sorted(unsorted_node_list, key=lambda x: self.nodes[x].start_idx)
        return sorted_node_list

    def check_consistency(self, node_idx):
        node = self.nodes[node_idx]
        if node.level == 0:
            return True
        else:
            children_nodes = self.find_children_nodes(node_idx, 0)
            for child_idx in children_nodes:
                if node.cluster_label != self.nodes[child_idx].cluster_label:
                    return False
            return True
    
    def assign_labels(self, node_idx, label):
        self.nodes[node_idx].cluster_label = label

    def unassign_labels(self, node_idx):
        self.nodes[node_idx].cluster_label = None
        for child_idx in self.find_children_nodes(node_idx, 0):
            self.nodes[child_idx].cluster_label = None

    def compute_distance(self, e1, e2, mode="l2"):
        if mode == "l2":
            return np.linalg.norm(e1 - e2)
        elif mode == "l1":
            return np.linalg.norm((e1 - e2), ord=1)
        elif mode == "cos":
            cos_similarity = np.dot(e1, e2.transpose()) / (np.linalg.norm(e1) * np.linalg.norm(e2))
            return 1 - cos_similarity
        elif mode == "js":
            mu_e1, var_e1 = np.split(e1, 2, axis=-1)
            mu_e2, var_e2 = np.split(e2, 2, axis=-1)
            def kl_normal(qm, qv, pm, pv):
                element_wise = 0.5 * (np.log(pv) - np.log(qv) + qv / pv + np.power(qm - pm, 2) / pv - 1)
                return element_wise.sum(-1)
            js_dist = 0.5 * (kl_normal(mu_e1, var_e1, mu_e2, var_e2) + kl_normal(mu_e2, var_e2, mu_e1, var_e1))
            return  js_dist

    def node_footprint(self, node: Node, embeddings, mode="global_pooling"):
        if mode == "centroid":
            return embeddings[node.centroid_idx]
        elif mode == "mean":
            embedding = np.mean([embeddings[node.start_idx], embeddings[node.centroid_idx], embeddings[node.end_idx]], axis=0)
            return embedding
        elif mode == "head":
            return embeddings[node.start_idx]
        elif mode == "tail":
            return embeddings[node.end_idx]
        elif mode == "concat_1":
            return np.concatenate([embeddings[node.start_idx], embeddings[node.centroid_idx], embeddings[node.end_idx]], axis=1)
        elif mode == "gaussian":
            mu = np.mean(embeddings[node.start_idx:node.end_idx+1], axis=0)
            var = np.mean(np.square(embeddings[node.start_idx:node.end_idx+1]), axis=0) - mu ** 2 + 1e-5
            assert(np.all(mu.shape == var.shape))
            return np.concatenate([mu, var], axis=1)
        elif mode == "global_pooling":
            return np.mean(embeddings[node.start_idx:node.end_idx+1], axis=0).reshape(1, -1)

    def find_nn(self, embeddings, node_idx, before_idx, after_idx, footprint_mode="global_pooling", dist_mode="cos"):
        f1 = self.node_footprint(self.nodes[node_idx], embeddings, mode=footprint_mode)
        f2 = self.node_footprint(self.nodes[before_idx], embeddings, mode=footprint_mode) 
        f3 = self.node_footprint(self.nodes[after_idx], embeddings, mode=footprint_mode)

        d1 = self.compute_distance(f1, f2, mode=dist_mode)
        d2 = self.compute_distance(f1, f3, mode=dist_mode)

        return d1 < d2

    def hierarchical_clustering(self, embeddings, step, footprint_mode="global_pooling", dist_mode="cos", len_penalty=True):
        idx = 0
        nodes = []
        terminate = False
        for i in range(len(embeddings)-1):
            if i % step == 0:
                if (i + 2 * step >= len(embeddings)-1):
                    start_idx = i
                    end_idx = len(embeddings) - 1
                    terminate = True
                else:
                    start_idx = i
                    end_idx = min(i + step, len(embeddings) - 1)
                node = Node(start_idx=start_idx,
                            end_idx=end_idx,
                            level=0,
                            idx=idx)

                idx += 1
                nodes.append(node)
                if terminate:
                    break
        self.add_nodes(nodes)

        while len(nodes) > 2:
            i = 1
            dist_seq = []            
            for i in range(len(nodes) - 1):
                dist = self.compute_distance(self.node_footprint(nodes[i], embeddings, mode=footprint_mode),
                                                      self.node_footprint(nodes[i+1], embeddings, mode=footprint_mode), mode=dist_mode)
                if len_penalty:
                    # Very simple penalty
                    # dist += (nodes[i].len + nodes[i+1].len) * (1./
                    # 10.)
                    
                    # Pentaly with respect to the whole length
                    dist += (nodes[i].len + nodes[i+1].len) / (5. * len(nodes))
                dist_seq.append(dist)

            target_idx = dist_seq.index(min(dist_seq))

            new_node = Node(start_idx=nodes[target_idx].start_idx,
                            end_idx=nodes[target_idx+1].end_idx,
                            level=max(nodes[target_idx].level, nodes[target_idx+1].level) + 1,
                            idx=idx)
            new_node.children_node_indices = [nodes[target_idx].idx, nodes[target_idx + 1].idx]
            nodes[target_idx].parent_idx = idx
            nodes[target_idx + 1].parent_idx  = idx
            self.add_node(new_node)
            idx += 1

            new_nodes = []
            visited_nodes = []

            for node in nodes:
                parent_node = self.find_parent(node.idx)
                if parent_node.idx not in visited_nodes:
                    new_nodes.append(parent_node)
                    visited_nodes.append(parent_node.idx)
            nodes = new_nodes

def get_feature_encoder(model_name="dinov2"):
    if model_name == "dinov2":
        return DinoV2ImageProcessor
    else:
        raise NotImplementedError
    
class Buds_Segmentation():
    def __init__(self,
                 buds_segmentation_cfg) -> None:
        
        self.cfg = buds_segmentation_cfg

        # initialize feature encoding model
        self.feature_encoder = get_feature_encoder(model_name=self.cfg.feature_encoder.model_name)(**self.cfg.feature_encoder.kwargs)

        # initialize hierarchical clustering model
        self.task_tree = HierarchicalAgglomerativeTree()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def reset(self):
        self.task_tree = HierarchicalAgglomerativeTree()

    def close(self):
        del self.feature_encoder
        del self.task_tree

    def encode_features(self,
                        video_seq):
           
        # Encode rightview_seq with dino
        ratio_height = 448. / video_seq[0].shape[0]
        ratio_width = 448. / video_seq[0].shape[1]
        ratio = max(ratio_height, ratio_width)
        reference_size = (int(video_seq[0].shape[0] * ratio), int(video_seq[0].shape[1] * ratio))
        print(f"Dinov2 will rescale images to : {reference_size}, original_size is {video_seq[0].shape}")

        resized_img_seq = [resize_image_to_same_shape(single_image, reference_size=reference_size) for single_image in video_seq]
        feature_seq = self.feature_encoder.process_images(resized_img_seq)

        # feature: (T, H, W, C) -> (T, C, H, W)
        feature_seq = feature_seq.permute(0, 3, 1, 2)

        feature_seq = rescale_feature_map(feature_seq, 
                                          target_h=1,
                                          target_w=1, 
                                          convert_to_numpy=False)
        return feature_seq

    def get_seq_saving_file(self, seq_name):
        return f"results/test/{seq_name}_dinov2_feature.pt" 

    def bfs_search(self, cfg):
        depth = init_depth = 0
        node_list = []
        while len(node_list) < cfg.temporal_segmentation.min_seg_num:
            node_list = self.task_tree.find_midlevel_abstraction(self.task_tree.root_node.idx, depth=depth, min_len=cfg.temporal_segmentation.min_len, sorted=cfg.temporal_segmentation.sorted)
            depth += 1
        return node_list

    def forward(self, 
                video_seq=None,
                seq_name="tmp_dinov2_feature_seq",
                save_feature=False,
                cfg=None):
        if cfg is None:
            cfg = self.cfg
        if video_seq is not None:
            feature_seq = self.encode_features(video_seq)
            if save_feature:
                torch.save(feature_seq, self.get_seq_saving_file(seq_name))
        else:
            feature_seq = torch.load(self.get_seq_saving_file(seq_name))

        feature_seq = feature_seq.detach().cpu().numpy()

        self.task_tree.hierarchical_clustering(feature_seq,
                                               step=cfg.hierarchical_clustering.shortest_len,
                                               footprint_mode=cfg.hierarchical_clustering.footprint_mode,
                                               dist_mode=cfg.hierarchical_clustering.distance_metric)
        self.task_tree.create_root_node()

        node_list = self.bfs_search(cfg)

        temporal_segments = TemporalSegments()
        for node_idx in node_list:
            node = self.task_tree.nodes[node_idx]
            temporal_segments.add_segment(node.start_idx, node.end_idx)
        return {"temporal_segments": temporal_segments,
                "feature_seq": feature_seq,
                "node_list": node_list}
    
    def segment_features(self, 
                         node_list, 
                         feature_seq=None,
                         seq_name="tmp_dinov2_feature_seq"):
        if feature_seq is None:
            feature_seq = torch.load(self.get_seq_saving_file(seq_name))
            feature_seq = feature_seq.detach().cpu().numpy()

        segment_features = []
        for node_idx in node_list:
            node = self.task_tree.nodes[node_idx]
            segment_features.append(self.task_tree.node_footprint(node, feature_seq, mode=self.cfg.hierarchical_clustering.footprint_mode).squeeze())

        return segment_features