import time
import os
import pickle
import random

import numpy as np
import dgl
import torch
import glob
from torch.utils.data import Dataset, DataLoader

class SROIEDataset(Dataset):

    def __init__(self,
                 data_dir,
                 split,
                 alphabet=None,
                 labels=None,
                 dropout=0.):
        assert isinstance(labels, list) and len(labels) > 1
        self.labels = labels
        self.split = split
        self.data_dir = data_dir
        self.alphabet = alphabet
        self.dropout = dropout

        self.node_labels = []
        self.graph_lists = []
        self.boxes = []
        self.edges = []
        self.text_lists = []
        self.text_lengths = []
        self.edata = []
        self.srcs = []
        self.dsts = []

        self._prepare(data_dir)
        self.n_samples = len(self.node_labels)

    def collate(self, samples):
        # import pdb; pdb.set_trace()
        graphs, labels, text, text_length = map(list, zip(*samples))
        labels = np.concatenate(labels)
        text_lengths = np.concatenate(text_length)
        max_length = text_lengths.max()

        texts = np.concatenate(text)

        new_text = [np.expand_dims(np.pad(t, (0, max_length - t.shape[0]), 'constant'), axis=0) for t in texts]
        texts = np.concatenate(new_text)


        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        graph_node_size = [graphs[i].number_of_nodes() for i in range(len(graphs))]
        graph_edge_size = [graphs[i].number_of_edges() for i in range(len(graphs))]


        return batched_graph, \
            torch.from_numpy(labels), \
            snorm_n, \
            snorm_e, \
            torch.from_numpy(texts), \
            torch.from_numpy(text_lengths), \
            graph_node_size, \
            graph_edge_size

    def _text_encode(self, text):
        text_encode = []
        for t in text.upper():
            if t not in self.alphabet:
                text_encode.append(self.alphabet.index(" "))
            else:
                text_encode.append(self.alphabet.index(t))
        return np.array(text_encode)

    def _load_annotation(self, annotation_file):
        texts = []
        text_lengths = []
        boxes = []
        labels = []
        with open(annotation_file, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                splits = line.strip().split(self.split)
                if len(splits) < 10:
                    continue
                text_encode = self._text_encode(splits[8])

                text_lengths.append(text_encode.shape[0])
                texts.append(text_encode)
                box_info = [int(x) for x in splits[:8]]
                # import pdb; pdb.set_trace()
                box_info.append(np.max(box_info[0::2]) - np.min(box_info[0::2]))
                box_info.append(np.max(box_info[1::2]) - np.min(box_info[1::2]))
                boxes.append([int(x) for x in box_info])
                labels.append(self.labels.index(splits[9]))
        return np.array(texts), np.array(text_lengths), np.array(boxes), np.array(labels)

    def _prepare(self, data_dir):
        annotation_list = os.listdir(data_dir)

        for ann in annotation_list:
            annotation_file = os.path.join(data_dir, ann)
            texts, text_lengths, boxes, labels = self._load_annotation(annotation_file)

            node_nums = labels.shape[0]
            src = []
            dst = []
            edge_data = []
            for i in range(node_nums):
                for j in range(node_nums):
                    if i == j:
                        continue

                    edata = []
                    # import pdb; pdb.set_trace()
                    #y distance
                    y_distance = np.mean(boxes[i][:8][1::2]) - np.mean(boxes[j][:8][1::2])
                    x_distance = np.mean(boxes[i][:8][0::2]) - np.mean(boxes[j][:8][0::2])
                    w = boxes[i, 8]
                    h = boxes[i, 9]

                    if np.abs(y_distance) >  3 * h:
                        continue
                    
                    edata.append(y_distance)
                    edata.append(x_distance)

                    edge_data.append(edata)
                    src.append(i)
                    dst.append(j)

            edge_data = np.array(edge_data)
            g = dgl.DGLGraph()
            g.add_nodes(node_nums)
            g.add_edges(src, dst)
            
            boxes, edge_data, texts, text_length = self._prepapre_pipeline(boxes, edge_data, texts, text_lengths)

            boxes = torch.from_numpy(boxes).float()
            edge_data = torch.from_numpy(edge_data).float()

            g.edata['feat'] = edge_data
            g.ndata['feat'] = boxes
            
            self.graph_lists.append(g)
            self.boxes.append(boxes)
            self.edges.append(edge_data)
            self.srcs.append(src)
            self.dsts.append(dst)
            self.text_lengths.append(text_lengths)
            self.text_lists.append(texts)
            self.node_labels.append(labels)
            # break
            
    #data pipeline
    def _prepapre_pipeline(self, boxes, edge_data, text, text_length):
        # boxes = boxes[[0, 1, 4, 5]]

        box_min = boxes.min(0)
        box_max = boxes.max(0)

        boxes = (boxes - box_min) / (box_max - box_min)
        boxes = (boxes - 0.5) / 0.5

        edge_min = edge_data.min(0)
        edge_max = edge_data.max(0)

        edge_data = (edge_data - edge_min) / (edge_max - edge_min)
        edge_data = (edge_data - 0.5) / 0.5

        return boxes, edge_data, text, text_length

    def __getitem__(self, idx):
        graph = self.graph_lists[idx]
        node = self.node_labels[idx]
        text = self.text_lists[idx]
        text_length = self.text_lengths[idx]

        if self.dropout > 0. and random.random() < 0.5:
            #重新构建graph， 因为graph没有clone()或者copy()方法
            boxes = self.boxes[idx]
            edge_data = self.edges[idx]
            src = self.srcs[idx]
            dst = self.dsts[idx]

            graph = dgl.DGLGraph()
            graph.add_nodes(node.shape[0])
            graph.add_edges(src, dst)
            graph.edata['feat'] = edge_data
            graph.ndata['feat'] = boxes

            node_nums = graph.number_of_nodes()
            others = np.where(node == 0)[0]
            drop_nums = random.randint(1, int(others.shape[0] * self.dropout))
            dropout_idx = random.sample(list(others), drop_nums)
            keep_idx = [i for i in range(node_nums) if i not in dropout_idx]
            
            graph.remove_nodes(dropout_idx)
            node = node[keep_idx]
            text = text[keep_idx]
            text_length = text_length[keep_idx]

        # text = np.concatenate(text)
        return graph, node, text, text_length

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

if __name__ == '__main__':

    seed = 100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    data_dir = '/data/yihao/src/MyLearning/graph/benchmarking-gnns/data/SROIE2019/filter_data_3'
    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ%[](){}<>&+=\'"!?~:/-@*_.,;·|#$\\^ '

    split = '\t'
    labels = ['other', 'company', 'address', 'date', 'total']
    dataset = SROIEDataset(data_dir=data_dir, split=split, labels=labels, alphabet=alphabet)


    train_loader = DataLoader(dataset, batch_size=12, shuffle=True, collate_fn=dataset.collate)

    for data in train_loader:
        g, labels, snorm_n, snorm_e, box, text, text_length = data
        print(labels.shape, box.shape, text.shape, text_length.shape, sum(text_length), g.number_of_edges(), g.number_of_nodes())

    for data in train_loader:
        g, labels, snorm_n, snorm_e, box, text, text_length = data
        print(labels.shape, box.shape, text.shape, text_length.shape, sum(text_length), g.number_of_edges(), g.number_of_nodes())

