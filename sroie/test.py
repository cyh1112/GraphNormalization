import os

from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import dgl
import cv2

from dataset import SROIEDataset
from gated_gcn import GatedGCNNet

def accuracy(scores, targets):
    S = targets.cpu().numpy()
    C = np.argmax( torch.nn.Softmax(dim=1)(scores).cpu().detach().numpy() , axis=1 )
    CM = confusion_matrix(S,C).astype(np.float32)
    # import pdb; pdb.set_trace()
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    recall = np.zeros(nb_classes)
    precision = np.zeros(nb_classes)
    F1_score = np.zeros(nb_classes)

    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            recall[r] = CM[r,r]/ float(cluster.shape[0])
            if np.sum(CM[:, r]) > 0:
                precision[r] = CM[r,r] / np.sum(CM[:, r])
            else:
                precision[r] = 0.0

            if (precision[r] + recall[r]) > 0.:
                F1_score[r] = 2 * recall[r] * precision[r] / (precision[r] + recall[r])
            else:
                F1_score[r] == 0.

            if CM[r,r]>0:
                nb_non_empty_classes += 1
        else:
            recall[r] = 0.0
            precision[r] = 0.0
            F1_score[r] = 0.0
    return recall, precision, F1_score

def _text_encode(text):
        text_encode = []
        for t in text.upper():                
            if t not in alphabet:
                text_encode.append(alphabet.index(" "))
            else:
                text_encode.append(alphabet.index(t))
        return np.array(text_encode)

def _load_annotation(annotation_file):
    # import pdb; pdb.set_trace()    
    # print(annotation_file)
    texts = []
    text_lengths = []
    boxes = []
    labels = []
    with open(annotation_file) as f:
        lines = f.readlines()
        for line in lines:
            splits = line.strip().split('\t')
            if len(splits) < 10:
                continue
            text_encode = _text_encode(splits[8])

            text_lengths.append(text_encode.shape[0])
            texts.append(text_encode)
            box_info = [int(x) for x in splits[:8]]
        
            box_info.append(np.max(box_info[0::2]) - np.min(box_info[0::2]))
            box_info.append(np.max(box_info[1::2]) - np.min(box_info[1::2]))
            boxes.append([int(x) for x in box_info])
            labels.append(node_labels.index(splits[9]))
            # labels.append(0)

    return np.array(texts), np.array(text_lengths), np.array(boxes), np.array(labels)


def _prepapre_pipeline(boxes, edge_data, text, text_length):
    box_min = boxes.min(0)
    box_max = boxes.max(0)

    boxes = (boxes - box_min) / (box_max - box_min)
    boxes = (boxes - 0.5) / 0.5

    edge_min = edge_data.min(0)
    edge_max = edge_data.max(0)

    edge_data = (edge_data - edge_min) / (edge_max - edge_min)
    edge_data = (edge_data - 0.5) / 0.5

    return boxes, edge_data, text, text_length

def load_data(annotation_file):
    texts, text_lengths, boxes, labels = _load_annotation(annotation_file)

    origin_boxes = boxes
    node_nums = text_lengths.shape[0]
    src = []
    dst = []
    edge_data = []
    for i in range(node_nums):
        for j in range(node_nums):
            if i == j:
                continue
                
            edata = []
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
    

    boxes, edge_data, text, text_length = _prepapre_pipeline(boxes, edge_data, texts, text_lengths)

    boxes = torch.from_numpy(boxes).float()
    edge_data = torch.from_numpy(edge_data).float()

    tab_sizes_n = g.number_of_nodes()
    tab_snorm_n = torch.FloatTensor(tab_sizes_n, 1).fill_(1./float(tab_sizes_n))
    snorm_n = tab_snorm_n.sqrt()  

    tab_sizes_e = g.number_of_edges()
    tab_snorm_e = torch.FloatTensor(tab_sizes_e, 1).fill_(1./float(tab_sizes_e))
    snorm_e = tab_snorm_e.sqrt()

    max_length = text_lengths.max()
    new_text = [np.expand_dims(np.pad(t, (0, max_length - t.shape[0]), 'constant'), axis=0) for t in text]
    texts = np.concatenate(new_text)

    labels = torch.from_numpy(np.array(labels))
    texts = torch.from_numpy(np.array(texts))
    text_length = torch.from_numpy(np.array(text_length))

    graph_node_size = [g.number_of_nodes()]
    graph_edge_size = [g.number_of_edges()]

    return g, labels, boxes, edge_data, snorm_n, snorm_e, texts, text_length, origin_boxes, annotation_file, graph_node_size, graph_edge_size


def load_gate_gcn_net(device, checkpoint_path):
    net_params = {}
    net_params['in_dim_text'] = len(alphabet)
    net_params['in_dim_node'] = 10
    net_params['in_dim_edge'] = 2
    net_params['hidden_dim'] = 512
    net_params['out_dim'] = 512
    net_params['n_classes'] = 5
    net_params['in_feat_dropout'] = 0.1
    net_params['dropout'] = 0.0
    net_params['L'] = 8
    net_params['readout'] = True
    net_params['graph_norm'] = True
    net_params['batch_norm'] = True
    net_params['residual'] = True
    net_params['device'] = 'cuda'
    net_params['OHEM'] = 3

    model = GatedGCNNet(net_params)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


node_labels = ['other', 'company', 'address', 'date', 'total']
alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ%[](){}<>&+=\'"!?~:/-@*_.,;·|#$\\^ '

def main():
    data_path = "./test/"
    image_path = "./test_image/"
    checkpoint_path = './checkpoints/test_newest.pkl'
    device = 'cuda'
    model = load_gate_gcn_net(device, checkpoint_path)

    acc_right  = [0, 0, 0, 0]
    annotation_list = os.listdir(data_path)

    for annotation_file in annotation_list:
        if 'jpg' in annotation_file:
            continue
        annotation_path = os.path.join(data_path, annotation_file)
        batch_graphs, batch_labels, batch_x, batch_e, batch_snorm_n, batch_snorm_e, text, text_length, boxes, ann_file, graph_node_size, graph_edge_size = load_data(annotation_path)

        batch_x = batch_x.to(device)  # num x feat
        batch_e = batch_e.to(device)

        text = text.to(device)
        text_length =  text_length.to(device)        
        batch_snorm_e = batch_snorm_e.to(device)
        batch_snorm_n = batch_snorm_n.to(device)         # num x 1

        batch_scores = model.forward(batch_graphs, batch_x, batch_e, text, text_length, batch_snorm_n, batch_snorm_e)

        os.path.join(image_path, os.path.basename(ann_file).replace("txt", 'jpg'))
        if not os.path.exists(image_file):
            continue

        image = cv2.imread(image_file)

        batch_scores = batch_scores.cpu().softmax(1)
        values, pred = batch_scores.max(1)

        length = pred.shape[0]
        for i in range(length):
            if pred[i] == batch_labels[i]:
                if pred[i] == 0:
                    continue

                msg = "{}".format(node_labels[pred[i]])
                color = (0, 255, 0)
            else:
                msg = "{}-{}".format(node_labels[pred[i]], node_labels[batch_labels[i]])
                color = (0, 0, 255)

            info = boxes[i]
            box = np.array([[int(info[0]), int(info[1])], [int(info[2]), int(info[3])], [int(info[4]), int(info[5])], [int(info[6]), int(info[7])]])
            
            cv2.polylines(image, [box], 1, color)
            cv2.putText(image, msg, (int(info[0]), int(info[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

        file_name = os.path.basename(image_file)
        cv2.imwrite('./visual_test/{}'.format(file_name), image)

if __name__ == '__main__':
    main()
