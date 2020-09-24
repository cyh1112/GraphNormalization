import argparse
import random
from datetime import datetime

import numpy as np
import torch
from dataset import SROIEDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from model import GATNet
from gated_gcn import GatedGCNNet
# from my_gcn import GatedGCNNet
# from iat_gcn import GatedGCNNet

torch.autograd.set_detect_anomaly(True)

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

def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    # epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0

    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e, text, text_length, graph_node_size, graph_edge_size) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)

        text = text.to(device)
        text_length =  text_length.to(device)        
        batch_snorm_e = batch_snorm_e.to(device)
        batch_labels = batch_labels.to(device)
        batch_snorm_n = batch_snorm_n.to(device)         # num x 1

        optimizer.zero_grad()
        batch_scores = model.forward(batch_graphs, batch_x, batch_e, text, text_length, batch_snorm_n, batch_snorm_e, graph_node_size, graph_edge_size)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print("epoch:{}, train loss:{}".format(epoch, epoch_loss))
    return epoch_loss, optimizer


def evaluate_train_network(key, model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    all_batch_scores = []
    all_batch_labels = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e, text, text_length, graph_node_size, graph_edge_size) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
            batch_e = batch_graphs.edata['feat'].to(device)

            text = text.to(device)
            text_length =  text_length.to(device)        
            batch_snorm_e = batch_snorm_e.to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_n = batch_snorm_n.to(device)         # num x 1

            batch_scores = model.forward(batch_graphs, batch_x, batch_e, text, text_length, batch_snorm_n, batch_snorm_e, graph_node_size, graph_edge_size)
            loss = model.loss(batch_scores, batch_labels) 
            
            epoch_test_loss += loss.detach().item()
            all_batch_scores.append(batch_scores)
            all_batch_labels.append(batch_labels)
        epoch_test_loss /= (iter + 1)

        recall, precision, F1_score = accuracy(torch.cat(all_batch_scores, 0), torch.cat(all_batch_labels, 0))

        results = []
        for i in range(5):
            results.append(recall[i])
            results.append(precision[i])
            results.append(F1_score[i])

        write_log(key, "epoch:{}, loss: {}, other:{:.3f}|{:.3f}|{:.3f}, company:{:.3f}|{:.3f}|{:.3f}, address:{:.3f}|{:.3f}|{:.3f}, date:{:.3f}|{:.3f}|{:.3f}, total:{:.3f}|{:.3f}|{:.3f}".format(epoch, epoch_test_loss, *results))        

def evaluate_test_network(key, model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    all_batch_scores = []
    all_batch_labels = []

    acc_right  = [0, 0, 0, 0]
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e, text, text_length, graph_node_size, graph_edge_size) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
            batch_e = batch_graphs.edata['feat'].to(device)

            text = text.to(device)
            text_length =  text_length.to(device)        
            batch_snorm_e = batch_snorm_e.to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_n = batch_snorm_n.to(device)         # num x 1

            batch_scores = model.forward(batch_graphs, batch_x, batch_e, text, text_length, batch_snorm_n, batch_snorm_e, graph_node_size, graph_edge_size)
            loss = model.loss(batch_scores, batch_labels) 
            
            epoch_test_loss += loss.detach().item()
            # all_batch_scores.append(batch_scores)
            # all_batch_labels.append(batch_labels)
            _, _, F1_score = accuracy(batch_scores, batch_labels)

            
            for i in range(1, len(F1_score)):
                if F1_score[i] == 1:
                    acc_right[i-1] += 1

        epoch_test_loss /= (iter + 1)

        acc_right = [i / len(data_loader) for i in acc_right] 
        acc_right.append(np.mean(acc_right))

        write_log(key, "epoch:{}, test loss: {}, company:{:.3f}, address:{:.3f}, date:{:.3f}, total:{:.3f}, mean:{:.3f}".format(epoch, epoch_test_loss, *acc_right))        

        return epoch_test_loss

def write_log(key, msg):
    print(msg)
    file = "./logs3/{}.log".format(key)
    with open(file, 'a') as f:
        f.writelines(str(datetime.now()) + ", " + msg + "\n")

def load_gat_net(device, alphabet):
    net_params = {}
    net_params['in_dim'] = len(alphabet)
    net_params['hidden_dim'] = 64
    net_params['out_dim'] = 64
    net_params['n_classes'] = 5
    net_params['n_heads'] = 8
    net_params['in_feat_dropout'] = 0.1
    net_params['dropout'] = 0.1
    net_params['L'] = 5
    net_params['readout'] = True
    net_params['graph_norm'] = True
    net_params['batch_norm'] = True
    net_params['residual'] = True
    net_params['device'] = device

    model = GATNet(net_params)
    model = model.to(device)
    return model

def load_gate_gcn_net(device, alphabet, checkpoint_path=None):
    net_params = {}
    net_params['in_dim_text'] = len(alphabet)
    net_params['in_dim_node'] = 10
    net_params['in_dim_edge'] = 2
    net_params['hidden_dim'] = 512
    net_params['out_dim'] = 512
    net_params['n_classes'] = 5
    net_params['dropout'] = 0.
    net_params['L'] = 4
    net_params['readout'] = True
    net_params['graph_norm'] = True
    net_params['batch_norm'] = True
    net_params['residual'] = True
    net_params['device'] = 'cuda'
    net_params['OHEM'] = 3

    model = GatedGCNNet(net_params)

    if checkpoint_path is not None:
        print("resume checkpoint from {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    return model


def show_graph(dataset1, dataset2):
    nodes = []
    edges = []
    for data in dataset1:
        nodes.append(data[0].number_of_nodes())
        edges.append(data[0].number_of_edges())

    for data in dataset2:
        nodes.append(data[0].number_of_nodes())
        edges.append(data[0].number_of_edges())

    print(sum(nodes), min(nodes), max(nodes))
    print(sum(edges), min(edges), max(edges))

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--batch_size', default=24, help="Please give a value for batch_size")
    parser.add_argument('--train_dataset', default='./data/train', help="Please give a value for dataset name")
    parser.add_argument('--test_dataset', default='./data/test', help="Please give a value for dataset name")
    parser.add_argument('--split', default='\t', help="Please give a value for dataset name")
    parser.add_argument('--labels', default=['other', 'company', 'address', 'date', 'total'], help="Please give a value for dataset name")
    parser.add_argument('--alphabet', default='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ%[](){}<>&+=\'"!?~:/-@*_.,;·|#$\\^ ', help="Please give a value for dataset name")
    parser.add_argument('--seed', default=0, help="Please give a value for seed")
    parser.add_argument('--init_lr', default=0.001, help="Please give a value for init_lr")
    parser.add_argument('--epochs', default=1000, help="Please give a value for epochs")
    parser.add_argument('--key', default='test', help="Please give a value for epochs")
    parser.add_argument('--node_dropout', default=0., help="Please give a value for epochs")
    
    args = parser.parse_args()

    args.seed = int(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = 'cuda'
    model = load_gate_gcn_net(device, args.alphabet)

    train_dataset = SROIEDataset(data_dir=args.train_dataset, split=args.split, labels=args.labels, alphabet=args.alphabet, dropout=args.node_dropout)
    test_dataset = SROIEDataset(data_dir=args.test_dataset, split=args.split, labels=args.labels, alphabet=args.alphabet)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate)

    print(len(train_dataset), len(test_dataset))

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    #                                                  factor=0.1,
    #                                                  patience=20,
    #                                                  verbose=True)

    for epoch in range(args.epochs):
        epoch_loss, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)

        # evaluate_train_network(args.key, model, device, train2_loader, epoch)
        epoch_test_loss = evaluate_test_network(args.key, model, device, test_loader, epoch)
        # scheduler.step(epoch_test_loss)

        torch.save(model.state_dict(), './checkpoints/{}_newest.pkl'.format(args.key))


if __name__ == '__main__':
    main()
