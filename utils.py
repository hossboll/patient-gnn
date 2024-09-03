import random
import os
import pandas as pd
import wandb
import numpy as np
import torch
import pickle as pkl
from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset
from torch.utils.data import DataLoader
from deepsnap.batch import Batch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import SAGEnorm, GATnorm, GraphTransformernorm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve, confusion_matrix
from datetime import date
from torch.optim.lr_scheduler import StepLR
os.environ["WANDB_API_KEY"] = "KEY" #@param {type:"string"}

GRAPH_PATH_TRANS=r"PATH_SIMILARITY_GRAPH"

def print_bestparam(sweep_path, metric='val_macro_f1'):
    api = wandb.Api()
    sweep = api.sweep(sweep_path)
    # best run parameters
    best_run = sweep.best_run(order=metric)
    best_parameters = best_run.config, best_run.name
    print(f"Best {sweep}: {best_parameters}")

# reproducibility
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_unmasked_node_ids(loader):
    unmasked_node_ids = []

    for graph in loader:
        unmasked_nodes = graph.node_label_index.tolist()
        patient_ids = graph.patient_id.tolist()

        for idx in unmasked_nodes:
            idx = int(idx)
            patient_id = patient_ids[idx]
            unmasked_node_ids.append(patient_id)
    
    return unmasked_node_ids


def class_weights(H_graph):
    node_labels = H_graph.node_label

    # calculate class weights
    class_counts = torch.bincount(node_labels)
    weights = 1. / class_counts.float()
    weights = weights / weights.sum()
    return weights


def load_trans(path=GRAPH_PATH_TRANS):
    # load networkx graph
    with open(path, 'rb') as file:
        print("Loading TRANSDUCTIVE graph...")
        G = pkl.load(file)
    print("The TRANSDUCTIVE graph has {} nodes and {} edges".format(G.number_of_nodes(), G.number_of_edges()))

    # change graph feature names to match deepsnap
    for node in G.nodes():
        # rename 'features' to 'node_feature' and 'label' to 'node_label'
        G.nodes[node]['node_feature'] = G.nodes[node].pop('features', None)
        G.nodes[node]['node_label'] = G.nodes[node].pop('label', None)

    for edge in G.edges():
        # rename 'weight' to 'edge_feature'
        G.edges[edge]['edge_feature'] = G.edges[edge].pop('weight', None)

    # create deepsnap graph from networkx graph
    H = Graph(G)
    
    # converting lists to tensors
    H.edge_feature = torch.tensor(H.edge_feature, dtype=torch.float)
    H.edge_index = torch.tensor(H.edge_index, dtype=torch.long)
    H.edge_label_index = torch.tensor(H.edge_label_index, dtype=torch.long)
    #print(H.node_feature[0])
    H.node_feature = torch.tensor(H.node_feature, dtype=torch.float)
    H.node_label = torch.tensor(H.node_label, dtype=torch.long)
    H.node_label_index = torch.tensor(H.node_label_index, dtype=torch.long)
    H.patient_id = torch.tensor(H.patient_id, dtype=torch.long)
    print(f"Transductive graph is ready: {H}")

    return H

def mask_and_batch_trans(H_graph):
    dataset = GraphDataset([H_graph], task="node")
    graph_train, graph_val, graph_test = dataset.split(transductive=True, split_ratio = [0.6, 0.2, 0.2]) # if inductive, use transductive=False

    # similar to cora transductive split used in example notebook
    train_loader = DataLoader(graph_train, collate_fn=Batch.collate(), batch_size=1)
    test_loader = DataLoader(graph_test, collate_fn=Batch.collate(), batch_size=1)
    val_loader = DataLoader(graph_val, collate_fn=Batch.collate(), batch_size=1)
    
    print(f"Transductive batches are ready.")
    
    return train_loader, test_loader, val_loader

def mask_and_batch_tran_interpret(H_graph):
    dataset = GraphDataset([H_graph], task="node")
    graph_train, graph_val, graph_test = dataset.split(transductive=True, split_ratio = [0.6, 0.2, 0.2]) # if inductive, use transductive=False

    # similar to cora transductive split used in example notebook
    train_loader = DataLoader(graph_train, collate_fn=Batch.collate(), batch_size=1)
    test_loader = DataLoader(graph_test, collate_fn=Batch.collate(), batch_size=1)
    val_loader = DataLoader(graph_val, collate_fn=Batch.collate(), batch_size=1)
    dataset_loader=DataLoader(dataset, collate_fn=Batch.collate(), batch_size=1)
    
    print(f"Transductive batches are ready.")
    
    return train_loader, test_loader, val_loader, dataset_loader

def create_model(config, loss_type="bce", alpha=None, gamma=None):
    if config["model_type"] == 'gat':
        model = GATnorm(config["hidden_size"], config["num_layers"], config["dropout"], config["activation_function"], config["num_heads"], loss_type=loss_type, alpha=alpha, gamma=gamma)
    elif config["model_type"] == 'graphsage':
        model = SAGEnorm(config["hidden_size"], config["num_layers"], config["dropout"], config["activation_function"], loss_type=loss_type, alpha=alpha, gamma=gamma)
    elif config["model_type"] == 'graphtransformer':
        model = GraphTransformernorm(config["hidden_size"], config["num_layers"], config["dropout"], config["activation_function"], config["num_heads"], loss_type=loss_type, alpha=alpha, gamma=gamma)
    print(model)
    return model


def set_optim(config, model):
    if config["optimizer"] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    return optimizer


def train(train_loader, val_loader, args, device='cuda'):
    set_seed(22)
    model = create_model(args).to(device)
    optimizer = set_optim(args, model)
    #print(model)

    for epoch in range(300):
        train_loss = 0            
        model.train()
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            label = batch.node_label
            loader_logits = logits[batch.node_label_index]
            loss = model.loss(loader_logits, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        train_acc, _, train_probs, train_preds, train_label = test(train_loader, model, device)
        val_acc, val_loss, val_probs, val_preds, val_label = test(val_loader, model, device)

        print("Epoch {}: Train: {:.4f}, Validation: {:.4f}. Train loss: {:.4f}, Val loss: {:.4f}"
              .format(epoch + 1, train_acc, val_acc, train_loss, val_loss))
        
    return model

def test(loader, model, device='cpu'): #cpu for interpret
    model.eval()
    correct = 0
    total = 0
    loss = 0
    probs, preds, labels = [], [], []

    with torch.inference_mode():
        set_seed(22)
        for batch in loader:
            batch.to(device)
            logits = model(batch)
            label = batch.node_label
            loader_logits = logits[batch.node_label_index]
            current_loss = model.loss(loader_logits, label)
            loss += current_loss.item()
            loader_probs = torch.sigmoid(loader_logits).squeeze() 
            loader_preds = loader_probs.round().long()
            correct += loader_preds.eq(label.view_as(loader_preds)).sum().item()  
            total += batch.node_label_index.size(0)

            probs.extend(loader_probs.cpu().numpy())
            preds.extend(loader_preds.cpu().numpy())
            labels.extend(label.cpu().numpy())

    acc = correct / total
    avg_loss = loss / total  
    return acc, avg_loss, probs, preds, labels


def plot_curves(test_loader, model, device='cuda'):
    test_acc, all_probs, all_labels, all_preds = test(test_loader, model, device) # change model here for model name

    # roc auc
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # aucpr
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall, precision)

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # plots
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.subplot(1, 3, 2)
    plt.plot(recall, precision, color='green', lw=2, label='PR curve (area = %0.4f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.subplot(1, 3, 3)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [0, 1], rotation=45)
    plt.yticks(tick_marks, [0, 1])

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    # accuracy
    plt.suptitle(f'Test Accuracy: {test_acc:.4f}')

    plt.tight_layout()
    plt.show()

def check_dict_loading(loader):
    # train model
    config = {
        "model_type": "graphsage",
        "hidden_size": 256,
        "num_layers": 2,
        "dropout": 0.5,
        "activation_function": "relu",
        "optimizer": "adam",
        "learning_rate": 0.001,
    }

    model = create_model(config)
    optimizer = set_optim(config, model)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

    for epoch in range(10):
        train_loss = 0            
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch)
            label = batch.node_label
            loader_logits = logits[batch.node_label_index]
            loss = model.loss(loader_logits, label)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss}")

    # saving checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch + 1,  
        "config": config,  
    }
    torch.save(checkpoint, "test_checkpoint.pth")

    # load model
    checkpoint_path = r"PATH_TO_CHECKPOINT.pth" 
    checkpoint = torch.load(checkpoint_path)

    model_loaded = create_model(checkpoint["config"])
    model_loaded.load_state_dict(checkpoint["model_state_dict"])
    optimizer = set_optim(checkpoint["config"], model_loaded)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    model.eval()
    model_loaded.eval()
    first_batch = next(iter(loader))

    with torch.inference_mode():
        original_model_output = model(first_batch)
        loaded_model_output = model_loaded(first_batch)

    if torch.allclose(original_model_output, loaded_model_output):
        print("The loaded model matches the original model.")
    else:
        print("The loaded model DOESNT match the original model.")

def get_metrics(acc, avg_loss, probs, preds, labels, loader):    
    balanced_acc = balanced_accuracy_score(labels, preds)
    recall = recall_score(labels, preds, average='binary', zero_division=0)
    precision = precision_score(labels, preds, average='binary', zero_division=0)
    f1 = f1_score(labels, preds, average='binary', zero_division=0)
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    cm = confusion_matrix(labels, preds) # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    if loader == "train_":
        metrics = {
            f"{loader}accuracy": acc,  
            f"{loader}balanced_accuracy": balanced_acc,
            f"{loader}recall": recall,
            f"{loader}precision": precision,
            f"{loader}f1_score": f1,
            f"{loader}auroc": auroc,
            f"{loader}auprc": auprc,
            f"{loader}tn": tn,
            f"{loader}fp": fp,
            f"{loader}fn": fn,
            f"{loader}tp": tp,
        }
    if loader == "val_" or "test_":
        metrics = {
            f"{loader}loss": avg_loss,
            f"{loader}accuracy": acc,  
            f"{loader}balanced_accuracy": balanced_acc,
            f"{loader}recall": recall,
            f"{loader}precision": precision,
            f"{loader}f1_score": f1,
            f"{loader}auroc": auroc,
            f"{loader}auprc": auprc,
            f"{loader}tn": tn,
            f"{loader}fp": fp,
            f"{loader}fn": fn,
            f"{loader}tp": tp,
        }

    return metrics

def log_roc_pr(labels, probs, loader):
    probs = np.array(probs)
    probs_2d = np.vstack((1 - probs, probs)).T
    pr_plot = wandb.plot.pr_curve(labels, probs_2d, labels=None, classes_to_plot=None)
    wandb.log({f"{loader}pr_curve": pr_plot})
    roc_plot = wandb.plot.roc_curve(labels, probs_2d, labels=None, classes_to_plot=None)
    wandb.log({f"{loader}roc_curve": roc_plot})

def plot_curves_trainval(train_loader, val_loader, model_loaded, device='cuda'):
    train_acc, train_avg_loss, train_probs, train_preds, train_labels = test(train_loader, model_loaded, device)
    val_acc, val_avg_loss, val_probs, val_preds, val_labels = test(val_loader, model_loaded, device)

    train_fpr, train_tpr, _ = roc_curve(train_labels, train_probs)
    train_roc_auc = auc(train_fpr, train_tpr)
    val_fpr, val_tpr, _ = roc_curve(val_labels, val_probs)
    val_roc_auc = auc(val_fpr, val_tpr)

    train_precision, train_recall, _ = precision_recall_curve(train_labels, train_probs)
    train_pr_auc = auc(train_recall, train_precision)
    val_precision, val_recall, _ = precision_recall_curve(val_labels, val_probs)
    val_pr_auc = auc(val_recall, val_precision)

    train_cm = confusion_matrix(train_labels, train_preds)
    val_cm = confusion_matrix(val_labels, val_preds)

    #normalizing cm colors
    train_cm_normalized = train_cm.astype('float') / train_cm.sum(axis=1)[:, np.newaxis]
    val_cm_normalized = val_cm.astype('float') / val_cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(24, 6))

    plt.subplot(1, 4, 1)
    plt.plot(train_fpr, train_tpr, color='#3890D9', lw=2, label='Train (%0.4f)' % train_roc_auc)
    plt.plot(val_fpr, val_tpr, color='#DF672A', lw=2, label='Validation (%0.4f)' % val_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUROC - Graph Transformer (α = 0.75, γ = 1)')
    plt.legend(loc="lower right")

    plt.subplot(1, 4, 2)
    plt.plot(train_recall, train_precision, color='#3890D9', lw=2, label='Train (%0.4f)' % train_pr_auc)
    plt.plot(val_recall, val_precision, color='#DF672A', lw=2, label='Validation (%0.4f)' % val_pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUPRC - Graph Transformer (α = 0.75, γ = 1)')
    plt.legend(loc="lower left")

    plt.subplot(1, 4, 3)
    plt.imshow(train_cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Train confusion matrix - Graph Transformer')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(train_labels)))
    plt.xticks(tick_marks, np.unique(train_labels), rotation=45)
    plt.yticks(tick_marks, np.unique(train_labels))
    for i in range(train_cm.shape[0]):
        for j in range(train_cm.shape[1]):
            plt.text(j, i, format(train_cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if train_cm_normalized[i, j] > 0.5 else "black")

    plt.subplot(1, 4, 4)
    plt.imshow(val_cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Validation confusion matrix - Graph Transformer')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(val_labels)))
    plt.xticks(tick_marks, np.unique(val_labels), rotation=45)
    plt.yticks(tick_marks, np.unique(val_labels))
    for i in range(val_cm.shape[0]):
        for j in range(val_cm.shape[1]):
            plt.text(j, i, format(val_cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if val_cm_normalized[i, j] > 0.5 else "black")

    plt.tight_layout()
    plt.show()


def plot_curves_test(test_loader, model_loaded, device='cuda'):
    test_acc, test_avg_loss, test_probs, test_preds, test_labels = test(test_loader, model_loaded, device)

    test_fpr, test_tpr, _ = roc_curve(test_labels, test_probs)
    test_roc_auc = auc(test_fpr, test_tpr)

    test_precision, test_recall, _ = precision_recall_curve(test_labels, test_probs)
    test_pr_auc = auc(test_recall, test_precision)

    test_cm = confusion_matrix(test_labels, test_preds)
    test_cm_normalized = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.plot(test_fpr, test_tpr, color='#3890D9', lw=2, label='Test (%0.4f)' % test_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('AUROC - Graph Transformer (α = 0.75, γ = 1)')
    plt.legend(loc="lower right")

    plt.subplot(1, 3, 2)
    plt.plot(test_recall, test_precision, color='#3890D9', lw=2, label='Test (%0.4f)' % test_pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUPRC - Graph Transformer (α = 0.75, γ = 1)')
    plt.legend(loc="lower left")

    plt.subplot(1, 3, 3)
    plt.imshow(test_cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Test confusion matrix - Graph Transformer')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(test_labels)))
    plt.xticks(tick_marks, np.unique(test_labels), rotation=45)
    plt.yticks(tick_marks, np.unique(test_labels))
    for i in range(test_cm.shape[0]):
        for j in range(test_cm.shape[1]):
            plt.text(j, i, format(test_cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if test_cm_normalized[i, j] > 0.5 else "black")

    plt.tight_layout()
    plt.show()