from utils import *
import torch
import torch.nn.functional as F
import wandb
import os
import optuna
from torch.optim.lr_scheduler import StepLR
from optuna.samplers import RandomSampler
from datetime import date # type: ignore

os.environ["WANDB_API_KEY"] = "KEY" #@param {type:"string"}
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class TuningExperiment:
    def __init__(self, model_type, graph_path, proj_name, n_trials, max_epoch, max_val, seed=222, base_model_path=None):
        super(TuningExperiment, self).__init__()
        self.model_type = model_type
        self.graph_path = graph_path
        self.proj_name=proj_name
        self.seed = seed
        self.max_epoch = max_epoch
        self.max_val = max_val
        self.base_model_path=base_model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.base_model_path is not None:
            self.n_trials = 1
        else: self.n_trials = n_trials
        set_seed(self.seed)

    def load_graph(self, graph_path):
        set_seed(self.seed)
        train_loader, test_loader, val_loader = mask_and_batch_trans(load_trans(path=graph_path))
        return train_loader, test_loader, val_loader

    def create_config(self, trial):
        set_seed(self.seed)

        if self.base_model_path: # if want to reproduce and log results from an architecture
            base_model = torch.load(self.base_model_path)
            config = base_model["config"]

        else:
            config = {
                "model_type": self.model_type,
                "hidden_size": trial.suggest_categorical("hidden_size", [128, 256, 512, 768, 1024]),
                "num_layers": trial.suggest_int("num_layers", 1, 4),
                "dropout": trial.suggest_float("dropout", 0.1, 0.8),
                "activation_function": trial.suggest_categorical("activation_function", ["relu", "leaky_relu"]),
                "optimizer": trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop", "adagrad"]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
                "momentum": trial.suggest_float("momentum", 0.75, 0.99),
                "num_heads": trial.suggest_int("num_heads", 2, 6), #comment for sage
            }

        return config

    def load_model(self, config):
        set_seed(self.seed)
        model = create_model(config)
        optimizer = set_optim(config, model)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)        

        return model.to(self.device), optimizer, scheduler
    
    def save_model(self, model, optimizer, scheduler, epoch, config, val_f1):
        set_seed(self.seed)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_f1": val_f1,
            "epoch": epoch,
            "config": config,  
        }
        run_name = f"/PATH/best_models/{config['model_type']}/{wandb.run.name}_ep-{epoch}_f1-{val_f1}.pth"
        torch.save(checkpoint, run_name)

    def run_experiment(self, trial):
        set_seed(self.seed)
        train_loader, test_loader, val_loader = self.load_graph(self.graph_path)
        trial_config = self.create_config(trial)
        model, optimizer, scheduler = self.load_model(trial_config)
        model.to(self.device)
        
        wandb.init(project=self.proj_name, name="RUN_NAME-rerun2", config=trial_config, reinit=True) #name=...-rerun1
        
        patience = 10
        best_val_f1 = 0  
        epochs_since_improvement = 0 
        warmup_epochs = 50 #50=GT,GAT 75=SAGE

        for epoch in range(self.max_epoch):
            train_loss = 0
            model.train()
            for batch in train_loader:
                batch.to(self.device)
                optimizer.zero_grad()
                logits = model(batch)
                label = batch.node_label
                loader_logits = logits[batch.node_label_index]
                loss = model.loss(loader_logits, label)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                
            train_metrics = get_metrics(*test(train_loader, model, self.device), loader="train_")
            val_metrics = get_metrics(*test(val_loader, model, self.device), loader="val_")
            test_metrics = get_metrics(*test(test_loader, model, self.device), loader="test_")

            wandb.log({**train_metrics, **val_metrics, **test_metrics, "train_loss": train_loss, "epoch": epoch})
            scheduler.step()
            
            val_f1 = val_metrics["val_f1_score"]
            print(f"Epoch: {epoch} | Val F1: {val_f1}")
            
            if epoch >= warmup_epochs:
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    epochs_since_improvement = 0
                    #if val_f1 > self.max_val:
                        #self.save_model(model, optimizer, scheduler, epoch, trial_config, val_f1)
                        #print(f"Saving model at epoch {epoch} with val F1: {val_f1}")
                #else:
                    #epochs_since_improvement += 1
                    #if epochs_since_improvement >= patience:
                        #print(f"Early stopping triggered at epoch {epoch}")
                        #break

        wandb.finish()
        return val_f1

    def run_optuna(self):
        set_seed(self.seed)
        study = optuna.create_study(sampler=RandomSampler(), direction='maximize')
        study.optimize(self.run_experiment, n_trials=self.n_trials)

if __name__ == "__main__":
    model_type = "gat" #"graphtransformer", "gat", "graphsage"
    graph_path = r"SIMILARITY_GRAPH_PATH"
    proj_name = "NAME"
    n_trials = 1
    max_epoch = 124
    max_val=0.54
    seed = 222
    base_model_path=r"/PATH/top10/BASE_MODEL.pth"
    
    experiment = TuningExperiment(model_type, graph_path, proj_name, n_trials, max_epoch, max_val, seed, base_model_path) #, base_model_path if reproduce results
    experiment.run_optuna()

    #wandb sync --clean for deleting wdb cache
    #'tmux attach -t [session_name]' to view each sessio, ctrlb , d for detaching
    # if main gpu heavily utilized, use export CUDA_VISIBLE_DEVICES=1 in terminal
