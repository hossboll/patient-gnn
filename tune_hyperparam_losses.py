from utils import *
import torch
import torch.nn.functional as F
import wandb
import os
import optuna
from torch.optim.lr_scheduler import StepLR
from optuna.samplers import RandomSampler
import glob
os.environ["WANDB_API_KEY"] = "KEY" #@param {type:"string"}

class LossExperiment:
    def __init__(self, model_list, graph_path, proj_name, n_trials, max_epoch, max_val, seed=222, loss_type='bce', base_model_path=None):
        super(LossExperiment, self).__init__()
        self.model_list = glob.glob(model_list)
        self.graph_path = graph_path
        self.proj_name=proj_name
        self.n_trials = n_trials
        self.loss_type=loss_type
        self.seed = seed
        self.max_epoch = max_epoch
        self.max_val = max_val
        self.base_model_path=base_model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(self.seed)

    def load_graph(self, graph_path):
        set_seed(self.seed)
        train_loader, test_loader, val_loader = mask_and_batch_trans(load_trans(path=graph_path))
        return train_loader, test_loader, val_loader

    def create_config(self, trial):
        set_seed(self.seed)
        #loss_type = trial.suggest_categorical("loss_type", self.loss_type)
        model_type = trial.suggest_categorical("model_type", self.model_list)
        base_model=model_type.split("/")[-1].split("_")[0]

        if self.base_model_path: # if want to reproduce and log results from an architecture
            base_model = torch.load(self.base_model_path)
            config = base_model["config"]

        else:
            config = {
            "loss_type": self.loss_type,
            "model_type": model_type,
            "base_model": base_model,
            "alpha": trial.suggest_categorical("alpha", [1.74, 3.48, 5.22, 6.96]), # [0.5, 0.75, 0.9]
            #"gamma": trial.suggest_categorical("gamma", [1, 2, 3, 4]), # comment for balanced bce
            }
        return config

    def load_model(self, config):
        set_seed(self.seed)
        checkpoint = torch.load(config['model_type'])
        
        model_loaded = create_model(checkpoint['config'], config['loss_type'], config['alpha']) #, config["gamma"])
        optimizer = set_optim(checkpoint['config'], model_loaded)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)        

        return model_loaded.to(self.device), optimizer, scheduler
    
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
        run_name = f"/PATH/{self.loss_type}/{config['base_model']}_{wandb.run.name}_ep-{epoch}_f1-{val_f1}.pth"
        torch.save(checkpoint, run_name)
        

    def run_experiment(self, trial):
        set_seed(self.seed)
        train_loader, test_loader, val_loader = self.load_graph(self.graph_path)
        trial_config = self.create_config(trial)
        model, optimizer, scheduler = self.load_model(trial_config)
        model.to(self.device)
        
        wandb.init(project=self.proj_name, config=trial_config, reinit=True)
            
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

            wandb.log({**train_metrics, **val_metrics, **test_metrics, "train_loss": train_loss})
            scheduler.step()
            
            val_f1 = val_metrics["val_f1_score"]
            print(f"Epoch: {epoch} | Val F1: {val_f1}")

            if val_f1 > self.max_val:
                self.save_model(model, optimizer, scheduler, epoch, trial_config, val_f1)
                print(f"Saving model at epoch {epoch} with val F1: {val_f1}")
            
        wandb.finish()
        return val_f1

    def run_optuna(self):
        set_seed(self.seed)
        study = optuna.create_study(sampler=RandomSampler(), direction='maximize')
        study.optimize(self.run_experiment, n_trials=self.n_trials)

if __name__ == "__main__":
    model_list = r"/PATH/top10/*"
    graph_path = r"/SIMILARITY_GRAPH_PATH"
    proj_name = "PROJECT"
    n_trials = 40 #40 bbce, 120 focal
    max_epoch = 130
    max_val = 0.56 # check
    seed = 222
    loss_type = "balanced_bce"
    #base_model_path=r"/PATH/top10/BASE_MODEL.pth"
    
    experiment = LossExperiment(model_list, graph_path, proj_name, n_trials, max_epoch, max_val, seed, loss_type)
    experiment.run_optuna()


    #wandb sync --clean for deleting wdb cache
    #'tmux attach -t [session_name]' to view each sessio, ctrlb , d for detaching