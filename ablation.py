from utils import *
import torch
import torch.nn.functional as F
import wandb
import os
from torch.optim.lr_scheduler import StepLR
os.environ["WANDB_API_KEY"] = "KEY" #@param {type:"string"}

class AblationExperiment:
    def __init__(self, base_model_path, graph_dir, epochs, repetitions=1, loss_type='focal', alpha=0.75, gamma=1, seed=222):
        super(AblationExperiment, self).__init__()
        self.base_model_path = base_model_path
        self.graph_dir = graph_dir
        self.epochs = epochs
        self.repetitions = repetitions
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.seed = seed
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        set_seed(seed)
        
    def load_graph(self, graph_path):
        set_seed(self.seed)
        train_loader, test_loader, val_loader = mask_and_batch_trans(load_trans(path=graph_path))
        return train_loader, test_loader, val_loader

    def load_model(self):
        set_seed(self.seed)
        base_model = torch.load(self.base_model_path)
        model_loaded = create_model(base_model["config"], self.loss_type, self.alpha, self.gamma)
        optimizer = set_optim(base_model["config"], model_loaded)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)        

        return model_loaded.to(self.device), optimizer, scheduler

    def run_experiment(self, proj_name):
        set_seed(self.seed)
        for graph_file in os.listdir(self.graph_dir):
            graph_type = graph_file.split(".")[0].split("-")[-1]
            graph_path = os.path.join(self.graph_dir, graph_file)
            for repetition in range(self.repetitions):
                train_loader, test_loader, val_loader = self.load_graph(graph_path)
                model, optimizer, scheduler = self.load_model()
                model.to(self.device)
                
                wandb.init(project=proj_name, name=f"{graph_type}-{repetition+1}", config={"graph": graph_type}, reinit=True)
                
                for epoch in range(self.epochs):
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
                    print(f"Graph type: {graph_type} | Repetition: {repetition+1} | Epoch: {epoch} | Val F1: {val_f1}")

                    if epoch == self.epochs - 1:
                        torch.save(model, f"/PATH/{graph_type}-{repetition+1}-f1{val_f1}")

                wandb.finish()

if __name__ == "__main__":
    base_model_path=r"/PATH/top10/BASE_MODEL.pth"
    graph_dir = r"SIMILARITY_GRAPH_PATH"
    epochs = 67
    repetitions = 3
    loss_type = 'focal'
    alpha = 0.75
    gamma = 1
    experiment = AblationExperiment(base_model_path, graph_dir, epochs, repetitions, loss_type, alpha, gamma)
    experiment.run_experiment("EXP_NAME")


    #wandb sync --clean for deleting wdb cache
    #'tmux attach -t [session_name]' to view each sessio, ctrlb , d for detaching