import torch
import os
from tqdm.auto import tqdm
from model import LeNet
from data_utils import load_data
from evaluate import evaluate

class Test_Task:
    def __init__(self, config):
        self.save_path = config["save_path"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = config["model"]
        self.num_classes = config["num_classes"]
        if self.model_name == "LeNet":
            self.model = LeNet.LeNet(config).to(self.device)

        self.dataloader = load_data.Load_Data(config)

    def predict(self):
        test = self.dataloader.load_test()
        best_model = f"{self.model_name}_best_model.pth"

        if os.path.join(self.save_path, best_model):
            checkpoint = torch.load(os.path.join(self.save_path, best_model))
            self.model.load_state_dict(checkpoint["model_state_dict"])

            ev_acc = 0.
            ev_prec = 0.
            ev_recall = 0.
            ev_f1 = 0.

            self.model.eval()
            with torch.inference_mode():
                for _, (X, y) in tqdm(enumerate(test)):
                    X, y = X.to(self.device), y.to(self.device)
                    y_logits = self.model(X)
                    y_preds = y_logits.argmax(dim = -1)

                    acc, prec, recall, f1 = evaluate.compute_score(self.num_classes, y, y_preds)

                    ev_acc += acc
                    ev_prec += prec
                    ev_recall += recall
                    ev_f1 += f1

                ev_acc /= len(test)
                ev_prec /= len(test)
                ev_recall /= len(test)
                ev_f1 /= len(test)

                print(f"test acc: {ev_acc:.4f} | test f1: {ev_f1:.4f} | test precision: {ev_prec:.4f} | test recall: {ev_recall:.4f}")

        else:
            print(f"Chưa train model {self.model_name}!!")
