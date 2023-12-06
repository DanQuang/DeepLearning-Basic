import torch
from torch import nn, optim
import os
from tqdm.auto import tqdm
from model import Model
from data_utils import load_data
from evaluate import evaluate

class Train_Task:
    def __init__(self, config):
        self.num_epochs = config["num_epochs"]
        self.learning_rate = config["learning_rate"]
        self.best_metric = config["best_metric"]
        self.save_path = config["save_path"]
        self.patience = config["patience"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_classes = config["num_classes"]
        self.model_name = config["model"]
        self.model = Model.Model(config).to(self.device)
        self.dataloader = load_data.Load_Data(config)
        self.loss = nn.CrossEntropyLoss()
        self.optim = optim.SGD(self.model.parameters(), lr= self.learning_rate, momentum= 0.5)

    def train(self):
        train, dev = self.dataloader.load_train_dev()

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        last_model = f"{self.model_name}_last_model.pth"
        best_model = f"{self.model_name}_best_model.pth"

        if os.path.exists(os.path.join(self.save_path, last_model)):
            checkpoint = torch.load(os.path.join(self.save_path, last_model))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optim.load_state_dict(checkpoint["optim_state_dict"])
            print("Load the last model")
            initial_epoch = checkpoint["epoch"] + 1
            print(f"Continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("First time training!!!")

        if os.path.exists(os.path.join(self.save_path, best_model)):
            checkpoint = torch.load(os.path.join(self.save_path, best_model))
            best_score = checkpoint['score']
        else:
            best_score = 0.

        threshold = 0

        self.model.train()
        for epoch in range(initial_epoch, initial_epoch + self.num_epochs):
            train_loss = 0.
            valid_acc = 0.
            valid_f1=0.
            valid_precision=0.
            valid_recall=0.

            for _, (X, y) in tqdm(enumerate(train)):
                self.optim.zero_grad()
                X, y = X.to(self.device), y.to(self.device)

                # Forward
                y_logits = self.model(X)
                loss = self.loss(y_logits, y)
                train_loss += loss

                # Backward
                loss.backward()
                self.optim.step()

            with torch.inference_mode():
                for _, (X, y) in tqdm(enumerate(dev)):
                    X, y = X.to(self.device), y.to(self.device)
                    y_logits = self.model(X)
                    y_preds = y_logits.argmax(dim = -1)

                    acc, prec, recall, f1 = evaluate.compute_score(self.num_classes, y, y_preds)

                    valid_acc += acc
                    valid_precision += prec
                    valid_recall += recall
                    valid_f1 += f1

                train_loss /= len(train)
                valid_acc /= len(dev)
                valid_precision /= len(dev)
                valid_recall /= len(dev)
                valid_f1 /= len(dev)

                print(f"Epoch {epoch + 1}/{initial_epoch + self.num_epochs}")
                print(f"Train loss: {train_loss:.5f}")
                print(f"valid acc: {valid_acc:.4f} | valid f1: {valid_f1:.4f} | valid precision: {valid_precision:.4f} | valid recall: {valid_recall:.4f}")

                if self.best_metric == 'accuracy':
                    score= valid_acc
                if self.best_metric == 'f1':
                    score= valid_f1
                if self.best_metric == 'precision':
                    score= valid_precision
                if self.best_metric == 'recall':
                    score= valid_recall

                # save last model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'score': score
                }, os.path.join(self.save_path, last_model))

                # save the best model
                if epoch > 0 and score < best_score:
                    threshold += 1
                else:
                    threshold = 0

                if score > best_score:
                    best_score = score
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optim_state_dict': self.optim.state_dict(),
                        'score':score
                    }, os.path.join(self.save_path, best_model))
                    print(f"Saved the best model with {self.best_metric} of {score:.4f}")

            # early stopping
            if threshold >= self.patience:
                print(f"Early stopping after epoch {epoch + 1}")
                break