"""
The provided code defines a SiameseBERT model for text similarity tasks using the BERT architecture from the transformers library.

The SiameseBERT class inherits from nn.Module and is composed of a BERT model followed by a single linear layer and a sigmoid activation function. 
The BERT model is loaded from the pre-trained "bert-base-uncased" checkpoint.

The SiameseBERTDataset class is a custom dataset class that takes in a list of tuples, each tuple containing two text strings and a label indicating whether the texts are similar or not. 
It tokenizes the two text strings using the BertTokenizer and returns the resulting input_ids, attention_mask, and label tensors for each tuple.

The Trainer class is responsible for training and evaluating the SiameseBERT model. 
It takes in the SiameseBERT model instance, train and validation dataset instances, batch size, maximum number of epochs, learning rate, and device type (CPU or GPU) as input arguments.

The train and evaluate methods are implemented in the Trainer class. 
The train method trains the model using the train dataset and prints the training and validation loss and accuracy at each epoch. 
The evaluate method evaluates the model using the validation dataset and returns the average loss and accuracy.

The AdamW optimizer and the CosineEmbeddingLoss loss function are used for training the model. The history dictionary keeps track of the training and validation loss and accuracy at each epoch.
"""


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import BertModel, BertTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

folderpath_model = "/data/cmokashi/bc2gm/named-entity-recognition/output/BC2GM"

class SiameseBERT(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=128, dropout_prob=0.1):
        super(SiameseBERT, self).__init__()

        # BERT model for token embeddings
        self.bert = BertModel.from_pretrained(folderpath_model)
        
        # Feed-forward network for projecting BERT embeddings
        # self.brother_fc = nn.Sequential(
        #     nn.Linear(embedding_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_prob),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout_prob),
        #     nn.Linear(hidden_dim, 1)
        # )
        self.brother_fc = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.BatchNorm1d(128),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(16, 2)
        )
        
        for module in self.brother_fc.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
        
        self.final_fc = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        pooled_output_1 = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1).pooler_output
        pooled_output_2 = self.bert(input_ids=input_ids_2, attention_mask=attention_mask_2).pooler_output
        # embedding_1 = self.brother_fc(pooled_output_1.squeeze(dim=1))
        # embedding_2 = self.brother_fc(pooled_output_2.squeeze(dim=1))
        # cosine_similarity = F.cosine_similarity(embedding_1, embedding_2)
        # normalized_distance = 1 - ((cosine_similarity + 1) / 2)
        
        output_1 = self.brother_fc(pooled_output_1.squeeze(dim=1))
        output_2 = self.brother_fc(pooled_output_2.squeeze(dim=1))
        output = self.final_fc((output_1 - output_2) ** 2)
        
        return output.squeeze()

    def predict_proba(self, dataloader):
        self.eval()
        # correct = 0
        # total = 0
        y_pred = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader)):
                batch = (item.to(device) for item in batch)
                input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = batch
                # input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = input_ids_1.cuda(), attention_mask_1.cuda(), input_ids_2.cuda(), attention_mask_2.cuda(), labels.cuda()
                # input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = input_ids_1.to(device), attention_mask_1.to(device), input_ids_2.to(device), attention_mask_2.to(device), labels.to(device)
                # input_ids_1 = batch['input_ids_1'].cuda()
                # attention_mask_1 = batch['attention_mask_1'].cuda()
                # input_ids_2 = batch['input_ids_2'].cuda()
                # attention_mask_2 = batch['attention_mask_2'].cuda()
                # labels = batch['label'].cuda()
                outputs = self(input_ids_1.unsqueeze(0), attention_mask_1.unsqueeze(0), input_ids_2.unsqueeze(0), attention_mask_2.unsqueeze(0))
                # predicted = torch.round(outputs)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                y_pred.append(outputs.detach().cpu().numpy())
        return y_pred
    
class SiameseBERTEval:
    def __init__(self, model):
        self.model = model
        
    def predict_proba(self, dataloader):
        self.eval()
        # correct = 0
        # total = 0
        y_pred = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = batch
                input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = input_ids_1.cuda(), attention_mask_1.cuda(), input_ids_2.cuda(), attention_mask_2.cuda(), labels.cuda()
                # input_ids_1 = batch['input_ids_1'].cuda()
                # attention_mask_1 = batch['attention_mask_1'].cuda()
                # input_ids_2 = batch['input_ids_2'].cuda()
                # attention_mask_2 = batch['attention_mask_2'].cuda()
                # labels = batch['label'].cuda()
                outputs = self(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                # predicted = torch.round(outputs)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                y_pred.append(outputs.detach().cpu().numpy())
        return np.concatenate(y_pred)
    
class SiameseBERTDataset(Dataset):
    def __init__(self, text1, text2, label, max_seq_len=128):
        self.text1 = text1
        self.text2 = text2
        self.label = label
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(folderpath_model)
    
    def __len__(self):
        return len(self.text1)
    
    def __getitem__(self, index):
        text1, text2, label = self.text1[index], self.text2[index], self.label[index]
        # if np.isnan(text1):
        #     text1 = ""
        # if np.isnan(text2):
        #     text2 = ""
        # text1, text2 = str(text1), str(text2)
        encoding_1 = self.tokenizer(text1, padding="max_length", max_length=self.max_seq_len, return_tensors="pt", return_token_type_ids=False, truncation=True)
        encoding_2 = self.tokenizer(text2, padding="max_length", max_length=self.max_seq_len, return_tensors="pt", return_token_type_ids=False, truncation=True)
        input_ids_1, attention_mask_1 = encoding_1["input_ids"], encoding_1["attention_mask"]
        input_ids_2, attention_mask_2 = encoding_2["input_ids"], encoding_2["attention_mask"]
        
        return input_ids_1.squeeze(), attention_mask_1.squeeze(), input_ids_2.squeeze(), attention_mask_2.squeeze(), torch.tensor(label, dtype=torch.float32)

class SiameseBERTTrainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size=16, lr=2e-5, epochs=10):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        
        total_steps = len(self.train_dataloader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        best_val_loss = float('inf')
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}")
            
            train_loss, train_acc = self._train_epoch(device, epoch=epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            val_loss, val_acc = self._eval_epoch(device, epoch=epoch)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join("results/siameseBERT", "best_model.pth"))
            
            # print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            self.scheduler.step()
            
            # plot and save the loss graph
            fig = plt.figure()
            plt.plot(train_losses, label="train loss")
            plt.plot(val_losses, label="val loss")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.savefig(os.path.join("results/siameseBERT", f"loss_plot.png"))
            plt.close(fig)
            
        return train_losses, train_accs, val_losses, val_accs
            
    def _train_epoch(self, device, epoch=0):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        THRESHOLD = 0.5
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = batch
            input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = input_ids_1.to(device), attention_mask_1.to(device), input_ids_2.to(device), attention_mask_2.to(device), labels.to(device)
            
            self.optimizer.zero_grad()
            
            # Calculate distance between embeddings and compute loss
            output = self.model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            loss = self.loss_fn(output, labels)
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            
            running_loss += loss.item()
            total += labels.size(0)
            correct += ((output < THRESHOLD) == labels).sum().item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx + 1}: Train Loss: {running_loss/(batch_idx + 1):.4f}, Train Acc: {(100*correct/total):.2f}%")
        
        train_loss = running_loss / len(self.train_dataloader)
        train_acc = 100 * correct / total
        writer.add_scalar('Loss/Train', loss.item(), epoch)
        
        return train_loss, train_acc
    
    def _eval_epoch(self, device, epoch=0):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        THRESHOLD = 0.5
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = batch
                input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = input_ids_1.to(device), attention_mask_1.to(device), input_ids_2.to(device), attention_mask_2.to(device), labels.to(device)
                
                distance = self.model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                loss = self.loss_fn(distance, labels)
                
                running_loss += loss.item()
                total += labels.size(0)
                correct += ((distance < THRESHOLD) == labels.byte()).sum().item()
                
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"Batch {batch_idx + 1}: Val Loss: {running_loss/(batch_idx + 1):.4f}, Val Acc: {(100*correct/total):.2f}%")
            
        val_loss = running_loss / len(self.val_dataloader)
        val_acc = 100 * correct / total
        writer.add_scalar('Loss/Validation', loss.item(), epoch)
            
        return val_loss, val_acc

if __name__ == "__main__":
    # Load datasets
    df_train = pd.read_csv("results/datasets/train.csv")
    df_train["is_success"] = df_train.apply(lambda row: 1 if row["entrez_id_text"] == row["entrez_id_term"] else 0, axis=1).tolist()
    
    # Split the data into training and validation sets
    train_data, valid_data = train_test_split(df_train[["text", "terms", "is_success"]], test_size=0.2, random_state=2023)
    train_data = train_data[~(train_data["text"].isnull() | train_data["terms"].isnull())]
    valid_data = valid_data[~(valid_data["text"].isnull() | valid_data["terms"].isnull())]
    
    train_dataset = SiameseBERTDataset(text1=train_data["text"].to_numpy(), text2=train_data["terms"].to_numpy(), label=train_data["is_success"].to_numpy())
    valid_dataset = SiameseBERTDataset(text1=valid_data["text"].to_numpy(), text2=valid_data["terms"].to_numpy(), label=valid_data["is_success"].to_numpy())
    
    log_dir = "results/siameseBERT/logs/"  # specify the log directory
    writer = SummaryWriter(log_dir)

    
    # Create model instance
    model = SiameseBERT()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    
    # Define hyperparameters
    batch_size = 16
    # lr = 2e-5
    lr = 0.005
    n_epochs = 20
    max_seq_len = 128
    
    
    # Initialize trainer object
    trainer = SiameseBERTTrainer(model=model, 
                      train_dataset=train_dataset, 
                      val_dataset=valid_dataset,
                      batch_size=batch_size,
                      lr=lr,
                      epochs=n_epochs)
    trainer.train()
    
    torch.save(model.state_dict(), os.path.join("results/siameseBERT", "siamese_bert_model.pth"))
