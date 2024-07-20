from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
# import pandas as pd 
from torch.utils.data import Dataset, DataLoader, random_split
import os
from tqdm import tqdm
import torch

class ImageTextDataset(Dataset):
    def __init__(self, labels_path, img_folder):
        with open(labels_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
        # self.img_names, self.labels = [i.split(',') for i in data] 
        # self.labels = [i.strip() for i in self.labels]
        # self.img_names = [i.strip() for i in self.img_names]
        self.img_names = []
        self.labels = []
        for i in data[1:]:
            img_name , label = i.split(',')
            self.img_names.append(img_name.strip())
            self.labels.append(label.strip())
        self.img_paths = [os.path.join(img_folder, i) for i in self.img_names]

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = int(self.labels[idx])
        return image, label

def collate_func(batch):
    pictures, labels = [],[]
    for items in batch:
        pictures.append(items[0])
        labels.append(items[1])
    return processor(images=pictures, return_tensors="pt"), labels

def train(model, train_loader, optimizer, criterion, device):
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader):
        optimizer.zero_grad()
        image = images.to(device).to(dtype=torch.float32)
        logits = model(**image).logits
        ground_truth = torch.zeros((len(labels),2), dtype=torch.float, device=device)
        i =0
        for label in labels:
          ground_truth[i][label] = 1
          i+=1
        loss = criterion(logits, ground_truth) / 2
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):
    model.to(device)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
          images = images.to(device).to(dtype=torch.float32)
          logits = model(**images).logits
          ground_truth = torch.zeros((len(labels),2), dtype=torch.float, device=device)
          i =0
          for label in labels:
            ground_truth[i][label] = 1
            i+=1
          loss = criterion(logits, ground_truth) / 2
          total_loss += loss.item()

          predicted = logits.max(1).indices
          total += len(labels)
          correct += predicted.eq(torch.tensor(labels).to(device)).sum().item()

    accuracy = 100. * correct / total
    return total_loss / len(test_loader), accuracy
if __name__ == '__main__':
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    model.config.id2label = {0:'0' ,1:'1'}
    model.classifier = torch.nn.Linear(768, 2)
    # model = model.to(dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss()
    device = 'cuda'
    label_path = '/root/css/phase1/trainset_label.txt'
    img_folder = '/root/css/phase1/trainset'
    batch_size = 256
    datasets = ImageTextDataset(label_path, img_folder)
    train_dataset ,test_dataset = random_split(datasets,[0.8,0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_func)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_func)
    
    num_epochs = 5
    optimizer =  torch.optim.AdamW(model.parameters(), lr=1e-4)
    # test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    # print(test_loss)
    for epoch in range(num_epochs):

        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    model_path = '5_epochs_vit.pt'
    torch.save(model, model_path)


