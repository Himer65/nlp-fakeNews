import pandas as pd
import numpy as np
from torch import nn
import torch
from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_numericalizer
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


class fakeNews(nn.Module):
    def __init__(self, vocab_len):
        super().__init__()
        self.lstm = nn.Sequential(
             nn.Embedding(vocab_len, 50), 
             nn.LSTM(50, 100, 1, batch_first=True))
        self.dance = nn.Sequential(
             nn.BatchNorm1d(100),
             nn.Linear(100, 50),
             nn.Dropout1d(0.2),
             nn.BatchNorm1d(50),
             nn.Softsign(),
             nn.Linear(50, 15),
             nn.Dropout1d(0.1),
             nn.BatchNorm1d(15),
             nn.Softsign(),
             nn.Linear(15, 1)) 

    def forward(self, x, sig=False):
        out, (h, n) = self.lstm(x)
        out = self.dance(h[0])
        if sig:
            out = torch.special.expit(out)
        return out

def content(file, vocab_len, hot_len): #переводит предложение в токены.
    def length(a, b):
        for i in range(len(a)):
            if len(a[i])>b:
                a[i] = a[i][:b]
            elif len(a[i])<b:
                a[i] = [0 for h in range(b-len(a[i]))]+a[i]

    generate_sp_model(file,
                      vocab_size=vocab_len,
                      model_type='unigram',
                      model_prefix='fake_tokeniz')
    sp_model = load_sp_model('fake_tokeniz.model')
    sp_generator = sentencepiece_numericalizer(sp_model)

    value = pd.read_csv(file)[['text','label']].values

    x = list(map(str, value[:,0])) 
    x = list(sp_generator(x))
    length(x, hot_len)
    
    return (torch.tensor(np.int64(x)), torch.tensor(np.float16(value[:,1]).reshape(-1,1)))

class generator: #маленький класс генератор
    def __init__(self, data, batch, shuffle):
        self.input, self.target = data[0], data[1] 
        self.batch = batch
        self.shuffle = shuffle

    def __len__(self):
        return int(len(self.input)/self.batch)

    def __getitem__(self, idx):
        x = self.input[self.batch*idx:self.batch*(idx+1)]
        y = self.target[self.batch*idx:self.batch*(idx+1)]
        if self.__len__()==idx:
            if self.shuffle: #перемешать масивы
                rn = torch.randperm(self.input.shape[0])
                self.input = self.input[rn]
                self.target = self.target[rn] 
            raise IndexError
        return (x, y)

def testing(model, loader, device): 
    acc_sum = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            pred = model.forward(x.to(device), sig=True)
            acc = accuracy_score(y, torch.round(torch.Tensor.cpu(pred)))
            acc_sum += acc

    return round(acc_sum/len(loader), 3)

def training(model, epoch, train_loader, test_loader, lossF, optimizer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    hist = {'acc': [], 'loss': []}
    model.train()
    for epochs in range(epoch):
        sum_loss = 0
        for batch, (x, y) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            pred = model(x.to(device))
            loss = lossF(pred, y.to(device))
            loss.backward()
            optimizer.step()
            
            sum_loss += loss.item()
            if batch%50==0:
                print(f'Loss: {round(sum_loss/50, 4)} <Epoch: {epochs+1} {int((batch/len(train_loader))*100)}%>')

                acc = testing(model, test_loader, device)
                hist['acc'].append(acc)
                hist['loss'].append(round(sum_loss/50, 3)) 
                print(f'Accuracy: {acc} <Epoch: {epochs+1}>\n')
    
                sum_loss = 0
                model.train()

    print('\n°¬°  ×~×')

    return hist

def visualiz(progress):
    acc = progress['acc']; loss = progress['loss']
    plt.plot(range(len(acc)), acc, label='Точность')
    plt.plot(range(len(loss)), loss, label='Ошибка')
    plt.legend(loc='lower left')
    plt.title('кек))')
    plt.savefig('/content/drive/MyDrive/nlp-history.png')
    plt.show()