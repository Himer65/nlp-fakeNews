from utils import content, generator, fakeNews, training, visualiz
from torch import nn
import torch

"""Датасет который я использовал: 
   https://drive.google.com/uc?id=178f_VkNxccNidap-5-uffXUW475pAuPy&confirm=t"""

vocab_len = 5000 #гиперпараметры
hot_len = 500
epoch = 8
lr = 1e-3
l2 = 1e-4

txt = content(file='train.csv',
              vocab_len=vocab_len,
              hot_len=hot_len)

train = generator(data=(txt[0][:-2000].long(), txt[1][:-2000]),
                  batch=50,
                  shuffle=True)
test = generator(data=(txt[0][-2000:].long(), txt[1][-2000:]),
                 batch=1,
                 shuffle=False)

model = fakeNews(vocab_len)
#model.load_state_dict(torch.load('/content/drive/MyDrive/nlp.pt')[0])

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
#optimizer.load_state_dict(torch.load('/content/drive/MyDrive/nlp.pt')[1])

loss = nn.BCEWithLogitsLoss()

hist = training(model=model,
                epoch=epoch,
                train_loader=train, 
                test_loader=test,
                lossF=loss, 
                optimizer=optimizer)

progress = {'acc': [], 'loss': []}
#progress = torch.load('/content/drive/MyDrive/nlp.pt')[2]
progress['acc'] += hist['acc']
progress['loss'] += hist['loss']

torch.save([model.state_dict(), optimizer.state_dict(), progress], '/content/drive/MyDrive/nlp.pt') #сохраним модель, и оптимизатор для дальнейшего обучения

visualiz(progress)
