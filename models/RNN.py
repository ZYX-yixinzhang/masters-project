import numpy as np
import pandas as pd
import torch
import gc
from torchmetrics.classification import BinaryAccuracy
from torch.utils.data import Dataset

# read the data
train = pd.read_csv('./train.csv')
train = train.sample(frac=1)
train = train.set_index('PTID')

test = pd.read_csv('./test.csv')
test = test.sample(frac=1)
test = test.set_index('PTID')

dataset = pd.concat([train, test])

# read the images
loaded = {}
path = './procesed/n/'
files = os.listdir(path)
for f in files:
    loaded[f[:-4]] = np.loadtxt(path+f, delimiter = ',').reshape((100, 100, 100))

label_tr = train['label'].replace('EMCI', 0.).replace('LMCI', 0.).replace('AD', 1.).replace('MCI', 0.)#.values
label_ts = test['label'].replace('EMCI', 0.).replace('LMCI', 0.).replace('AD', 1.).replace('MCI', 0.)#.values
train = train.drop('label', axis=1)
test = test.drop('label', axis=1)
print(label_tr.value_counts(), label_ts.value_counts())

# build pytorch dataloader
class MultiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, numeric, pictures, labels, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.numeric = numeric
        self.pictures = pictures
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.numeric)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = torch.tensor(self.pictures[self.numeric.index[idx]].reshape((1, 80, 256, 170))).to(torch.float32)
        numeric = torch.tensor(self.numeric.iloc[idx].values.reshape((1, -1))).to(torch.float32)
        label = torch.tensor(self.labels[self.numeric.index[idx]].reshape((1, -1))).to(torch.float32)
        sample = {'image': image, 'numeric': numeric, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
from torch.utils.data import DataLoader
# train loader in pytorch
train_dta = MultiDataset(train, loaded, label_tr)
train_loader = DataLoader(train_dta, batch_size=4, shuffle=True)
for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['numeric'].size(), sample_batched['label'].size())
    
# test loader in pytorch
test_dt = MultiDataset(test, loaded, label_ts)
test_loader = DataLoader(test_dt, batch_size=4, shuffle=True)
for i_batch, sample_batched in enumerate(test_loader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['numeric'].size(), sample_batched['label'].size())
    
# build the model
# label_ts
import numpy as np
# nn
class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.brnn = torch.nn.GRU(17, hidden_size = 512, num_layers=1, batch_first=True, bidirectional=True)
        self.brnn2 = torch.nn.GRU(512*2, hidden_size = 256, num_layers=1, batch_first=True, bidirectional=True)
        self.relu = torch.nn.ReLU()
        self.linear = torch.nn.Linear(256*2, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, 1)
        self.sigmoids = torch.nn.Sigmoid()
        
    def forward(self, x):
        x, _ = self.brnn(x)
        x, _ = self.brnn2(x)
        x = self.relu(self.linear(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.sigmoids(x)
        return x
    
# define customised loss
def binary_cross_entropy(y, yhat):
    alpha = 0.55
    loss = -(torch.mean((alpha * y * torch.log(yhat + 1e-6)) + ((1.0- alpha) * (1 - y) * torch.log(1 - yhat + 1e-6))))
    return loss*2

gc.collect()
torch.cuda.empty_cache()
model = CNNClassifier().cuda()
criterion = torch.nn.BCELoss().cuda() # change to binary_cross_entropy for customised loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

losses = []
accs = []
num_epochs = 100

# train the model
for epoch in range(num_epochs):
    torch.cuda.empty_cache()
    model.train()
    running_loss, correct = 0.0, 0.0
    acc = BinaryAccuracy().cuda()
    sensitivity = []
    sensitivity_test = []
    for i_batch, sample_batched in enumerate(train_loader):
        num = sample_batched['numeric'].cuda()
        lb = sample_batched['label'].cuda()
        optimizer.zero_grad()
        outputs = model(num)
        loss = criterion(outputs, lb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pred = (outputs >= 0.5).float()
        TP = torch.sum((pred == 1.) & (lb == 1.)).item()
        FN = torch.sum((pred == 0.) & (lb == 1.)).item()
        if TP+FN != 0:
            sensitivity.append(TP / (TP + FN))
    acc = acc.compute()
    losses.append(running_loss / len(train_loader))#
    accs.append(acc.item())
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Acc: {acc:.3f}, Sen: {sum(sensitivity)/len(sensitivity):.3f}')
    model.eval()
    ac_test = BinaryAccuracy().cuda()
    test_loss = 0.
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            num_est = sample_batched['numeric'].cuda()
            lb_test = sample_batched['label'].cuda()
            outputs_test = model(num_est)
            ac_test(outputs_test, lb_test)
            loss_test = criterion(outputs_test, lb_test)
            test_loss += loss_test.item()
            pred_test = (outputs_test >= 0.5).float()
            TP_test = torch.sum((pred_test == 1.) & (lb_test == 1.)).item()
            FN_test = torch.sum((pred_test == 0.) & (lb_test == 1.)).item()
            if TP_test+FN_test != 0:
                sensitivity_test.append(TP_test / (TP_test + FN_test))
    ac_test = ac_test.compute()
    print(f'Accuracy of the model on the test images: {ac_test : .3f}, Loss: {test_loss / len(test_loader):.4f}, Sen: {sum(sensitivity_test)/len(sensitivity_test):.3f}')