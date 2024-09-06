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
import numpy as np
import torch
# nn
class CNNRNNClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(1, 8, 3, stride=1, padding=1, dilation=1)
        self.conv2 = torch.nn.Conv3d(8, 8, 3, stride=1, padding=1, dilation=1)
        self.relu = torch.nn.ReLU()
        self.pol = torch.nn.MaxPool3d(2, stride=2)
        self.dropout = torch.nn.Dropout(0.3)
        self.conv3 = torch.nn.Conv3d(8, 16, 3, stride=1, padding=1, dilation=1)
        self.conv4 = torch.nn.Conv3d(16, 16, 3, stride=1, padding=1, dilation=1)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.conv5 = torch.nn.Conv3d(16, 32, 3, stride=1, padding=1, dilation=1)
        self.conv6 = torch.nn.Conv3d(32, 32, 3, stride=1, padding=1, dilation=1)
        self.conv7 = torch.nn.Conv3d(32, 32, 3, stride=1, padding=1, dilation=1)
        self.dropout3 = torch.nn.Dropout(0.4)
        self.conv8 = torch.nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv9 = torch.nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv10 = torch.nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.dropout4 = torch.nn.Dropout(0.4)
        self.flatten = torch.nn.Flatten()
        self.brnn = torch.nn.GRU(64*6*6*6+17, hidden_size = 512, num_layers=1, batch_first=True, bidirectional=True)
        self.linear = torch.nn.Linear(512*2, 128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, 1)
        self.sigmoids = torch.nn.Sigmoid()
        
    def forward(self, x, a):
        x = self.conv(x)
        x = self.relu(self.conv2(x))
        x = self.pol(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu(self.conv4(x))
        x = self.pol(x)
        x = self.dropout2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.relu(self.conv7(x))
        x = self.pol(x)
        x = self.dropout3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.relu(x)
        x = self.relu(self.conv10(x))
        x = self.pol(x)
        x = self.dropout4(x)
        x = torch.flatten(x, start_dim=1).reshape((x.shape[0], 1, -1))
        x = torch.cat((x, a), axis=2)
        x, _ = self.brnn(x)
        x = self.relu(self.linear(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        x = self.sigmoids(x)
        return x
    

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
        img = sample_batched['image'].cuda()
        lb = sample_batched['label'].cuda()
        optimizer.zero_grad()
        outputs = model(img)
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
            img_est = sample_batched['image'].cuda()
            lb_test = sample_batched['label'].cuda()
            outputs_test = model(img_est)
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