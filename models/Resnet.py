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
from torch.utils.data import Dataset
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
        image = torch.tensor(self.pictures[self.numeric.index[idx]].reshape((1, 100, 100, 100))).to(torch.float32)
        numeric = torch.tensor(self.numeric.iloc[idx].values.reshape((-1))).to(torch.float32)
        label = torch.tensor(self.labels[self.numeric.index[idx]])
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
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels * BasicBlock3D.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels * BasicBlock3D.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock3D.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels * BasicBlock3D.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * BasicBlock3D.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet3D(nn.Module):
    def __init__(self, block, num_block, num_classes=1000):
        super(ResNet3D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion+17, 128)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128, 64)
        self.linear3 = torch.nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.relu(self.fc(output))
        output = self.relu(self.linear2(output))
        output = self.linear3(output)
        return output

def resnet18_3d(num_classes=1000):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes)

# train the model
model = resnet18_3d(num_classes=2).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

losses = []
accs = []
num_epochs = 80
for epoch in range(num_epochs):
    model.train()
    torch.cuda.empty_cache()
    running_loss, correct = 0.0, 0.0
    acc = BinaryAccuracy().cuda()
    sensitivity = []
    sensitivity_test = []
    for i_batch, sample_batched in enumerate(train_loader):
        img = sample_batched['image'].cuda()
        lb = sample_batched['label'].cuda()
        optimizer.zero_grad()
        outputs = model(img)
        prd = torch.argmax(outputs, dim=1)
        acc(prd, lb)
        loss = criterion(outputs, lb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        TP = torch.sum((prd == 1) & (lb == 1)).item()
        FN = torch.sum((prd == 0) & (lb == 1)).item()
        if TP+FN != 0:
            sensitivity.append(TP / (TP + FN))
    acc = acc.compute()
    losses.append(running_loss / len(train_loader))
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
            prd_tst = torch.argmax(outputs_test, dim=1)
            ac_test(prd_tst, lb_test)
            loss_test = criterion(outputs_test, lb_test)
            test_loss += loss_test.item()
            TP_test = torch.sum((prd_tst == 1) & (lb_test == 1)).item()
            FN_test = torch.sum((prd_tst == 0) & (lb_test == 1)).item()
            if TP_test+FN_test != 0:
                sensitivity_test.append(TP_test / (TP_test + FN_test))
    ac_test = ac_test.compute()
    print(f'Accuracy of the model on the test images: {ac_test : .3f}, Loss: {test_loss / len(test_loader):.4f}, Sen: {sum(sensitivity_test)/len(sensitivity_test):.3f}')