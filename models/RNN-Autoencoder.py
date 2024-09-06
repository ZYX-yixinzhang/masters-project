from torch.utils.data import Dataset
class MultiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, numeric, demographic, labels, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.numeric = numeric
        self.demographic = demographic
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.numeric)

    def __getitem__(self, idx):
        indices = list(self.numeric.keys())
        if torch.is_tensor(idx):
            idx = idx.tolist()
        msk = torch.Tensor([[-1.]*14]).reshape((1, 14))
        mask_d = torch.Tensor([[-1.]*3])
        numeric = torch.tensor(self.numeric[indices[idx]].reshape((2, -1))).to(torch.float32)
        numeric = torch.cat((numeric, msk, msk), 0)
        demographic = torch.tensor(self.demographic.loc[indices[idx]].values).to(torch.float32)
        label = torch.tensor(self.labels.loc[indices[idx]].values).to(torch.float32)[-1].reshape(1,)
        sample = {'numeric': numeric, 'demographic': demographic, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
from torch.utils.data import DataLoader
train_dta = MultiDataset(features, demographic, labels)
train_loader = DataLoader(train_dta, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test, batch_size=4, shuffle=True)
for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched['numeric'].size(), 
          sample_batched['demographic'].size(), sample_batched['label'].size())
    #break
for i_batch, sample_batched in enumerate(test_dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['numeric'].size(), sample_batched['label'].size())
    
import numpy as np
# nn
class PPADClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_value = -1
        self.repeat_vector = 4
        
        self.gru1 = torch.nn.RNN(14, hidden_size = 16, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout1 = torch.nn.Dropout(0.4)
        self.gru2 = torch.nn.GRU(32, hidden_size = 8, batch_first=True, bidirectional=True)
        self.dropout2 = torch.nn.Dropout(0.4)
        self.gru3 = torch.nn.GRU(16, hidden_size = 8, batch_first=True, bidirectional=True)
        self.dropout3 = torch.nn.Dropout(0.4)
        self.gru4 = torch.nn.GRU(16, hidden_size = 16, batch_first=True, bidirectional=True)
        self.dropout4 = torch.nn.Dropout(0.4)
        self.flatten = torch.nn.Flatten()
        self.output_dropout = torch.nn.Dropout(0.4)
        self.linear1 = torch.nn.Linear(32+3, 128)
        self.linear2 = torch.nn.Linear(128, 16)
        self.linear3 = torch.nn.Linear(16, 4)
        self.linear4 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x, d):
        mask = (x == self.mask_value)
        x, _ = self.gru1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout1(x)
        x, _ = self.gru2(x)
        x = x[:, -1]
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = x.unsqueeze(1)
        x = x.repeat(1, self.repeat_vector, 1)
        x, _ = self.gru3(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout3(x)
        x, _ = self.gru4(x)
        x = x[:, -1]
        x = torch.nn.functional.relu(x)
        x = self.dropout4(x)
        x = self.flatten(x)
        x = self.output_dropout(x)
        x = torch.cat([x, d],dim=1)
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.linear3(x)
        x = torch.nn.functional.relu(x)
        x = self.linear4(x)
        x = self.sigmoid(x)
        return x

def binary_cross_entropy(y, yhat):
    alpha = 0.7
    loss = -(torch.mean((alpha * y * torch.log(yhat + 1e-6)) + ((1.0- alpha) * (1 - y) * torch.log(1 - yhat + 1e-6))))
    return loss

a = torch.Tensor([[[0, 0, 0, 0]]*4]).T
b = torch.Tensor([[[0.5, 0.5, 0.5, 0.5]]*4]).T
binary_cross_entropy(a, b)

import gc
from torchmetrics.classification import BinaryAccuracy

gc.collect()
torch.cuda.empty_cache()

model = PPADClassifier().cuda()
criterion = binary_cross_entropy().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

losses = []
accs = []
num_epochs = 800

for epoch in range(num_epochs):
    model.train()
    running_loss, correct = 0.0, 0.0
    acc = BinaryAccuracy().cuda()
    for i_batch, sample_batched in enumerate(train_loader):
        dm = sample_batched['demographic'].cuda()
        num = sample_batched['numeric'].cuda()
        lb = sample_batched['label'].cuda()
        optimizer.zero_grad()
        
        outputs = model(num, dm)
        loss = binary_cross_entropy(outputs, lb)
        loss = loss.requires_grad_()
        loss.backward()
        acc(outputs, lb)
        optimizer.step()
        running_loss += loss.item()
    acc = acc.compute()
    losses.append(running_loss / len(train_loader))
    accs.append(acc.item())
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Acc: {acc:.3f}')