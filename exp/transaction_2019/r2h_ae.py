##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# December 2019
##########################################################

from   r2h_models    import R2H, QDNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def train_r2h(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_f = nn.MSELoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).view(data.shape[0], -1), data.to(device).view(target.shape[0], -1)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train_qdnn(model_r2h, model_qdnn, device, train_loader, optimizer, epoch):

    model_qdnn.train()
    loss_f = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).view(data.shape[0], -1), target.to(device)
        optimizer.zero_grad()

        # Project to H space
        projected = model_r2h(data, trained=True)

        # Then classify
        output = model_qdnn(projected)

        loss = loss_f(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_qdnn(model_r2h, model_qdnn, device, test_loader):

    model_r2h.eval()
    model_qdnn.eval()
    test_loss = 0
    correct = 0
    loss_f = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).view(data.shape[0], -1), target.to(device)

            # Project to H space
            projected = model_r2h(data, trained=True)

            # Then classify
            output = model_qdnn(projected)

            test_loss += loss_f(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():


    # Training settings
    use_cuda = True
    device   = torch.device("cuda" if use_cuda else "cpu")
    kwargs   = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # DataGenerator
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128, shuffle=True, **kwargs)

    # Model initialization
    # 'tanh' can be changed to 'hardtanh' or 'relu'
    model_r2h = R2H(256,'tanh',True,28*28).to(device)
    model_qdnn = QDNN(256,10).to(device)

    # Training Options and optimizer and loss
    optim_r2h = optim.Adam(model_r2h.parameters(), lr=0.0001)
    optim_qdnn = optim.Adam(model_qdnn.parameters(), lr=0.001)

    print("Training the R2H autoencoder ...")
    # Train R2H
    for epoch in range(1, 10):
        train_r2h(model_r2h, device, train_loader, optim_r2h, epoch)

    print("Training the R2H-QDNN classifier ...")
    # Train R2H-QDNN
    for epoch in range(1, 101):
        train_qdnn(model_r2h, model_qdnn, device, train_loader, optim_qdnn, epoch)
        test_qdnn(model_r2h, model_qdnn, device, test_loader)


if __name__ == '__main__':
    main()
