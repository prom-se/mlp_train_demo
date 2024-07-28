import torch
import onnx
from torch.utils import data
from torchvision import datasets, transforms


class MlpNet(torch.nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1 * 28 * 28, 256),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 64),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 11),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)


class MlpDataset(data.Dataset):
    def __init__(self, train=True):
        self.cifar_trans = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.RandomAffine(degrees=(-5, 5), translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.RandomErasing(scale=(0.02, 0.02)),
        ])
        self.mnist_trans = transforms.Compose([
            transforms.AutoAugment(),
            transforms.ToTensor(),
        ])
        self.mnist = datasets.MNIST(root='data', train=train, transform=self.mnist_trans, download=True)
        self.cifar = datasets.CIFAR100(root='data', train=train, transform=self.cifar_trans, download=True)

    def __getitem__(self, index):
        if index < len(self.mnist):
            img, target = self.mnist[index]
            return img, target
        elif index >= len(self.mnist):
            img, _ = self.cifar[index - len(self.mnist)]
            target = 10
            return img, target

    def __len__(self):
        return len(self.mnist) + len(self.cifar)


def save_model(model, path):
    # Save as onnx
    dummy_input = torch.randn(1, 28, 28, 1)
    torch.onnx.export(model, dummy_input, path, input_names=['input'], output_names=['output'])

    # Check onnx
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)


def main():
    net = MlpNet()
    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-03)
    train_dataset = MlpDataset(train=True)
    test_dataset = MlpDataset(train=False)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=True)
    best_acc = 0
    for epoch in range(30):
        for batch, (x, y) in enumerate(train_data):
            yhat = net(x)
            loss = loss_fun(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                print(f'Epoch {epoch + 1}, batch {batch}, loss {loss.item()}')
        # Evaluate.
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in test_data:
                yhat = net(x)
                _, predicted = torch.max(yhat.data, 1)
                total += y.size(0)
                correct += torch.Tensor(predicted == y).sum().item()
            acc = correct / total
            if acc > best_acc:
                best_acc = acc
                save_model(net, 'model/best_model.onnx')
            save_model(net, 'model/last_model.onnx')
            print(f'Epoch: {epoch + 1}, Accuracy: {acc * 100}%')
    print('Finished Training')
    print(f'Best Accuracy: {best_acc * 100}%')


if __name__ == '__main__':
    print("This is a mlp trainer")
    main()
