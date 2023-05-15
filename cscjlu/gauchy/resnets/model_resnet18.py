import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torchsummary import summary
from torchvision.models.resnet import ResNet, BasicBlock


def main():
    # 定义超参数
    batch_size = 200
    epochs = 20
    learning_rate = 0.001

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载MNIST数据集
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=6)

    # 使用ResNet18
    # model = resnet18(num_classes=10)

    class CustomResNet(ResNet):
        def __init__(self, num_classes=10):
            ''' based on resnet-18 but inchannels 3 replace as 1 '''
            super(CustomResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # 使用自定义的ResNet
    model = CustomResNet(num_classes=10)

    # 如果CUDA可用，使用GPU训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, (1, 224, 224))

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # 训练模型
    acc_list = [0.] * epochs
    time_diff_list = [None] * epochs
    for epoch in range(epochs):
        time_start = datetime.datetime.now()
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

        time_end = datetime.datetime.now()
        time_diff = time_end - time_start
        print(f"Train-time on train set: {time_diff} of epoch{epoch + 1}/{epochs}")
        # 测试模型
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100 * correct / total
        acc_list[epoch] = accuracy
        time_diff_list[epoch] = time_diff
        print(f"Accuracy on test set: {accuracy:.2f}% of epoch{epoch + 1}/{epochs}")
    print("Accuracy per epoch is : ")
    print(acc_list)
    print(" and the train-time of per epoch is : ")
    print(time_diff_list)


if __name__ == '__main__':
    time_start = datetime.datetime.now()
    print("start resnet18=======>\n\t"+str(time_start))
    main()
    time_end = datetime.datetime.now()
    print("=======#end\n\t" + str(time_end))
    time_diff = time_end - time_start
