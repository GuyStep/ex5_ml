from gcommand_loader import GCommandLoader
import torch
import torch.nn as nn

DROPOUT = 0.5
CLASSES = 30
EPOCHS = 25
BATCH = 200
ETA = 0.001
TRAIN_PATH = "short/train"
VALIDATION_PATH = "short/valid"
TEST_PATH = "short/test"


def data_loader(dic_path, batch_size, train):
    shuffle = train
    dataset = GCommandLoader(dic_path)
    test_set = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=20, pin_memory=True, sampler=None)
    return test_set, dataset


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        first_channels = 32
        second_channels = 62
        third_channels = 40  # was 25, changed to 40
        filter_size = 4
        fc1_size = 1200
        fc2_size = 320
        # 3 conv layer with filters ,  out chancels , Relu activation function and max pooling
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, first_channels, kernel_size=filter_size, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(first_channels, second_channels, kernel_size=filter_size, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(second_channels, 40, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(40, 30, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # apply drop out layer to avoid over fitting
        self.drop_out = nn.Dropout(DROPOUT)
        # fully connected layer
        self.fc1 = nn.Linear(1800, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, CLASSES)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


def train(train_loader, num_epochs, criterion, optimizer, model):
    model.train()
    # train the model
    total_step = len(train_loader)
    losses = []
    accuracy = []
    for epoch in range(num_epochs):
        enumerated_train = enumerate(train_loader)
        for i, (images, labels) in enumerated_train:
            images = images  # .to("cuda")
            labels = labels  # .to("cuda")
            # run the forward pass
            outputs = model(images)
            # pass model outputs and labels to loss function
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # bckprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy.append(correct / total)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))


def test_model(valid_loader, model):
    # set the modal to evaluation mode
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        print("before enum")
        enumerated_valid = enumerate(valid_loader)
        print("after enum")
        for i, (images, labels) in enumerated_valid:
            print("after for")
            images = images  # .to("cuda")
            labels = labels  # .to("cuda")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model: {} %'.format((correct / total) * 100))


def get_print_list(file_names, predictions):
    print_list = []
    #  combine name and prediction to list for the file
    for name, prediction in zip(file_names, predictions):
        print_list.append(name + ", " + str(prediction.item()))
    return print_list


def print_predict(test_loader, model, dataset):
    files = []
    for name in dataset.spects:
        files.append(name[0][name[0].rindex('/')+1:])
    model.eval()
    predictions = []
    with torch.no_grad():
        enumerated_test = enumerate(test_loader)
        for i, (images, labels) in enumerated_test:
            images = images  # .to("cuda")
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.tolist())

        list_to_print = get_print_list(files, predictions)
        with open('test_y', 'w') as f:
            for item in list_to_print:
                f.write("%s\n" % item)
    f.close()


def main():
    train_set, _ = data_loader(TRAIN_PATH, BATCH, train=True)
    validation_set, _ = data_loader(VALIDATION_PATH, BATCH, train=False)
    # test_set, dataset = load_test_date(TEST_PATH,BATCH)
    cnn = Cnn()  # .to("cuda")

    # loss and optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=ETA)
    train(train_set, EPOCHS, nn.CrossEntropyLoss(), optimizer, cnn)
    # test_model(validation_set, cnn)
    # print_predict(test_set,cnn, dataset)
    print_predict(validation_set,cnn, _)


if __name__ == "__main__":
    main()
