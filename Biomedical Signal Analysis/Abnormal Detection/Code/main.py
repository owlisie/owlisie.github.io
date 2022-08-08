from __future__ import print_function, division

import os
import sys
import csv
import cv2
import copy
import time
import operator
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import config as cf
import torchvision
from torchvision import datasets, models, transforms

from model import *
from gradcam import *
from gradcam import BackPropagation, GradCAM, GuidedBackPropagation, save_class_activation_on_image


class Options:
    def __init__(self):
        self.num_epochs = 100
        self.net_type = 'resnet'
        self.lr = 1e-3
        self.depth = 18 # 18, 34, 50, 101, 152
        self.weight_decay = 5e-4
        self.finetune = False
        self.addlayer = False
        self.resetClassifier = True
        self.testOnly = False

# Chap 1 : Train Network
print('\n[Chapter 1] : Train the network before gradCAM')

# Phase 1 : Data Upload
print('\n[Phase 1] : Data Preperation')
opts = Options()

data_transforms = {
    'train': transforms.Compose([
        # transforms.Resize(236),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean, cf.std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean, cf.std)
    ]),
}

data_dir = cf.aug_dir
dataset_dir = cf.data_base.split("/")[-1] + os.sep
print("| Preparing model trained on %s dataset..." % (cf.data_base.split("/")[-1]))
dsets = {
    x: datasets.ImageFolder(data_dir + '/' + x, data_transforms[x])
    for x in ['train', 'val']
}
dset_loaders = {
    x: torch.utils.data.DataLoader(dsets[x], batch_size=cf.batch_size, shuffle=(x == 'train'), num_workers=4)
    for x in ['train', 'val']
}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
dset_classes = dsets['train'].classes

use_gpu = torch.cuda.is_available()

# Phase 2 : Model setup
print('\n[Phase 2] : Model setup')
now_time = datetime.datetime.now().strftime('%m%d_%H%M_')

def getNetwork(opts):
    if (opts.net_type == 'alexnet'):
        net = models.alexnet(pretrained=opts.finetune)
        file_name = 'alexnet'
    elif (opts.net_type == 'vggnet'):
        if (opts.depth == 11):
            net = models.vgg11(pretrained=opts.finetune)
        elif (opts.depth == 13):
            net = models.vgg13(pretrained=opts.finetune)
        elif (opts.depth == 16):
            net = models.vgg16(pretrained=opts.finetune)
        elif (opts.depth == 19):
            net = models.vgg19(pretrained=opts.finetune)
        else:
            print('Error : VGGnet should have depth of either [11, 13, 16, 19]')
            sys.exit(1)
        file_name = 'vgg-%s' % (opts.depth)
    elif (opts.net_type == 'resnet'):
        net = model.resnet(opts.finetune, opts.depth)

        file_name = 'resnet-%s' % (opts.depth)
    else:
        print('Error : Network should be either [alexnet / vggnet / resnet / densenet]')
        sys.exit(1)

    return net, file_name

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Test only option
if (opts.testOnly):
    print("| Loading checkpoint model for test phase...")
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(opts)
    print('| Loading ' + file_name + ".t7...")
    checkpoint = torch.load('./checkpoint/' + dataset_dir + '/' + file_name + '.t7')
    model = checkpoint['model']

    if use_gpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    testsets = datasets.ImageFolder(cf.test_dir, data_transforms['val'])

    testloader = torch.utils.data.DataLoader(
        testsets,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    print("\n[Phase 3 : Inference on %s]" % cf.test_dir)
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)

        print(outputs.data.cpu().numpy()[0])
        file_name = 'densenet-%s' % (opts.depth)
        softmax_res = softmax(outputs.data.cpu().numpy()[0])

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100. * correct / total
    print("| Test Result\tAcc@1 %.2f%%" % (acc))

    sys.exit(0)

# Training model
def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=opts.num_epochs):
    global dataset_dir
    since = time.time()

    best_model, best_acc = model, 0.0

    print('\n[Phase 3] : Training Model')
    print('| Training Epochs = %d' % num_epochs)
    print('| Initial Learning Rate = %f' % opts.lr)
    print('| Optimizer = SGD')
    print('| Resnet depth = %d' % opts.depth)
    output_file = "./logs/" + opts.net_type + str(opts.depth) + '_' + now_time + ".csv"

    with open(output_file, 'w', newline='') as csvfile:
        fields = ['epoch', 'train_acc', 'val_acc']
        writer = csv.writer(csvfile)

        writer.writerow(fields)

        for epoch in range(num_epochs):
            train_acc = 0
            val_acc = 0
            print("Train for epoch : ", epoch + 1)
            for phase in ['train', 'val']:

                if phase == 'train':
                    optimizer, lr = lr_scheduler(optimizer, epoch)
                    print('\n=> Training Epoch #%d, LR=%f' % (epoch + 1, lr))
                    model.train(True)
                else:
                    model.train(False)
                    model.eval()

                running_loss, running_corrects, tot = 0.0, 0, 0

                for batch_idx, (inputs, labels) in enumerate(dset_loaders[phase]):
                    if use_gpu:
                        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    optimizer.zero_grad()

                    # Forward Propagation
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    # Backward Propagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # Statistics
                    # running_loss += loss.item()
                    running_loss += loss.item()
                    running_corrects += preds.eq(labels.data).cpu().sum()
                    tot += labels.size(0)

                    if (phase == 'train'):
                        sys.stdout.write('\r')
                        sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d]\t\tLoss %.4f\tAcc %.2f%%'
                                         % (epoch + 1, num_epochs, batch_idx + 1,
                                            (len(dsets[phase]) // cf.batch_size) + 1, loss.item(),
                                            100. * running_corrects / tot))
                        sys.stdout.flush()
                        sys.stdout.write('\r')

                epoch_loss = running_loss / dset_sizes[phase]
                epoch_acc = running_corrects / dset_sizes[phase]

                if (phase == 'train'):
                    train_acc = epoch_acc

                if (phase == 'val'):
                    print('\n| Validation Epoch #%d\t\t\tLoss %.4f\tAcc %.2f%%'
                          % (epoch + 1, loss.item(), 100. * epoch_acc))

                    if epoch_acc > best_acc:
                        print('| Saving Best model...\t\t\tTop1 %.2f%%' % (100. * epoch_acc))
                        print('Top 1 label : ', preds.data)
                        best_acc = epoch_acc
                        best_model = copy.deepcopy(model)
                        state = {
                            'model': best_model,
                            'acc': epoch_acc,
                            'epoch': epoch,
                        }
                        if not os.path.isdir('checkpoint'):
                            os.mkdir('checkpoint')
                        save_point = './checkpoint/' + dataset_dir
                        if not os.path.isdir(save_point):
                            os.mkdir(save_point)
                        torch.save(state, save_point + file_name + now_time + '.t7')

                    val_acc = epoch_acc

            writer.writerow([epoch + 1, train_acc.item(), val_acc.item()])

    csvfile.close()
    time_elapsed = time.time() - since
    print('\nTraining completed in\t{:.0f} min {:.0f} sec'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc\t{:.2f}%'.format(best_acc * 100))

    return best_model


def exp_lr_scheduler(optimizer, epoch, init_lr=opts.lr, weight_decay=opts.weight_decay,
                     lr_decay_epoch=cf.lr_decay_epoch):
    lr = init_lr * (0.5 ** (epoch // lr_decay_epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = weight_decay

    return optimizer, lr


model_ft, file_name = getNetwork(opts)

if (opts.resetClassifier):
    print('| Reset final classifier...')
    if (opts.addlayer):
        print('| Add features of size %d' % cf.feature_size)
        num_ftrs = model_ft.fc.in_features
        feature_model = list(model_ft.fc.children())
        feature_model.append(nn.Linear(num_ftrs, cf.feature_size))
        feature_model.append(nn.BatchNorm1d(cf.feature_size))
        feature_model.append(nn.ReLU(inplace=True))
        feature_model.append(nn.Linear(cf.feature_size, len(dset_classes)))
        model_ft.fc = nn.Sequential(*feature_model)
    else:
        if (opts.net_type == 'alexnet' or opts.net_type == 'vggnet'):
            num_ftrs = model_ft.classifier[6].in_features
            feature_model = list(model_ft.classifier.children())
            feature_model.pop()
            feature_model.append(nn.Linear(num_ftrs, len(dset_classes)))
            model_ft.classifier = nn.Sequential(*feature_model)
        elif (opts.net_type == 'resnet'):
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, len(dset_classes))

if use_gpu:
    model_ft = model_ft.cuda()
    model_ft = torch.nn.DataParallel(model_ft, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

def gradCAM():
    # Chap 2 : Train Network
    print('\n[Chapter 2] : Operate gradCAM with trained network')

    # Phase 1 : Model Upload
    print('\n[Phase 1] : Model Weight Upload')
    use_gpu = torch.cuda.is_available()

    # upload labels
    data_dir = cf.test_dir
    trainset_dir = cf.data_base.split("/")[-1] + os.sep

    dsets = datasets.ImageFolder(data_dir, None)
    H = datasets.ImageFolder(cf.aug_base + '/train/')
    dset_classes = H.classes

    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
        # return np.exp(x) / np.sum(np.exp(x), axis=1)


    def getNetwork(opts):
        if (opts.net_type == 'alexnet'):
            file_name = 'alexnet'
        elif (opts.net_type == 'vggnet'):
            file_name = 'vgg-%s' % (opts.depth)
        elif (opts.net_type == 'resnet'):
            file_name = 'resnet-%s' % (opts.depth)
        else:
            print('[Error]: Network should be either [alexnet / vgget / resnet]')
            sys.exit(1)

        return file_name

    def random_crop(image, dim):
        if len(image.shape):
            W, H, D = image.shape
            w, h, d = dim
        else:
            W, H = image.shape
            w, h = dim[0], dim[1]

        left, top = np.random.randint(W - w + 1), np.random.randint(H - h + 1)
        return image[left:left + w, top:top + h], left, top

    # uploading the model
    print("| Loading checkpoint model for grad-CAM...")
    assert os.path.isdir('./path'), '[Error]: No checkpoint directory found!'
    assert os.path.isdir('./path/' + trainset_dir), '[Error]: There is no model weight to upload!'
    file_name = getNetwork(opts)
    checkpoint = torch.load('./path/' + trainset_dir + file_name + '.t7')
    model = checkpoint['model']

    if use_gpu:
        model.cuda()
        cudnn.benchmark = True

    model.eval()

    sample_input = Variable(torch.randn(1, 3, 224, 224), volatile=False)
    if use_gpu:
        sampe_input = sample_input.cuda()

    def is_image(f):
        return f.endswith(".png") or f.endswith(".jpg")

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean, cf.std)
    ])

    """
    #@ Code for inference test

    img = Image.open(cf.image_path)
    if test_transform is not None:
        img = test_transform(img)
    inputs = img
    inputs = Variable(inputs, volatile=False, requires_grad=True)

    if use_gpu:
        inputs = inputs.cuda()
    inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

    outputs = model(inputs)
    softmax_res = softmax(outputs.data.cpu().numpy()[0])

    index,score = max(enumerate(softmax_res), key=operator.itemgetter(1))

    print('| Uploading %s' %(cf.image_path.split("/")[-1]))
    print('| prediction = ' + dset_classes[index])
    """

    # @ Code for extracting a grad-CAM region for a given class
    gcam = GradCAM(list(model._modules.items())[0][1],
                   cuda=use_gpu)  # model=model._modules.items()[0][1], cuda=use_gpu)
    gbp = GuidedBackPropagation(model=list(model._modules.items())[0][1], cuda=use_gpu)

    # print(dset_classes)
    WBC_id = 5  # BHX class
    print("Checking Activated Regions for " + dset_classes[WBC_id] + "...")

    fileList = os.listdir('./samples/')
    i = 1
    for f in fileList:
        file_name = './samples/' + f
        print("Opening " + file_name + "...")

        original_image = cv2.imread(file_name)
        resize_ratio = 224. / min(original_image.shape[0:2])
        resized = cv2.resize(original_image, (0, 0), fx=resize_ratio, fy=resize_ratio)
        cropped, left, top = random_crop(resized, (224, 224, 3))
        print(cropped.size)
        if test_transform is not None:
            img = test_transform(Image.fromarray(cropped, mode='RGB'))
        # center_cropped = original_image[16:240, 16:240, :]
        # expand the image based on the short side

        inputs = img
        inputs = Variable(inputs, requires_grad=True)

        if use_gpu:
            inputs = inputs.cuda()
        inputs = inputs.view(1, inputs.size(0), inputs.size(1), inputs.size(2))

        probs, idx = gcam.forward(inputs)
        # probs, idx = gbp.forward(inputs)

        # Grad-CAM
        gcam.backward(idx=WBC_id)
        if opts.depth == 18:
            output = gcam.generate(target_layer='layer4.1')
        else:
            output = gcam.generate(target_layer='layer4.2')  # a module name to be visualized (required)

        # Guided Back Propagation
        # gbp.backward(idx=WBC_id)
        # feature = gbp.generate(target_layer='conv1')

        # Guided Grad-CAM
        # output = np.multiply(feature, region)

        gcam.save('./results/%s.png' % str(i), output, cropped)
        cv2.imwrite('./results/map%s.png' % str(i), cropped)

        for j in range(3):
            print('\t{:5f}\t{}\n'.format(probs[j], dset_classes[idx[j]]))

        i += 1

    """
    @ Code for extracting the Top-3 Results for each image
    topk = 3

    for i in range(0, topk):
        gcam.backward(idx=idx[i])
        output = gcam.generate(target_layer='layer4.2')

        gcam.save('./results/{}_gcam.png'.format(dset_classes[idx[i]]), output, center_cropped)
        print('\t{:.5f}\t{}'.format(probs[i], dset_classes[idx[i]]))
    """

if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=cf.num_epochs)

    gradCAM()

