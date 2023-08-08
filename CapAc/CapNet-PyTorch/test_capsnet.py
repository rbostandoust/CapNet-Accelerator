import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from capsnet import CapsNet
from capsnet import DigitCaps
from data_loader import Dataset
from tqdm import tqdm

USE_CUDA = True if torch.cuda.is_available() else False
BATCH_SIZE = 100
N_EPOCHS = 150
LEARNING_RATE = 0.01
MOMENTUM = 0.9

'''
Config class to determine the parameters for capsule net
'''
global flagz
flagz = True

class Config:
    def __init__(self, dataset='mnist'):
        if dataset == 'mnist':
            # CNN (cnn)
            self.cnn_in_channels = 1
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 6 * 6

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 6 * 6
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 28
            self.input_height = 28

        elif dataset == 'cifar10':
            # CNN (cnn)
            self.cnn_in_channels = 3
            self.cnn_out_channels = 256
            self.cnn_kernel_size = 9

            # Primary Capsule (pc)
            self.pc_num_capsules = 8
            self.pc_in_channels = 256
            self.pc_out_channels = 32
            self.pc_kernel_size = 9
            self.pc_num_routes = 32 * 8 * 8

            # Digit Capsule (dc)
            self.dc_num_capsules = 10
            self.dc_num_routes = 32 * 8 * 8
            self.dc_in_channels = 8
            self.dc_out_channels = 16

            # Decoder
            self.input_width = 32
            self.input_height = 32

        elif dataset == 'your own dataset':
            pass


def train(model, optimizer, train_loader, epoch):
    capsule_net = model
    capsule_net.train()
    n_batch = len(list(enumerate(train_loader)))
    total_loss = 0
    step = 0
    for batch_id, (data, target) in enumerate(tqdm(train_loader)):

        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        #TODO: Return from Capsule_Net
        output, reconstructions, masked, c_ij, W, squashed_u, not_squashed_u, mnist_data, s_j = capsule_net(data, 0, epoch)
        # print("MY_S = " , s_j.data.cpu().numpy()[0][0])
        step = step + 1
        # if step >= 600:
        if (epoch == 16) and (step == 500):
            # c_ij = c_ij.data.cpu().numpy()
            # c_ij = np.asarray(c_ij)
            # print("C.SHAPE = " , c_ij.shape)
            # np.save('c.npy', c_ij)
            W = W.data.cpu().numpy()
            W = W[0]
            W = np.asarray(W)

            #______________________________
            # print("W.SHPAPE = " , W.shape)
            np.save('TRAIN_W_STEP='+str(step)+'_EPOCH='+str(epoch)+'.npy', W)
            #______________________________

            # ______________________________
            s_j = s_j.data.cpu().numpy()
            s_j = np.asarray(s_j)
            np.save('TRAIN_Sj_STEP='+str(step)+'_EPOCH='+str(epoch)+'.npy', s_j)
            print("S.SHAPE = ", s_j.shape)
            print("S = ", s_j[0][0])
            # ______________________________
            # squashed_u = squashed_u.data.cpu().numpy()
            # squ = np.asarray(squashed_u)
            # np.save('TRAIN_SQ_STEP='+str(step)+'_EPOCH='+str(epoch)+'.npy', squ)
            # ______________________________

            not_squashed_u = not_squashed_u.data.cpu().numpy()
            nsqu = np.asarray(not_squashed_u)
            np.save('TRAIN_NSQ_STEP='+str(step)+'_EPOCH='+str(epoch)+'.npy', nsqu)

            # ______________________________
            # mnist_data = mnist_data.data.cpu().numpy()
            # mnist_data = np.asarray(mnist_data)
            # np.save('mnist_data.npy', mnist_data)
            # ______________________________
            print("TRAINING FEATURE DONE!")

        loss = capsule_net.loss(data, output, target, reconstructions)
        loss.backward()
        optimizer.step()
        correct = sum(np.argmax(masked.data.cpu().numpy(), 1) == np.argmax(target.data.cpu().numpy(), 1))
        # print("LOSS DATA = " , torch.tensor.item(loss.data))
        train_loss = loss.item()
        total_loss += train_loss
        f = open("train.txt", "a+")
        f.write("Train ACC =  %f\r\n" % (correct / float(BATCH_SIZE)))
        if batch_id % 100 == 0:
            tqdm.write("Epoch: [{}/{}], Batch: [{}/{}], train accuracy: {:.6f}, loss: {:.6f}".format(
                epoch,
                N_EPOCHS,
                batch_id + 1,
                n_batch,
                correct / float(BATCH_SIZE),
                train_loss / float(BATCH_SIZE)
                ))
    tqdm.write('Epoch: [{}/{}], train loss: {:.6f}'.format(epoch,N_EPOCHS,total_loss / len(train_loader.dataset)))



def test(capsule_net, test_loader, epoch):
    capsule_net.eval()
    test_loss = 0
    correct = 0
    step = 0
    for batch_id, (data, target) in enumerate(test_loader):

        target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
        data, target = Variable(data), Variable(target)

        if USE_CUDA:
            data, target = data.cuda(), target.cuda()
        #TODO: change outputs of capsule_net
        # output, reconstructions, masked = capsule_net(data)
        output, reconstructions, masked, c_ij, W, squashed_u, not_squashed_u, mnist_data, s_j = capsule_net(data, 1, epoch)
        step = step + 1
        # print("STEP = ", step)
        if step % 10 == 0 and epoch == 16:
            # c_ij = c_ij.data.cpu().numpy()
            # c_ij = np.asarray(c_ij)
            # np.save('TEST_c'+str(step)+'.npy', c_ij)
            # W = W.data.cpu().numpy()
            # W = W[0]
            # W = np.asarray(W)
            # np.save('TEST_W'+str(step)+'.npy', W)
            # squashed_u = squashed_u.data.cpu().numpy()
            # squ = np.asarray(squashed_u)
            # np.save('TEST_squashed'+str(step)+'.npy', squ)
            not_squashed_u = not_squashed_u.data.cpu().numpy()
            nsqu = np.asarray(not_squashed_u)
            np.save('TEST_NSQ_STEP='+str(step)+'_EPOCH='+str(epoch)+'.npy', nsqu)
            # mnist_data = mnist_data.data.cpu().numpy()
            # mnist_data = np.asarray(mnist_data)
            # np.save('TEST_mnist_data'+str(step)+'.npy', mnist_data)
            np.save('TEST_TARGET_STEP='+str(step)+'_EPOCH='+str(epoch)+'.npy', target.cpu())
            print("###########TEST FEATURES EXTRACTED " + str(step) + "##################")
        loss = capsule_net.loss(data, output, target, reconstructions)

        # test_loss += loss.data[0]
        test_loss += loss.item()
        correct += sum(np.argmax(masked.data.cpu().numpy(), 1) ==
                       np.argmax(target.data.cpu().numpy(), 1))

    tqdm.write(
        "Epoch: [{}/{}], test accuracy: {:.6f}, loss: {:.6f}".format(epoch, N_EPOCHS, correct / len(test_loader.dataset),
                                                                  test_loss / len(test_loader)))


if __name__ == '__main__':
    torch.manual_seed(1)
    dataset = 'cifar10'
    # dataset = 'mnist'
    config = Config(dataset)
    mnist = Dataset(dataset, BATCH_SIZE)

    capsule_net = CapsNet(config)
    capsule_net = torch.nn.DataParallel(capsule_net)
    if USE_CUDA:
        capsule_net = capsule_net.cuda()
    capsule_net = capsule_net.module

    optimizer = torch.optim.Adam(capsule_net.parameters())

    for e in range(1, N_EPOCHS + 1):
        train(capsule_net, optimizer, mnist.train_loader, e)
        print("!!!!!!!!!!!!!!!!!!!!TEST STARTED!!!!!!!!!!!!!!!!!")
        test(capsule_net, mnist.test_loader, e)
