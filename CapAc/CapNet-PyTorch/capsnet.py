import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

USE_CUDA = True if torch.cuda.is_available() else False


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=256, kernel_size=9):
        super(ConvLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=1
                              )

    def forward(self, x):
        #TODO: Return in ConvLayer
        return F.relu(self.conv(x)), x


class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, num_routes=32 * 6 * 6):
        super(PrimaryCaps, self).__init__()
        self.num_routes = num_routes
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)
            for _ in range(num_capsules)])

    def forward(self, x, train_or_test, epch):
        # print("TRAIN_OR_TEST = ", train_or_test)
        u = [capsule(x[0]) for capsule in self.capsules]
        u = torch.stack(u, dim=1)
        u = u.view(x[0].size(0), self.num_routes, -1)#Shape of U: (100, 1152, 8)
        mnist_data = x[1]
        # print("U = " ,u[0][1][0])
        # self.my_squash(u)
        # print("U2 = " , first_elem[0])
        # print("U_NEW = " , 2*u[0][0][0]+1)
        #TODO: Return in PrimCaps
        # if train_or_test == 0: #Train
        #     return [self.squash(u) , u, mnist_data]
        # else: #Test
        #     # print("WE ARE IN TEST!")
        #     return [self.my_squash(u), u, mnist_data]
        # return [self.squash(u), u, mnist_data]
        # return [self.my_squash(u), u, mnist_data]

        # return [self.squash(u), u, mnist_data]
        return [self.my_squash2(u), u, mnist_data]

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

    def my_squash(self, input_tensor):
        # output_tensor = input_tensor
        first_elem = input_tensor[:, :, 0] #(100,2048)
        pow2_first_elem = first_elem ** 2
        # predicted_squash_magnitude = -0.0000926908806010465*first_elem + 0.0759881128033938
        #CIFAR10
        predicted_squash_magnitude = -1.51153803894976E-06 * first_elem + 0.030329381680003
        # predicted_squash_magnitude = -0.000214131919292268*pow2_first_elem-0.0000266771680630858*first_elem+0.0848912148365851
        sag = predicted_squash_magnitude.unsqueeze_(-1)
        # sag = sag.expand(100, 1152, 8)#(100, 1152, 8)
        #CIFAR10
        sag = sag.expand(100, 2048, 8)  # (100, 2048, 8)
        return sag * input_tensor
        # print("SAG = " , sag[0][0], predicted_squash_magnitude[0][0] )
        # print("SAG2 = ", sag[1][0], predicted_squash_magnitude[1][0])
        # output_tensor = sag*output_tensor
        # print("START")
        # for i in range(100):
        #     for j in range(1152):
        #         # print(predicted_squash_magnitude[i][j].data.cpu().numpy())
        #         output_tensor[i][j] = float(predicted_squash_magnitude[i][j].data.cpu().numpy())* output_tensor[i][j]
        # print("END")

        # print(input_tensor[0][0][0] , output_tensor[0][0][0])

    def my_squash2(self, input_tensor):

        # [-inf, -20]:
        # a: 0.000173212560997997
        # b: 0.0203090817918987
        #
        # [-20: 0]:
        # a: 0.00220944298121543
        # b: 0.0581210382685285
        #
        # [0: 20]:
        # a: -0.00223103488465464
        # b: 0.0583972258884477
        #
        # [20: inf]:
        # a: -0.000151376
        # b: 0.019439751
        # [-inf, -13.46416092]
        # Intercept
        # 0.024488359
        # X
        # Variable
        # 1
        # 0.000242759
        #
        # [-13.46416092, 0]
        # Intercept
        # 0.06089699
        # X
        # Variable
        # 1
        # 0.002769205
        #
        # [0, 13.23405266]
        # Intercept
        # 0.061313814
        # X
        # Variable
        # 1 - 0.002828244
        #
        # [13.23405266, inf]
        # Intercept
        # 0.023874787
        # X
        # Variable
        # 1 - 0.000219038
        first_elem = input_tensor[:, :, 0] #(100,2048)

        # sorted, indices = torch.sort(first_elem)
        # print(first_elem[0][0], sorted[0][0])
        # kk = torch.scatter(0, indices, sorted)
        # print(kk[0][0])
        # print("----------------------------")

        sorted, indices = first_elem.sort(dim=1)
        new_sorted = sorted

        for i in range(100):
            sagzi = sorted[i][sorted[i] < -13.46416092]
            index1 = len(sagzi)
            if index1 > 0:
                # print("LEN1 = ", index1)
                new_sorted[i][0:index1-1] = 0.000242759*sorted[i][0:index1-1] + 0.024488359
                # new_sorted[i][0:index1 - 1] = sorted[i][0:index1 - 1] + 0.024488359

            sagzi = sorted[i][sorted[i] < 0]
            index2 = len(sagzi)
            if index2 > 0 and index2 > index1:
                # print("LEN2 = ", index2)
                new_sorted[i][index1:index2 - 1] = 0.002769205 * sorted[i][index1:index2 - 1] + 0.06089699
                # new_sorted[i][index1:index2 - 1] =  sorted[i][index1:index2 - 1] + 0.06089699

            sagzi = sorted[i][sorted[i] < 13.23405266]
            index3 = len(sagzi)
            if index3 > 0 and index3 > index2:
                # print("LEN3 = ", index3)
                new_sorted[i][index2:index3 - 1] = -0.002828244 * sorted[i][index2:index3 - 1] + 0.061313814
                # new_sorted[i][index2:index3 - 1] = sorted[i][index2:index3 - 1] + 0.061313814
            if index3 < 2048:
                new_sorted[i][index3:2047] = -0.000219038 * sorted[i][index3:2047] + 0.023874787
                # new_sorted[i][index3:2047] = sorted[i][index3:2047] + 0.023874787


        # print("LEN = ", len(sagzi), "....SHAPE = ", sagzi.shape)

        _, reverse_indices = indices.sort(dim=1)
        inp_reunf = new_sorted.gather(dim=1, index=reverse_indices)


        # for i in range(100):
        #     for j in range(2048):
        #         if first_elem[i][j] <= -20:
        #             first_elem[i][j] = 0.000173212560997997 * first_elem[i][j] + 0.0203090817918987
        #         elif first_elem[i][j] <= 0:
        #             first_elem[i][j] = 0.00220944298121543 * first_elem[i][j] + 0.0581210382685285
        #         elif first_elem[i][j] <= 20:
        #             first_elem[i][j] = -0.00223103488465464 * first_elem[i][j] + 0.0583972258884477
        #         else:
        #             first_elem[i][j] = -0.000151375670608762 * first_elem[i][j] + 0.0194397509688571
        sag = inp_reunf.unsqueeze_(-1)
        sag = sag.expand(100, 2048, 8)  # (100, 2048, 8)
        return sag * input_tensor


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x, train_or_test, epch):
        batch_size = x[0].size(0)
        # print("U1 = ", x[0][0][0][0])
        x[0] = torch.stack([x[0]] * self.num_capsules, dim=2).unsqueeze(4)
        not_squashed_u = x[1]
        squashed_u = x[0]
        # print("U2 = ", x[0][0][0][0])
        mnist_data = x[2]

        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x[0])

        b_ij = Variable(torch.zeros(1, self.num_routes, self.num_capsules, 1))
        if USE_CUDA:
            b_ij = b_ij.cuda()

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # v_j = self.squash(s_j)
            v_j = self.my_squash2(s_j)

            # print("V = " , v_j.shape)#100 1 10 16 1


            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij.squeeze(4).mean(dim=0, keepdim=True)
        #TODO: Return in DigitCaps
        # print("S SHAPE DIGIT= " , s_j.squeeze(1).squeeze(1).shape)
        return v_j.squeeze(1), c_ij, W , squashed_u , not_squashed_u, mnist_data, s_j.squeeze(1)

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    def my_squash(self, input_tensor):

        first_elem = input_tensor[:, :, :, 0] #(100,1152)
        print(first_elem.shape)
        # print(first_elem[0][0][0][0] , input_tensor[0][0][0][0][0])
        # predicted_squash_magnitude = 0.00726040599407078*first_elem + 0.252882557126168
        #CIFAR10
        predicted_squash_magnitude = 0.0322603477722202 * first_elem + 0.339795176
        sag = predicted_squash_magnitude.unsqueeze_(-1)
        sag = sag.expand(100, 1, 10, 16, 1)#(100, 1152, 8)
        # print(sag[0][0][1], sag[0][0][1])

        return sag * input_tensor

    def my_squash2(self, input_tensor):
        first_elem = input_tensor[:, :, :, 0]
        # print("INPUT TENSOR = ", input_tensor.shape) #[100, 1, 10, 16, 1])
        # print("FIRST ELEM = ", first_elem.shape) #[100, 1, 10, 1])
        new_fist_elem = first_elem[:,0,:,0] #(100, 10)
        sorted, indices = new_fist_elem.sort(dim=1)
        new_sorted = sorted

        # [-inf, -0.075410217]
        # Intercept
        # 0.349297946
        # X
        # Variable
        # 1 - 0.074520095
        #
        # [-0.075410217, 0]
        # Intercept
        # 0.27196494
        # X
        # Variable
        # 1 - 0.534473989
        #
        # [0, 0.062207676]
        # Intercept
        # 0.295330779
        # X
        # Variable
        # 1
        # 0.637642944
        #
        # [0.062207676, inf]
        # Intercept
        # 0.353784456
        # X
        # Variable
        # 1
        # 0.169344703

        for i in range(100):
            sagzi = sorted[i][sorted[i] < -0.075410217]
            index1 = len(sagzi)
            if index1 > 0:
                new_sorted[i][0:index1 - 1] = -0.074520095 * sorted[i][0:index1 - 1] + 0.349297946
                # new_sorted[i][0:index1 - 1] = sorted[i][0:index1 - 1] + 0.349297946

            sagzi = sorted[i][sorted[i] < 0]
            index2 = len(sagzi)
            if index2 > 0 and index2 > index1:
                new_sorted[i][index1:index2 - 1] = -0.534473989 * sorted[i][index1:index2 - 1] + 0.27196494
                # new_sorted[i][index1:index2 - 1] =  sorted[i][index1:index2 - 1] + 0.27196494

            sagzi = sorted[i][sorted[i] < 0.062207676]
            index3 = len(sagzi)
            if index3 > 0 and index3 > index2:
                new_sorted[i][index2:index3 - 1] = 0.637642944 * sorted[i][index2:index3 - 1] + 0.295330779
                # new_sorted[i][index2:index3 - 1] = sorted[i][index2:index3 - 1] + 0.295330779
            if index3 < 10:
                new_sorted[i][index3:9] = 0.169344703 * sorted[i][index3:9] + 0.353784456
                # new_sorted[i][index3:9] = sorted[i][index3:9] + 0.353784456

        _, reverse_indices = indices.sort(dim=1)
        inp_reunf = new_sorted.gather(dim=1, index=reverse_indices)
        hat_fist_elem = first_elem
        hat_fist_elem[:,0,:,0] = inp_reunf
        #CIFAR10
        sag = hat_fist_elem.unsqueeze_(-1)
        # print("SAG BEFORE = ", sag.shape) #[100, 1, 10, 1, 1])
        sag = sag.expand(100, 1, 10, 16, 1)
        # print("SAG AFTER = ", sag.shape) #[100, 1, 10, 16, 1])
        # print(sag[0][0][1], sag[0][0][1])

        return sag * input_tensor

class Decoder(nn.Module):
    def __init__(self, input_width=28, input_height=28, input_channel=1):
        super(Decoder, self).__init__()
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.reconstraction_layers = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.input_height * self.input_height * self.input_channel),
            nn.Sigmoid()
        )

    def forward(self, x, data):
        classes = torch.sqrt((x ** 2).sum(2))
        classes = F.softmax(classes, dim=0)

        _, max_length_indices = classes.max(dim=1)
        masked = Variable(torch.sparse.torch.eye(10))
        if USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=Variable(max_length_indices.squeeze(1).data))
        t = (x * masked[:, :, None, None]).view(x.size(0), -1)
        reconstructions = self.reconstraction_layers(t)
        reconstructions = reconstructions.view(-1, self.input_channel, self.input_width, self.input_height)
        return reconstructions, masked


class CapsNet(nn.Module):
    def __init__(self, config=None):
        super(CapsNet, self).__init__()
        if config:
            self.conv_layer = ConvLayer(config.cnn_in_channels, config.cnn_out_channels, config.cnn_kernel_size)
            self.primary_capsules = PrimaryCaps(config.pc_num_capsules, config.pc_in_channels, config.pc_out_channels,
                                                config.pc_kernel_size, config.pc_num_routes)
            self.digit_capsules = DigitCaps(config.dc_num_capsules, config.dc_num_routes, config.dc_in_channels,
                                            config.dc_out_channels)
            self.decoder = Decoder(config.input_width, config.input_height, config.cnn_in_channels)
        else:
            self.conv_layer = ConvLayer()
            self.primary_capsules = PrimaryCaps()
            self.digit_capsules = DigitCaps()
            self.decoder = Decoder()

        self.mse_loss = nn.MSELoss()
    #TODO: FORWARD FUNCTION INPUT
    def forward(self, data, train_or_test, epch):
        #TODO: Return in CapsNet
        output, c_ij, W , squashed_u, not_squashed_u, mnist_data, s_j = self.digit_capsules(self.primary_capsules(self.conv_layer(data), train_or_test, epch), train_or_test, epch)
        reconstructions, masked = self.decoder(output, data)
        # print("S SHAPE RETURN= ", s_j.squeeze(1).squeeze(1).shape)
        return output, reconstructions, masked, c_ij, W, squashed_u, not_squashed_u, mnist_data, s_j

    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstruction_loss(data, reconstructions)

    def margin_loss(self, x, labels, size_average=True):
        batch_size = x.size(0)

        v_c = torch.sqrt((x ** 2).sum(dim=2, keepdim=True))

        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)

        loss = labels * left + 0.5 * (1.0 - labels) * right
        loss = loss.sum(dim=1).mean()

        return loss

    def reconstruction_loss(self, data, reconstructions):
        loss = self.mse_loss(reconstructions.view(reconstructions.size(0), -1), data.view(reconstructions.size(0), -1))
        return loss * 0.0005

