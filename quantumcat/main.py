# (C) Copyright Artificial Brain 2021.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from quantumcat.circuit import QCircuit
from quantumcat.utils import providers
from quantumcat.algorithms import GroversAlgorithm
import numpy as np
import torchvision
from torchvision import datasets
import torch
from matplotlib import pyplot as plt
from quantumcat.applications.classifier import Net
import torch.optim as optim
import time
import torch.nn.functional as F


def create_circuit_demo():
    circuit = QCircuit(2, 2)
    circuit.x_gate(0)
    circuit.rzz_gate(12, 0, 1)
    circuit.rzx_gate(30, 0, 1)
    circuit.sx_gate(1)
    circuit.sxd_gate(0)
    circuit.td_gate(1)
    circuit.s_gate(0)
    circuit.sdg_gate(0)
    circuit.rxx_gate(60, 0, 1)
    circuit.r_gate(30, 10, 1)
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    circuit.draw_circuit(provider=providers.GOOGLE_PROVIDER)
    print(circuit.execute(provider=providers.GOOGLE_PROVIDER, repetitions=10))


def grovers_demo():
    clause_list_sudoku = [[0, 1], [0, 2], [1, 3], [2, 3]]
    clause_list_light_board = [[0, 1, 3], [1, 0, 2, 4], [2, 1, 5], [3, 0, 4, 6],
                               [4, 1, 3, 5, 7], [5, 2, 4, 8], [6, 3, 7], [7, 4, 6, 8],
                               [8, 5, 7]]

    input_arr = [0, 0, 0, 1, 0, 1, 1, 1, 0]

    grovers_algorithm_unknown_solution = GroversAlgorithm(clause_list=clause_list_light_board, input_arr=input_arr,
                                                          flip_output=True, solution_known='N')

    grovers_algorithm_known_solution = GroversAlgorithm(solution_known='Y', search_keyword=101)

    results = grovers_algorithm_unknown_solution.execute(repetitions=1024, provider=providers.IBM_PROVIDER)

    # grovers_algorithm_unknown_solution.draw_grovers_circuit()

    print(results)


def qml_demo():
    #converts the image into tensors
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    trainset = datasets.MNIST(root='./mnistdata', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(preprocess(trainset), batch_size=1, shuffle=True)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.shape)
    print(labels.shape)
    plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

    network = Net()
    print(network)
    optimizer = optim.SGD(network.parameters(), lr=0.003, momentum=0.9)
    epochs = 30
    time0 = time.time()
    loss_list = []
    for epoch in range(epochs):
        total_loss = []
        target_list = []
        for batch_idx, (data, target) in enumerate(trainloader):
            target_list.append(target.item())
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        loss_list.append(sum(total_loss)/len(total_loss))
        print("Loss = {:.2f} after epoch #{:2d}".format(loss_list[-1],epoch+1))
        #print(loss_list[-1])

    # Normalise the loss between 0 and 1
    print("Training finished, took {:.2f}s  after epoch #{:2d}".format(time() - time0,epochs))
    for i in range(len(loss_list)):
        loss_list[i] += 1


def preprocess(trainset):
    labels = trainset.targets
    labels = labels.numpy()
    index1 = np.where(labels == 0) # filter 0's
    index2 = np.where(labels == 1) # filter on 1's
    n=200 # Number of datapoints per class
    index = np.concatenate((index1[0][0:n],index2[0][0:n]))
    trainset.targets = labels[index]
    trainset.data = trainset.data[index]
    return trainset


if __name__ == '__main__':
    qml_demo()
