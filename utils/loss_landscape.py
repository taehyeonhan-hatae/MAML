
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhessian import hessian
import matplotlib.pyplot as plt
import numpy as np
import copy

class landscape(nn.Module):
    def __init__(self, model, criterion):
        super(landscape, self).__init__()

        self.model = model
        self.criterion = criterion

    def get_params(self, model_orig, model_perb, direction, alpha):
        for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
            m_perb.data = m_orig.data + alpha * d
        return model_perb

    def save__landscape_image(self, loss_list):

        loss_list = np.array(loss_list)

        fig = plt.figure()
        landscape = fig.gca(projection='3d')
        landscape.plot_trisurf(loss_list[:, 0], loss_list[:, 1], loss_list[:, 2], alpha=0.8, cmap='viridis')
        # cmap=cm.autumn, #cmamp = 'hot')

        landscape.set_title('Loss Landscape')
        landscape.set_xlabel('ε_1')
        landscape.set_ylabel('ε_2')
        landscape.set_zlabel('Loss')

        landscape.view_init(elev=15, azim=75)
        landscape.dist = 6

        # plt.savefig('savefig_default.png')

    def show(self, inputs, targets):

        model = self.model.cuda()
        inputs, targets = inputs.cuda(), targets.cuda()

        hessian_comp = hessian(model, self.criterion, data=(inputs, targets), cuda=True)

        # get the top eigenvector
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)

        # lambda is a small scalar that we use to perturb the model parameters along the eigenvectors
        lams1 = np.linspace(-0.5, 0.5, 21).astype(np.float32)
        lams2 = np.linspace(-0.5, 0.5, 21).astype(np.float32)

        model_perb1 = copy.deepcopy(model)
        model_perb1.eval()
        model_perb1.cuda()

        model_perb2 = copy.deepcopy(model)
        model_perb2.eval()
        model_perb2.cuda()

        loss_list = []

        for lam1 in lams1:
            for lam2 in lams2:
                model_perb1 = self.get_params(model, model_perb1, top_eigenvector[0], lam1)
                model_perb2 = self.get_params(model_perb1, model_perb2, top_eigenvector[1], lam2)
                loss_list.append((lam1, lam2, self.criterion(model_perb2(inputs), targets).item()))

        self.save__landscape_image(loss_list)



