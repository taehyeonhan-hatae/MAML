import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
import copy
from mypyhessian import my_hessian
import torch.nn.functional as F


class landscape(nn.Module):
    def __init__(self, model, criterion):
        super(landscape, self).__init__()

        self.model = model
        self.criterion = criterion

    def get_params(self, model_orig, model_perb, direction, alpha):
        i=0
        for m_orig, m_perb in zip(model_orig.parameters(), model_perb.parameters()):
            if m_orig.requires_grad:
                m_perb.data = m_orig.data + alpha * direction[i]
                i = i+1

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

        plt.savefig('savefig_default.png')

    def show(self, inputs, targets):

        model = self.model.cuda()
        inputs, targets = inputs.cuda(), targets.cuda()

        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(name)

        hessian_comp = my_hessian.my_hessian(model, self.criterion, data=(inputs, targets), cuda=True)

        # get the top eigenvector
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)

        print("top_eigenvector")
        print(len(top_eigenvector[0]))
        print(len(top_eigenvector[1]))

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
                # loss_list.append((lam1, lam2, self.criterion(model_perb2(inputs), targets).item()))
                preds, out_feature_dict = model_perb2.forward(x=inputs, num_step=5)
                loss = F.cross_entropy(input=preds, target=targets)

                loss_list.append((lam1, lam2, loss.item()))

        self.save__landscape_image(loss_list)



