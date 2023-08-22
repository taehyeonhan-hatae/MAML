import torch
import torch.nn as nn
import numpy as np

from meta_neural_network_architectures import extract_top_level_dict

class PadPrompter(nn.Module):
    def __init__(self, args, prompt_size, image_size):
        super(PadPrompter, self).__init__()

        self.pad_size = prompt_size
        b, c, self.h, self.w = image_size

        self.base_size = self.w - self.pad_size * 2
        self.pad_dict = nn.ParameterDict()

        self.build_network()

    def build_network(self):

        self.pad_dict['pad_up'] = nn.Parameter(torch.empty([3, self.pad_size, self.w]))
        self.pad_dict['pad_down'] = nn.Parameter(torch.empty([3, self.pad_size, self.w]))
        self.pad_dict['pad_left'] = nn.Parameter(torch.empty([3, self.w - self.pad_size * 2, self.pad_size]))
        self.pad_dict['pad_right'] = nn.Parameter(torch.empty([3, self.w - self.pad_size * 2, self.pad_size]))

        # for name, param in self.named_parameters():
        #     print("param.shape == ", param.shape)

        # 추가하면 안됨
        # for name, param in self.named_parameters():
        #     print("build_network == ", name)
        #     nn.init.xavier_uniform_(param)

    def forward(self, x, params=None):

        # print("x.shape == ", x.shape)

        if params is not None:
            # param이 지정될 경우 (inner-loop)

            #print("PadPrompter == ", params.keys())

            param_dict = extract_top_level_dict(current_dict=params)
            pad_up = param_dict['pad_up']
            pad_down = param_dict['pad_down']
            pad_left = param_dict['pad_left']
            pad_right = param_dict['pad_right']

        # print("pad_up.shape == ", pad_up.shape)
        # print("pad_down.shape == ", pad_down.shape)
        # print("pad_left.shape == ", pad_left.shape)
        # print("pad_left.shape == ", pad_left.shape)

        base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([pad_left, base, pad_right], dim=3)
        prompt = torch.cat([pad_up, prompt, pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])


        return x + prompt

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, :self.psize, :self.psize] = self.patch

        return x + prompt


class RandomPatchPrompter(nn.Module):
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

        return x + prompt


def padding(args, prompt_size, image_size):
    return PadPrompter(args, prompt_size, image_size)


def fixed_patch(args):
    return FixedPatchPrompter(args)


def random_patch(args):
    return RandomPatchPrompter(args)