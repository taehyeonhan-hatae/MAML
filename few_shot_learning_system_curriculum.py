import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from meta_neural_network_architectures import VGGReLUNormNetwork, ResNet12, MetaCurriculumNetwork
from inner_loop_optimizers import LSLRGradientDescentLearningRule

from utils.storage import save_statistics

from AdMSLoss import AdMSoftmaxLoss


def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


class MAMLFewShotClassifier(nn.Module):
    def __init__(self, im_shape, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__()
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.im_shape = im_shape
        self.current_epoch = 0

        self.rng = set_torch_seed(seed=args.seed)

        self.experiment_name = self.args.experiment_name
        self.comprehensive_loss_excel_create = True

        if self.args.backbone == 'ResNet12':
            self.classifier = ResNet12(im_shape=self.im_shape, num_output_classes=self.args.
                                       num_classes_per_set,
                                       args=args, device=device, meta_classifier=True).to(device=self.device)
        else:  # Conv-4
            self.classifier = VGGReLUNormNetwork(im_shape=self.im_shape, num_output_classes=self.args.
                                                 num_classes_per_set,
                                                 args=args, device=device, meta_classifier=True).to(device=self.device)

        self.task_learning_rate = args.init_inner_loop_learning_rate

        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device,
                                                                    init_learning_rate=self.task_learning_rate,
                                                                    init_weight_decay=args.init_inner_loop_weight_decay,
                                                                    total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                    use_learnable_weight_decay=self.args.alfa,
                                                                    use_learnable_learning_rates=self.args.alfa,
                                                                    alfa=self.args.alfa,
                                                                    random_init=self.args.random_init)

        names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())
        #print("names_weights_copy == ", names_weights_copy)

        self.inner_loop_optimizer.initialise(names_weights_dict=names_weights_copy)

        if self.args.curriculum:

            # adaptive curriculum learning을 구성하기 위해서는 두가지 방법이 있다
            ## 1) meta_nerual_network_architectures를 사용
            #self.meta_curriculum = MetaCurriculumNetwork(input_dim = num_layers, args=args, device=device).to(device=self.device)

            ## 2) few_shot_learning_system_curriculum에서 직접 사용
            self.Inner_loop_Aribiter = nn.Sequential(
                    nn.Conv1d(in_channels=3, out_channels=1, kernel_size=2),
                    nn.Linear(9,4),
                    nn.Sigmoid()).to(device=self.device)


        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)
        print("=====================")


        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)


        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)
        print("=====================")

        # ALFA
        if self.args.alfa:
            ## ALFA에서는 Inner loop interation동안 주어진 task에 적응할 수 있게 하는 Hyper Parmeter(learning rate, weight decay)를 생성한다
            num_layers = len(names_weights_copy)
            input_dim = num_layers * 2
            print("self.update_rule_learner input_dim == ", input_dim)

            self.update_rule_learner = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, input_dim)
            ).to(device=self.device)

        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            print(torch.cuda.device_count())
            if torch.cuda.device_count() > 1:
                self.to(torch.cuda.current_device())
                self.classifier = nn.DataParallel(module=self.classifier)
            else:
                self.to(torch.cuda.current_device())

            self.device = torch.cuda.current_device()  ##

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (
                1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.to(device=self.device)
                else:
                    if "norm_layer" not in name:
                        param_dict[name] = param.to(device=self.device)

        return param_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, generated_alpha_params, generated_beta_params,
                                use_second_order, current_step_idx):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.classifier.module.zero_grad(params=names_weights_copy)
        else:
            self.classifier.zero_grad(params=names_weights_copy)

        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order, allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

        names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

        names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                     names_grads_wrt_params_dict=names_grads_copy,
                                                                     generated_alpha_params=generated_alpha_params,
                                                                     generated_beta_params=generated_beta_params,
                                                                     num_step=current_step_idx)

        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}

        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = dict()

        losses['loss'] = torch.mean(torch.stack(total_losses))
        losses['accuracy'] = np.mean(total_accuracies)

        return losses

    def layer_wise_mean_grad(self, grads):

        layerwise_mean_grads = []

        for i in range(len(grads)):
            layerwise_mean_grads.append(grads[i].mean())

        layerwise_mean_grads = torch.stack(layerwise_mean_grads)

        # gradient 값을 0~1로 Normalization
        layerwise_mean_grads = F.normalize(layerwise_mean_grads, dim=0)

        return layerwise_mean_grads

    def layer_wise_similarity(self, grad1, grad2):

        # grad1과 grad2의 차원이 같을 때만 사용 가능하다

        layerwise_sim_grads = []

        for i in range(len(grad1)):
            cos_sim = F.cosine_similarity(grad1[i], grad2[i])
            layerwise_sim_grads.append(cos_sim.mean())

        layerwise_sim_grads = torch.stack(layerwise_sim_grads)

        return layerwise_sim_grads

    def get_task_embeddings(self, x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task, names_weights_copy):
        # Use gradients as task embeddings
        support_loss, support_preds = self.net_forward(x=x_support_set_task,
                                                       y=y_support_set_task,
                                                       weights=names_weights_copy,
                                                       backup_running_statistics=True,
                                                       training=True, num_step=0)

        target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                     y=y_target_set_task,
                                                     weights=names_weights_copy,
                                                     backup_running_statistics=True, training=True,
                                                     num_step=0)

        self.classifier.zero_grad(names_weights_copy)

        support_grads = torch.autograd.grad(support_loss, names_weights_copy.values(), create_graph=True)
        target_grads = torch.autograd.grad(target_loss, names_weights_copy.values(), create_graph=True)

        support_grads_mean = self.layer_wise_mean_grad(support_grads)
        target_grads_mean = self.layer_wise_mean_grad(target_grads)
        grad_similarity_mean = self.layer_wise_similarity(support_grads, target_grads)

        # print("support_grads_mean len == ", len(support_grads_mean))
        # print("target_grads_mean len == ", len(target_grads_mean))
        # support_grads_mean len ==  10, target_grads_mean len ==  10
        ## [Conv4 x 2(weigth와 bias)] + 2(linear layer의 weight와 bias)

        # print("grad_similarity_mean len == ", len(grad_similarity_mean))


        # for i in range(len(grad_similarity_mean)):
        #     print("grad_similarity_mean" + "[" + str(i) + "] .shape == ", grad_similarity_mean[i].item())

        return support_grads_mean, target_grads_mean, grad_similarity_mean

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        [b, ncs, spc] = y_support_set.shape

        self.num_classes_per_set = ncs

        total_losses = []
        total_accuracies = []
        per_task_target_preds = [[] for i in range(len(x_target_set))]
        self.classifier.zero_grad()
        task_accuracies = []

        # print(" x_support_set === ", len(x_support_set))
        # print(" x_target_set === ", len(x_target_set))
        # print(" y_support_set === ", len(y_support_set))
        # print(" y_target_set === ", len(y_target_set))

        # Outer-loop Start
        ## batch size만큼, 1 iteration을 수행한다.
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in enumerate(
                zip(x_support_set,
                    y_support_set,
                    x_target_set,
                    y_target_set)):

            # print("task_id == ", task_id)
            ## task_id ==  0
            ## task_id ==  1 이 반복된다
            ## batch_size가 2이기 때문이다. 즉 batch_size는 한번에 학습할 task의 수를 뜻한다

            task_losses = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

            names_weights_copy = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_weights_copy.items()}

            n, s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            comprehensive_losses = {}
            comprehensive_losses["epoch"] = epoch
            comprehensive_losses["task_id"] = task_id

            if training_phase:
                comprehensive_losses["phase"] = "train"
            else:
                comprehensive_losses["phase"] = "val"

            comprehensive_losses["num_steps"] = self.args.number_of_training_steps_per_iter

            for num_step in range(self.args.number_of_training_steps_per_iter):
                comprehensive_losses["support_loss_" + str(num_step)] = "null"
                comprehensive_losses["support_accuracy_" + str(num_step)] = "null"

            if self.args.curriculum:
                support_grads_mean, target_grads_mean, grad_similarity_mean = self.get_task_embeddings(
                    x_support_set_task=x_support_set_task,
                    y_support_set_task=y_support_set_task,
                    x_target_set_task=x_target_set_task,
                    y_target_set_task=y_target_set_task,
                    names_weights_copy=names_weights_copy)

                per_step_task = []
                per_step_task.append(support_grads_mean)
                per_step_task.append(target_grads_mean)
                per_step_task.append(grad_similarity_mean)

                # print("per_step_task == ", len(per_step_task))
                ## "per_step_task ==  3
                per_step_task = torch.stack(per_step_task)

                # print("per_step_task == ",  per_step_task.shape)
                ## per_step_task ==  torch.Size([3, 10])

                step = self.Inner_loop_Aribiter(per_step_task)
                #print("step == ", step)
                num_steps = int(torch.argmax(step)) + 1
                # print("num_steps === ", num_steps)
                comprehensive_losses["num_steps"] = num_steps

            ## Inner-loop Start
            for num_step in range(num_steps):
                support_loss, support_preds = self.net_forward(
                    x=x_support_set_task,
                    y=y_support_set_task,
                    weights=names_weights_copy,
                    backup_running_statistics=
                    True if (num_step == 0) else False,
                    training=True, num_step=num_step)

                generated_alpha_params = {}
                generated_beta_params = {}

                if self.args.alfa:

                    support_loss_grad = torch.autograd.grad(support_loss, names_weights_copy.values(),
                                                            retain_graph=True)
                    per_step_task_embedding = []
                    for k, v in names_weights_copy.items():
                        per_step_task_embedding.append(v.mean())

                    for i in range(len(support_loss_grad)):
                        # For the computational efficiency, layer-wise means of gradients and weights
                        per_step_task_embedding.append(support_loss_grad[i].mean())

                    per_step_task_embedding = torch.stack(per_step_task_embedding)

                    generated_params = self.update_rule_learner(per_step_task_embedding)
                    num_layers = len(names_weights_copy)

                    generated_alpha, generated_beta = torch.split(generated_params, split_size_or_sections=num_layers)
                    g = 0
                    for key in names_weights_copy.keys():
                        generated_alpha_params[key] = generated_alpha[g]
                        generated_beta_params[key] = generated_beta[g]
                        g += 1

                # print("support_loss == " , support_loss)
                comprehensive_losses["support_loss_" + str(num_step)] = support_loss.item()

                _, support_predicted = torch.max(support_preds.data, 1)

                support_accuracy = support_predicted.float().eq(y_support_set_task.data.float()).cpu().float()
                comprehensive_losses["support_accuracy_" + str(num_step)] = np.mean(list(support_accuracy))

                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  generated_beta_params=generated_beta_params,
                                                                  generated_alpha_params=generated_alpha_params,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step)

                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task, weights=names_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 num_step=num_step)

                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)

                #elif num_step == (self.args.number_of_training_steps_per_iter - 1):
                elif num_step == num_steps-1:
                    target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task, weights=names_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 num_step=num_step)
                    task_losses.append(target_loss)
                    # print("target_loss == ", target_loss)
                    comprehensive_losses["target_loss_" + str(num_step)] = target_loss.item()

                    _, target_predicted = torch.max(target_preds.data, 1)
                    target_accuracy = target_predicted.float().eq(y_target_set_task.data.float()).cpu().float()
                    comprehensive_losses["target_accuracy_" + str(num_step)] = np.mean(list(target_accuracy))

                ## Inner-loop END

            # Inner-loop 결과를 csv로 생성한다.
            if self.comprehensive_loss_excel_create:
                save_statistics(experiment_name=self.experiment_name,
                                line_to_add=list(comprehensive_losses.keys()),
                                filename=self.experiment_name+".csv", create=True)
                self.comprehensive_loss_excel_create = False
                save_statistics(experiment_name=self.experiment_name,
                                line_to_add=list(comprehensive_losses.values()),
                                filename=self.experiment_name+".csv", create=False)
            else:
                save_statistics(experiment_name=self.experiment_name,
                                line_to_add=list(comprehensive_losses.values()),
                                filename=self.experiment_name+".csv", create=False)

            # for key, val in comprehensive_losses.items():
            #     print("key = {key}, value={value}".format(key=key, value=val))
            # print("==========================================================")

            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)

            accuracy = predicted.float().eq(y_target_set_task.data.float()).cpu().float()

            # batch에 대한 학습이 끝나고, acc와 loss를 기록
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.extend(accuracy)

            if not training_phase:
                self.classifier.restore_backup_stats()

            # Outer-loop End

        # 왜 평균을 내고 있을까?
        ## iteration (task 1, 2)에 대한 평균
        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   total_accuracies=total_accuracies)

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        return losses, per_task_target_preds

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """
        preds = self.classifier.forward(x=x, params=weights,
                                        training=training,
                                        backup_running_statistics=backup_running_statistics, num_step=num_step)

        loss = F.cross_entropy(input=preds, target=y)

        # num_classes = 5
        # adms_loss = AdMSoftmaxLoss(3, num_classes, s=10.0, m=0.5)
        # loss = adms_loss(preds, y)

        return loss, preds

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                                     use_second_order=self.args.second_order and
                                                                      epoch > self.args.first_order_to_second_order_epoch,
                                                     use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                     num_steps=self.args.number_of_training_steps_per_iter,
                                                     training_phase=True)
        return losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                                     use_multi_step_loss_optimization=True,
                                                     num_steps=self.args.number_of_evaluation_steps_per_iter,
                                                     training_phase=False)

        return losses, per_task_target_preds

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        # if 'imagenet' in self.args.dataset_name:
        #    for name, param in self.classifier.named_parameters():
        #        if param.requires_grad:
        #            param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed
        # for name, param in self.classifier.named_parameters():
        #    print(param.mean())

        self.optimizer.step()

    def run_train_iter(self, data_batch, epoch):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        # print("run_train_iter x_support_set shape == ", x_support_set.shape)
        ## run_train_iter x_support_set shape ==  torch.Size([2, 5, 5, 3, 84, 84])
        ## 2가 batch_size

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        # print("run_train_iter epoch == " , epoch) 정확하다

        losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch)

        self.meta_update(loss=losses['loss'])
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds

    def run_validation_iter(self, data_batch):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.training:
            self.eval()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch)

        losses['loss'].backward()  # uncomment if you get the weird memory error
        self.zero_grad()
        self.optimizer.zero_grad()

        return losses, per_task_target_preds

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state['network'] = self.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.load_state_dict(state_dict=state_dict_loaded)
        return state