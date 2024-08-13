import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, precision_score, f1_score
from collections import OrderedDict
from captum.attr import DeepLift
from torch.optim.lr_scheduler import StepLR

# Stratified Shuffle Split 적용 -> train과 test가 동일한 라벨 비율을 유지하도록 함 -> 훈련 데이터: 라벨 1 데이터 311개 중 80%인 249개, 라벨0 데이터 33개 중 80%인 26개
# 테스트 데이터 : 라벨 1 데이터 311개 중 20%인 62개, 라벨 0 데이터 33개 중 20%인 7개   -> 구체적인 샘플 수가 아닌 예시
class GeneExpressionDataset(Dataset):
    def __init__(self, data_folder, target_file, test_split=0.2):
        self.train_datasets = []
        self.meta_test_dataset = None
        
        for csv_file in os.listdir(data_folder):
            if csv_file.endswith('.csv'):
                data = pd.read_csv(os.path.join(data_folder, csv_file))
                features = data.iloc[:, :-1].values
                labels = data.iloc[:, -1].values
                
                # Use StratifiedShuffleSplit to split data into support (80%) and query (20%) sets
                sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=42)
                for support_index, query_index in sss.split(features, labels):
                    support_features, query_features = features[support_index], features[query_index]
                    support_labels, query_labels = labels[support_index], labels[query_index]
                
                dataset = {
                    'support_features': support_features,
                    'support_labels': support_labels,
                    'query_features': query_features,
                    'query_labels': query_labels,
                    'name': csv_file
                }
                
                if csv_file == target_file:
                    self.meta_test_dataset = dataset
                else:
                    self.train_datasets.append(dataset)
# geneexpression_dataset(dataset)클래스
# init함수 : data folder에 있는 csv파일들을 읽어와서 support 8: query 2로 나눔, 이때 라벨 비율을 유지하도록 함
# create_train_tasks함수 : 훈련 데이터셋에 대해 태스크를 생성하는 함수 n_tasks는 어떤 역할?
# create_test_tasks함수 : 테스트 데이터셋에 대해 태스크를 생성하는 함수
    def create_train_tasks(self, n_tasks, K):
        tasks = []
        for dataset in self.train_datasets:
            support_indices = np.random.choice(len(dataset['support_features']), K, replace=False)
            x_support = dataset['support_features'][support_indices]
            y_support = dataset['support_labels'][support_indices]
            x_query = dataset['query_features']
            y_query = dataset['query_labels']
            
            tasks.append(
                ((torch.tensor(x_support).float(), torch.tensor(y_support).float().unsqueeze(1)),
                 (torch.tensor(x_query).float(), torch.tensor(y_query).float().unsqueeze(1)))
            )
        return tasks
# 
    def create_test_task(self):
        x_support = self.meta_test_dataset['support_features']
        y_support = self.meta_test_dataset['support_labels']
        x_query = self.meta_test_dataset['query_features']
        y_query = self.meta_test_dataset['query_labels']
        
        return ((torch.tensor(x_support).float(), torch.tensor(y_support).float().unsqueeze(1)),
                (torch.tensor(x_query).float(), torch.tensor(y_query).float().unsqueeze(1)))



class MAMLModel(nn.Module): #논문의 구조와 동일하게 4계층 mlp로 설정 -> 추후 패스웨이의 수만큼 hidden dim을 설정해도 될듯
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MAMLModel, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(input_dim, hidden_dim)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(hidden_dim, hidden_dim//4)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(hidden_dim//4, hidden_dim//8)),
            ('relu3', nn.ReLU()),
            ('l4', nn.Linear(hidden_dim//8, output_dim)),
        ]))
        self.output_dim = output_dim
# MAMLModel클래스
# init함수: 4계층 mlp 모델 생성
# forward함수: 모델의 순전파 과정을 정의(sigmoid 함수 사용)
# parameterised함수: 모델의 파라미터를 반환(sigmoid 함수 사용)
    def forward(self, x):
        return torch.sigmoid(self.model(x)) #논문과 달리 출력값에 sigmoid를 사용(각 출력의 이진분류에 대한 확률)
    
    def parameterised(self, x, weights):
        x = nn.functional.linear(x, weights[0], weights[1]) # weights[0]은 첫번째 레이어의 가중치, [1]은 첫번째 레이어의 편향
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[2], weights[3])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[4], weights[5])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[6], weights[7])
        return torch.sigmoid(x)

class MAML():
    def __init__(self, model, dataset, inner_lr, meta_lr, K=1, inner_steps=1, tasks_per_meta_batch=3):
        self.dataset = dataset
        self.model = model
        self.weights = list(model.parameters())
        self.criterion = nn.BCELoss()
        self.meta_optimiser = optim.Adam(self.weights, meta_lr)
        self.scheduler = StepLR(self.meta_optimiser, step_size=30, gamma=0.1) #Early Stopping
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps
        self.tasks_per_meta_batch = tasks_per_meta_batch

        #Early Stopping - 3 lines
        self.best_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
# MAML클래스
# init함수: 데이터셋, 모델, 가중치, 손실함수, 메타 옵티마이저, 스케줄러, 내부 학습률, 메타 학습률, K, 내부 학습 스텝, 메타 배치당 태스크 수 등을 초기화
# load_model함수: state_dict에 저장된 모델을 불러옴
# inner_loop함수: 서포트,쿼리 셋을 task로 지정해서 support에 대해 loss를 계산해서 그 loss로 grad를 계산 temp_weights에 저장  
    def load_model(self, file_path):
        state_dict = torch.load(file_path)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = 'model.' + key  # Add 'model.' prefix to match the MAMLModel's state_dict
            new_state_dict[new_key] = value
        self.model.load_state_dict(new_state_dict)
        print(f"Model weights loaded from {file_path}")

        
    def inner_loop(self, task):
        temp_weights = [w.clone() for w in self.weights]
        (x_support, y_support), (x_query, y_query) = task

        for step in range(self.inner_steps): # outer loop iteration과 달리 동일한 task에 대해 학습을 진행하는 것 -> K가 클수록, task가 중요할수록 횟수 증가
            loss = self.criterion(self.model.parameterised(x_support, temp_weights), y_support) #현재 k=1이므로 inner step도 1로 최소화하는 것이 좋음
            grad = torch.autograd.grad(loss, temp_weights, create_graph=True, allow_unused=True)
            temp_weights = [w - self.inner_lr * g if g is not None else w for w, g in zip(temp_weights, grad)]
        
        # Evaluate on the entire query set
        query_loss = self.criterion(self.model.parameterised(x_query, temp_weights), y_query) # use adapted weights to compute query_loss 
        return query_loss, temp_weights #loss와 task에 대해 학습한 가중치도 명시적으로 반환

    def main_loop(self, num_iterations):
        for iteration in range(1, num_iterations + 1): # outer루프이기 때문에 num iteration만큼 반복
            tasks = self.dataset.create_train_tasks(len(self.dataset.train_datasets), self.K)
            meta_loss = torch.tensor(0., device=self.weights[0].device)
            meta_grads = [torch.zeros_like(w) for w in self.weights]

            for task in tasks:
                loss, temp_weights = self.inner_loop(task)
                task_grad = torch.autograd.grad(loss, self.weights, create_graph=True)
                meta_grads = [mg + tg for mg, tg in zip(meta_grads, task_grad)]
                meta_loss += loss # 1차 maml까지 밖에 안돼있음
            #meta_grads = torch.autograd.grad(meta_loss, self.weights, create_graph=True) 코드가 존재해야하지않을까?
            meta_loss /= len(tasks)
            for w, g in zip(self.weights, meta_grads):
                w.grad = g / len(tasks)
            
            self.meta_optimiser.step()
            self.meta_optimiser.zero_grad()

            print(f"Iteration {iteration}/{num_iterations}. Loss: {meta_loss.item()}")
            
            #Early Stopping -before save
            if meta_loss < self.best_loss:
                self.best_loss = meta_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at iteration {iteration}")
                    break

            self.scheduler.step()
# main_loop함수: num iteration만큼 반복하면서 tasks에 대해서 각 task마다 loss계산 -> grad계산 -> meta loss에 loss추가 optimization진행, iteration마다 metaloss 출력
# meta loss가 best loss보다 작으면 patience count를 하나씩 올려서 10번이 채워지면 early stop 이는 inner loop에서 실행됨-> 의도?        
        # Save the meta-trained model
        torch.save(self.model.state_dict(), 'meta_trained_model.pth') #명시적으로 Meta-Train과정의 학습된 가중치를 저장 -> 명시적으로가 무슨 의미?

        total_data_size = len(self.dataset)
        support_set_size = self.total_support_samples
        query_set_size = self.total_query_samples
        #test_size = total_data_size - (support_set_size + query_set_size)
        print(f"Total dataset size: {total_data_size}, Support set size: {support_set_size}, Query set size: {query_set_size}")

    def evaluate(self):
        # 저장된 meta-trained model 가중치 불러오기
        self.model.load_state_dict(torch.load('meta_trained_model.pth'))
        self.model.eval()
        test_task = self.dataset.create_test_task()
        (x_support, y_support), (x_query, y_query) = test_task
        
        # Perform inner loop adaptation
        temp_weights = [w.clone() for w in self.weights]
        for _ in range(self.inner_steps):
            support_loss = self.criterion(self.model.parameterised(x_support, temp_weights), y_support)
            grad = torch.autograd.grad(support_loss, temp_weights, create_graph=True, allow_unused=True)
            temp_weights = [w - self.inner_lr * g if g is not None else w for w, g in zip(temp_weights, grad)]

        # Evaluate on query set using adapted weights
        with torch.no_grad():
            predictions = self.model.parameterised(x_query, temp_weights).squeeze()
            loss = self.criterion(predictions, y_query)
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        num_samples = 0        
        
        # Perform inner loop adaptation
        temp_weights = [w.clone() for w in self.weights]
        for _ in range(self.inner_steps):
            support_loss = self.criterion(self.model.parameterised(x_support, temp_weights), y_support)
            grad = torch.autograd.grad(support_loss, temp_weights, create_graph=True, allow_unused=True)
            temp_weights = [w - self.inner_lr * g if g is not None else w for w, g in zip(temp_weights, grad)]

        # Evaluate on query set
        with torch.no_grad():
            predictions = self.model.parameterised(x_query, temp_weights).squeeze()
            loss = self.criterion(predictions, y_query)
                

            total_loss += loss.item()
            num_samples += x_query.size(0)

            if predictions.dim() == 0:
                predictions = [predictions.item()]
            else:
                predictions = predictions.tolist()

            all_predictions.extend(predictions) # predictions를 리스트로 변환해서 all predictions에 넣어줌 (예측값 저장)
            all_targets.extend(y_query.tolist()) # y_query를 리스트로 변환해서 all targets에 넣어줌 (정답 레이블 저장)

        # 성능 지표 계산
        predicted_labels = [1 if pred > 0.5 else 0 for pred in all_predictions]
        accuracy = sum(1 for true, pred in zip(all_targets, predicted_labels) if true == pred) / num_samples
        auc = roc_auc_score(all_targets, all_predictions)
        precision = precision_score(all_targets, predicted_labels)
        f1 = f1_score(all_targets, predicted_labels)

        print(f"테스트 손실: {total_loss / num_samples}")
        print(f"테스트 정확도: {accuracy * 100:.2f}%")
        print(f"테스트 AUC: {auc:.2f}")
        print(f"테스트 Precision: {precision:.2f}")
        print(f"테스트 F1 Score: {f1:.2f}")

#DeepLIFT -> ref.값 1. 0으로 설정 2. 각 유전자 별로 정상/비정상의 중앙값과 평균값을 설정(총 4개) -> 더 많은 수의 바이오마커를 탐색해주는 ref.값이 더 좋은 값

#def get_attributes_DeepLift(model_, tensor_data, reference, deeplift_target):
#    dl = DeepLift(model_)
#    attribution = dl.attribute(tensor_data,
#                               target=deeplift_target,
#                               baselines=reference)
#    attribution_sum = torch.sum(attribution, 0)
#
#    return attribution_sum

#def run_deeplift(model, tensor_data, reference, deeplift_target):
#    attribution = get_attributes_DeepLift(model, tensor_data, reference, deeplift_target)
#    return attribution
        
# Usage
data_folder = '/mnt/disk1/data_jin/TCGA_preprocessing/data/geo/non_cancer/parkinson/processed'
target_file = 'specific_target_file.csv' # Tissue로 선정
dataset = GeneExpressionDataset(data_folder, target_file, test_split=0.2)

input_dim = dataset.train_datasets[0]['support_features'].shape[1]
hidden_dim = input_dim // 5
output_dim = 1
model = MAMLModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
maml = MAML(model, dataset, inner_lr=0.01, meta_lr=0.001, K=1)

# 가중치 파일 경로
weights_path = '/mnt/disk1/data_jin/TCGA_preprocessing/result/ron_omics/model_weights.pth'
maml.load_model(weights_path)

# 메타 학습 시작
maml.main_loop(num_iterations=5) #outer loop 반복 횟수

# 평가
maml.evaluate()



# 수정 사항

# 이전 : target disease의 1개의 csv파일을 로딩하여 8:2로 서포트셋과 쿼리셋이 분리됨, 이후 inner loop에서 각 task마다 k개의 서포트셋과 k개의 쿼리셋을 사용하여 평가를 진행
# 현재 : target disease에 해당하는 폴더를 탐색하여 폴더에 있는 csv파일(세포타입별 데이터)을 8:2로 서포트셋과 쿼리셋을 분리, 이후 세포별로 별도의 task를 형성하여 k개의 서포트셋과 전체
#        쿼리셋을 사용하여 평가를 진행 + MLP도 4계층으로 변경함
#        추가적으로, 학습할 때, 사전학습과 타겟학습에서 반복을 허용하여 학습을 진행하도록 수정 ,논문은 사전학습은 특정 tissue를 랜덤하게 선택 후 100번 반복학습, 타겟학습은 30번 반복학습했음
#        이때, 논문과는 달리 비암성 데이터는 암성 데이터처럼 유전자 발현량 값의 차이가 뚜렷하지 않을 수 있기에 사전학습 횟수 및 방법 수정이 필요할 수 있음
#        우선, 논문에서 사용한 데이터의 개수보다 약 10배~20배는 데이터의 수가 작으므로, 반복학습되는 iteration의 횟수를 감소시켜야 함

# -> tissue, DR, PD, AD? 여기서 30~40, 15, 15, 15 개씩 데이터가 있고 해당되는 데이터가 조직, 세포이기 때문에 조직세포의 발현량과 그냥 세포단위의 발현량을 구분하여 판단할 수 있을지?
#        


##         사전학습 단계 : 논문-14개의 TCGA, 내 데이터-5개의 TCGA 타겟학습 단계 : 논문-대략적으로 각 330개의 데이터, 내 데이터-task별 15개, 세포 별 총 50~70개의 데이터 
##         Iteration : 사전학습-100 -> 40, 타겟학습-30 -> 6 (대신 task의 수가 1->3으로 증가함) -> 사전학습도 MAML을 사용 시 횟수를 감소

# 고려할 점: hidden layer의 노드 수를 패스웨이로 수정하고, 유전자와 패스웨이의 계층적 구조를 반영하여, DeepLIFT를 통해 바로 패스웨이를 탐색할 수 있도록 수정 + 바이오마커 탐색
#           데이터 샘플의 양이 매우 작기에 20%에 해당하는 쿼리셋을 task 평가에서 사용할지 말지를 정해야함
# 추후 확장 방안 : 발굴한 바이오마커(패스웨이, 유전자)와 연관성이 있는 약물을 GNN or Randomwalk를 통해 확인하여, 비암성 질병이 걸린 사람의 세포와, 정상 세포를 넣으면 효과가 있을 
#                 가능성이 있는 약물을 제안해주는 모델로 확장, 약물 데이터도 concat해서 앙상블 모델로서 활용? 무슨 의미?
# 시사점 : 현 모델을 토대로 비암성 질병에서도 약물까지 찾아주는 모델이 개발되며, 각기 다른 전처리 과정을 거쳐 파편화 되어 있는 데이터를 추후 연구에서 모델 인풋에 맞게 통일된 전처리가
#          진행되도록 할 가능성이 있고, TCGA와 같이 2차 연구의 토대가 될 가능성이 있음


# 논문에서의 특이사항 : K=1이라고 명시했으나, 각 task에서 k=1만큼 데이터가 들어가는 것이 아닌, 전체 데이터셋이 사용됨, K는 명시적으로 사용되고 있지 않음
#                     또한 task=1이므로, MAML보단 MLP에 더 가깝다고 판단됨, 단지 MAML은 outer loop의 iteration마다 이전 파라미터를 초기파라미터로 사용하고, 일반적인 MLP는 독립적일뿐
#                     논문에서는 여러 task에 대한 학습을 통해 일반화와 적응력이라는 MAML의 키워드를 없앴음
#                     또한 main loop를 통해 초기 파라미터 업데이트 이후의 성능평가를 진행하지 않음, target main loop에서 29번까지는 30번째 루프의 meta loss로 학습이 되지만 30이후는 없음

# 적은 데이터에 대해 k개의 서포트셋과 k개의 쿼리셋을 사용하여 각기 다른 조합의 task를 형성하여 범용성을 높이고, iteration 횟수를 늘리는게
# 적은 양의 20%에 해당하는 전체 쿼리셋을 사용하여 평가를 진행하는 것보다 더 좋지 않을까? 

# k개의 서포트셋과 k개의 쿼리셋 vs k개의 서포트셋과 전체 쿼리셋 -> 전자는 동일한 데이터로 학습되더라도 평가가 다를 수 있기에 iteration 횟수를 높이면 범용성을 높이게 됨
#                                                           -> 후자는 동일한 데이터로 학습되면 여러개의 데이터로 평가를 받지만 동일한 loss값을 출력하기에 과적합의 위험성이 있음 
# 이 문제를 자체적으로 비암성 질병(극소수 데이터)의 MAML 최적화 실험으로 사용할 수 있지 않을까? -> 여러 데이터 처리에 따른 MAML 성능비교



# MAML 정리
# outer loop 이후의 모델 평가 -> inner loop에서 사용(meta-training)되지 않았던 새로운 태스크들의 데이터셋으로 수행됨 -> 이를 meta-testing이라 부름, 이 데이터 또한 서포트셋과 쿼리셋으로