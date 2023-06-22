# -*- coding: utf-8 -*-

from data import MetaLearningSystemDataLoader
from experiment_builder import ExperimentBuilder
from few_shot_learning_system_curriculum import MAMLFewShotClassifier
from utils.parser_utils import get_args
from utils.dataset_tools import maybe_unzip_dataset

from multiprocessing import freeze_support

## python train_maml_system.py --name_of_args_json_file experiment_config/my_experiment/MAML.json --gpu_to_use $GPU_ID
## python train_maml_system.py --name_of_args_json_file experiment_config/my_experiment/maml+curriculum.json --gpu_to_use 0

if __name__ == '__main__':
    freeze_support()

    # Combines the arguments, model, data and experiment builders to run an experiment
    args, device = get_args()

    # 모델을 구성한다
    model = MAMLFewShotClassifier(args=args, device=device,
                                  im_shape=(2, 3,
                                            args.image_height, args.image_width))
    maybe_unzip_dataset(args=args)

    # 데이터를 불러온다
    data = MetaLearningSystemDataLoader

    # 학습
    maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
    maml_system.run_experiment()

