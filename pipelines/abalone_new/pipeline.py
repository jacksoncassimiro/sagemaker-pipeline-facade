import os

import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker_pipeline_facade import Param
from sagemaker_pipeline_facade.pipeline import Pipeline

from pipelines.abalone_new.preprocessing import PreprocessingFacadeStep
from sagemaker_pipeline_facade.training_step import (
    TrainingFacadeStep, ModelArgs
)

if __name__ == '__main__':
    region = 'us-east-1'
    role = 'arn:aws:iam::898487103080:role/ml-experiments'
    bucket = 'turin-complete-experiments'
    prefix = 'abalone'
    model_package_group_name = 'Abalone'
    os.environ['AWS_DEFAULT_REGION'] = region

    sagemaker_session = sagemaker.session.Session()
    pipeline_session = PipelineSession()

    pipeline = Pipeline(
        name='AbaloneNew',
        root_dir=(
            '/home/jackson/Dropbox/Workspaces/SageMaker'
            '/sagemaker-pipeline-facade/pipelines'
        ),
        bucket=bucket,
        region=region,
        role=role,
        pipeline_session=pipeline_session,
    )

    create_dataset_step = PreprocessingFacadeStep(
        inputs=[
            Param(name='data', source='../../datasets/abalone/data.csv')
        ],
        outputs=[
            Param(name='train'),
            Param(name='validation'),
            Param(name='test'),
        ]
    )

    pipeline.add_processing_step(create_dataset_step)

    training_step = TrainingFacadeStep(
        inputs=[
            create_dataset_step.get_output_value_as_param(
                'train', 'train', 'text/csv'
            )
        ],
        model_args=ModelArgs(
            name='xgboost',
            hyper_params={
                'objective': 'reg:linear',
                'num_round': 50,
                'max_depth': 5,
                'eta': 0.2,
                'gamma': 4,
                'min_child_weight': 6,
                'subsample': 0.7,
            }
        )
    )

    pipeline.add_training_step(training_step)

    pipeline.execute()
