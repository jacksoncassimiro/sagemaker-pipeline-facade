import os
from pathlib import Path

from sagemaker.workflow.pipeline_context import PipelineSession

from pipelines.abalone_new.create_dataset import CreateDatasetFacadeStep
from pipelines.abalone_new.evaluation import EvaluationFacadeStep
from pipelines.abalone_new.register import RegisterModelFacadeStep
from sagemaker_pipeline_facade import Param, PropertyParam
from sagemaker_pipeline_facade.batch_transform_step import (
    BatchTransformFacadeStep
)
from sagemaker_pipeline_facade.images import ImageLoader
from sagemaker_pipeline_facade.pipeline import (
    Pipeline, get_trained_model_as_param, get_output_value_as_param
)
from sagemaker_pipeline_facade.training_step import TrainingFacadeStep

if __name__ == '__main__':
    name = 'AbaloneNew'
    model_group_name = 'Abalone'
    model_approval_status = 'Approved'
    root_dir = Path(os.getcwd()).parent.absolute().as_posix()
    region = 'us-east-1'
    image_uri = ImageLoader.xgboost.load(region)
    role = 'arn:aws:iam::898487103080:role/ml-experiments'
    bucket = 'turin-complete-experiments'
    pipeline_session = PipelineSession()

    pipeline = Pipeline(
        name=name,
        root_dir=root_dir,
        image_uri=image_uri,
        bucket=bucket,
        region=region,
        role=role,
        pipeline_session=pipeline_session,
    )

    create_dataset_step = CreateDatasetFacadeStep(
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
            get_output_value_as_param(
                create_dataset_step, 'train', 'train', 'text/csv'
            )
        ],
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

    pipeline.add_training_step(training_step)

    evaluation_step = EvaluationFacadeStep(
        inputs=[
            get_trained_model_as_param(training_step),
            get_output_value_as_param(
                create_dataset_step, 'test', 'test', 'text/csv'
            ),
        ],
        outputs=[
            PropertyParam(name='evaluation'),
        ]
    )

    pipeline.add_processing_step(evaluation_step)

    batch_transform_step = BatchTransformFacadeStep(
        model_data=get_trained_model_as_param(training_step),
        batch_data=Param(
            name='data',
            source='../../datasets/abalone/batch',
        ),
    )

    pipeline.add_batch_transform_step(batch_transform_step)

    register_step = RegisterModelFacadeStep(
        model=batch_transform_step.get_model_as_param(),
        group_name=model_group_name,
        approval_status=model_approval_status,
        content_type='text/csv',
        response_type='text/csv',
        steps={
            'evaluation_step': evaluation_step
        }
    )

    pipeline.add_register_step(register_step)

    pipeline.execute()
