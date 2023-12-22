import os

import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker_pipeline_facade.pipeline import (
    Pipeline
)
from pipelines.abalone_new.preprocessing import CreateDatasetStep

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
        root_dir='/home/jackson/Dropbox/Workspaces/SageMaker/sagemaker-pipeline-facade/pipelines',
        role=role,
        pipeline_session=pipeline_session,
    )

    pipeline.add_processing_step(CreateDatasetStep())
    pipeline.execute()

