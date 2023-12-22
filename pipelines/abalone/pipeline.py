import os

import sagemaker
from sagemaker import TrainingInput, Model, ModelMetrics, MetricsSource
from sagemaker.estimator import Estimator
from sagemaker.inputs import TransformInput
from sagemaker.processing import ProcessingInput, ProcessingOutput, \
    ScriptProcessor
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.transformer import Transformer
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, \
    ParameterFloat
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, \
    TransformStep

if __name__ == '__main__':
    region = 'us-east-1'
    role = 'arn:aws:iam::898487103080:role/ml-experiments'
    bucket = 'turin-complete-experiments'
    prefix = 'abalone'
    model_package_group_name = 'Abalone'
    base_uri = f"s3://{bucket}/{prefix}/data"

    os.environ['AWS_DEFAULT_REGION'] = region

    input_data_uri = sagemaker.s3.S3Uploader.upload(
        local_path='data.csv',
        desired_s3_uri=base_uri,
    )
    print(input_data_uri)

    batch_data_uri = sagemaker.s3.S3Uploader.upload(
        local_path='batch.csv',
        desired_s3_uri=base_uri,
    )
    print(batch_data_uri)

    sagemaker_session = sagemaker.session.Session()
    pipeline_session = PipelineSession()

    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1)
    instance_type = ParameterString(name="TrainingInstanceType",
                                    default_value="ml.m5.xlarge")
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputData",
        default_value=input_data_uri,
    )
    batch_data = ParameterString(
        name="BatchData",
        default_value=batch_data_uri,
    )
    mse_threshold = ParameterFloat(name="MseThreshold", default_value=6.0)

    sklearn_processor = SKLearnProcessor(
        framework_version='1.2-1',
        instance_type="ml.m5.xlarge",
        instance_count=processing_instance_count,
        base_job_name="sklearn-abalone-process",
        role=role,
        sagemaker_session=pipeline_session,
    )

    processor_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(source=input_data,
                            destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(output_name="train",
                             source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation",
                             source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test",
                             source="/opt/ml/processing/test"),
        ],
        code="preprocessing.py",
    )

    step_process = ProcessingStep(
        name="Processing", step_args=processor_args
    )

    model_path = f"s3://{bucket}/{prefix}/AbaloneTrain"

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type="ml.m5.xlarge",
    )

    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=instance_type,
        instance_count=1,
        output_path=model_path,
        role=role,
        sagemaker_session=pipeline_session,

    )

    xgb_train.set_hyperparameters(
        objective="reg:linear",
        num_round=50,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.7,
    )

    train_args = xgb_train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )

    step_train = TrainingStep(
        name="Train",
        step_args=train_args,
    )

    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="script-abalone-eval",
        role=role,
        sagemaker_session=pipeline_session,
    )

    eval_args = script_eval.run(
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs[
                    "test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation",
                             source="/opt/ml/processing/evaluation"),
        ],
        code="evaluation.py",
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation",
        path="evaluation.json"
    )

    step_eval = ProcessingStep(
        name="Eval",
        step_args=eval_args,
        property_files=[evaluation_report],
    )

    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role,
    )

    step_create_model = ModelStep(
        name="CreateModel",
        step_args=model.create(instance_type="ml.m5.large",
                               accelerator_type="ml.eia1.medium"),
    )

    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=f"s3://{bucket}/{prefix}/AbaloneTransform",
    )

    step_transform = TransformStep(
        name="Transform", transformer=transformer,
        inputs=TransformInput(data=batch_data)
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0][
                    "S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    register_args = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    step_register = ModelStep(name="RegisterModel", step_args=register_args)

    step_fail = FailStep(
        name="Fail",
        error_message=Join(on=" ", values=["Execution failed due to MSE >",
                                           mse_threshold]),
    )

    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value",
        ),
        right=mse_threshold,
    )

    step_cond = ConditionStep(
        name="Condition",
        conditions=[cond_lte],
        if_steps=[step_register, step_create_model, step_transform],
        else_steps=[step_fail],
    )

    pipeline_name = "Abalone"
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            instance_type,
            model_approval_status,
            input_data,
            batch_data,
            mse_threshold,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
    )

    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
