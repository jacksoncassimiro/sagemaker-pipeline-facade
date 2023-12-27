from sagemaker_pipeline_facade import Param, FacadeStep


class BatchTransformFacadeStep(FacadeStep):
    def __init__(self, model_data: Param, batch_data: Param):
        super().__init__()
        self.model_data = model_data
        self.batch_data = batch_data

        self.model = None
        self.parsed_model_step = None
        self.parsed_transform_step = None

    def get_model_as_param(self):
        return Param(
            name='model',
            source=self.model
        )
