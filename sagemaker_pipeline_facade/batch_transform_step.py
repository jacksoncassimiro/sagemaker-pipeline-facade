from sagemaker_pipeline_facade import Param, FacadeStep


class BatchTransformFacadeStep(FacadeStep):
    def __init__(self, model: Param, data: Param):
        super().__init__()
        self.model = model
        self.data = data

        self.parsed_model_step = None
        self.parsed_transform_step = None
