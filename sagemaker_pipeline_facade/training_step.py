from sagemaker_pipeline_facade import Param, FacadeStep


class TrainingFacadeStep(FacadeStep):

    def __init__(
            self,
            inputs: list[Param],
            hyper_params: dict
    ):
        super().__init__()

        self.parsed_step = None
        self.inputs = inputs
        self.hyper_params = hyper_params
