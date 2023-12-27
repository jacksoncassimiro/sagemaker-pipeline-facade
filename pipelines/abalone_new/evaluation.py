import pickle
import tarfile

import numpy as np
import xgboost
from sklearn.metrics import mean_squared_error

from sagemaker_pipeline_facade.processing_step import ProcessingFacadeStep


class EvaluationFacadeStep(ProcessingFacadeStep):

    def execute(self):
        model_path = '/opt/ml/processing/input/model/model.tar.gz'
        with tarfile.open(model_path) as tar:
            tar.extractall(path='.')

        model = pickle.load(open('xgboost-model', 'rb'))

        df = self.read_input_csv('test', 'test.csv', header=None)

        y_test = df.iloc[:, 0].to_numpy()
        df.drop(df.columns[0], axis=1, inplace=True)

        X_test = xgboost.DMatrix(df.values)

        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        std = np.std(y_test - predictions)
        properties = {
            'regression_metrics': {
                'mse': {'value': mse, 'standard_deviation': std},
            },
        }

        self.write_output_json('evaluation', properties)
