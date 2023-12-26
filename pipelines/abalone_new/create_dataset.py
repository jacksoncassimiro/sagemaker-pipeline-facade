import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sagemaker_pipeline_facade.processing_step import ProcessingFacadeStep


class CreateDatasetFacadeStep(ProcessingFacadeStep):

    def execute(self):
        columns = {
            'sex': str,
            'length': np.float64,
            'diameter': np.float64,
            'height': np.float64,
            'whole_weight': np.float64,
            'shucked_weight': np.float64,
            'viscera_weight': np.float64,
            'shell_weight': np.float64,
            'rings': np.float64,
        }

        df = self.read_input_csv(
            'data',
            'data.csv',
            header=None,
            names=list(columns.keys()),
            dtype=columns
        )

        numeric_features = list(columns.keys())
        numeric_features.remove('sex')
        numeric_features.remove('rings')
        numeric_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]
        )
        categorical_features = ['sex']
        categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]
        )

        preprocess = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
            ]
        )

        y = df.pop('rings')
        X_pre = preprocess.fit_transform(df)
        y_pre = y.to_numpy().reshape(len(y), 1)

        X = np.concatenate((y_pre, X_pre), axis=1)

        np.random.shuffle(X)
        train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

        self.write_output_csv('train', train, header=False, index=False)
        self.write_output_csv('validation', validation, header=False, index=False)
        self.write_output_csv('test', test, header=False, index=False)
