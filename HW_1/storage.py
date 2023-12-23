import pandas as pd
import pickle
import os

from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier

class ModelStorage(object):
    """
    Class for models manipulation
    models_dir : str, path to save models, required

    """

    def __init__(self, models_dir):
        self.current_models = {}

        if not os.path.exists(models_dir):
            raise ValueError('Invalid directory for models.')

        self.models_dir = models_dir
    

    def get_trained_models(self):
        """
        Returns trained models from the directory

        """
        return list(self.current_models.keys())
    
    def delete(self, model_name):
        """Delete a model by name."""
        model_path = os.path.join(self.models_dir, f'{model_name}.pkl')

        if not os.path.exists(model_path):
            raise KeyError(f"Model '{model_name}' not found.")

        if model_name in self.current_models:
            del self.current_models[model_name]

        os.remove(model_path)

        return {"message": f"Model {model_name} was successfully deleted"}

    
    def create(self, model_type, model_class, model_name, train_features, train_target, model_params=None):
        """
        Generates and fits model object with given parameters.

        """

        if model_type == 1:  
            if model_class == 'LinReg':
                if model_params:
                    model = LinearRegression(**model_params)
                else:
                    model = LinearRegression()
            elif model_class == 'Lasso':
                if model_params:
                    model = Lasso(**model_params)
                else:
                    model = Lasso()
            else:
                raise ValueError("Invalid model class for regression")
        elif model_type == 2:  
            if model_class == 'LogReg':
                if model_params:
                    model = LogisticRegression(**model_params)
                else:
                    model = LogisticRegression()
            elif model_class == 'RF':
                if model_params:
                    model = RandomForestClassifier(**model_params)
                else:
                    model = RandomForestClassifier()
            else:
                raise ValueError("Invalid model class for classification")
        else:
            raise ValueError("Invalid model type")

        # Конвертация JSON данных в pandas DataFrame
        features_df = pd.DataFrame.from_dict(train_features)
        target_df = pd.Series(train_target)

        # Обучение модели
        model.fit(features_df, target_df)

        # Сохранение модели
        model_path = f'{self.models_dir}/{model_name}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Обновление текущих моделей
        self.current_models[model_name] = model

        return f"Model {model_name} was successfully created!"