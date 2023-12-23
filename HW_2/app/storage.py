import pandas as pd
import pickle
import json
import os

from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier

class ModelStorage(object):
    """
    Class for models manipulation
    models_dir : str, path to save models, required

    """

    def __init__(self, models_dir, minio_client):
        self.minio_client = minio_client
        self.models_dir = models_dir

        # Создание бакетов, если они не существуют
        if not self.minio_client.bucket_exists("models"):
            self.minio_client.make_bucket("models")
        if not self.minio_client.bucket_exists("datasets"):
            self.minio_client.make_bucket("datasets")
    
    def save_model(self, model_name, model):
        model_path = os.path.join(self.models_dir, f'{model_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        self.minio_client.fput_object("models", f'{model_name}.pkl', model_path)
    

    def get_trained_models(self):
        """
        Returns a list of trained models from MinIO storage.
        """
        trained_models = []
        try:
            # Получаем список объектов в бакете 'models'
            objects = self.minio_client.list_objects("models")
            for obj in objects:
                # Добавляем только имя модели без расширения файла
                model_name = os.path.splitext(obj.object_name)[0]
                trained_models.append(model_name)
        except Exception as e:
            print(f"Error accessing MinIO: {e}")
        
        return trained_models
    
    
    def delete(self, model_name):
        """Delete a model and its datasets by name."""
        # Пути к файлам модели и датасетов
        model_path = os.path.join(self.models_dir, f'{model_name}.pkl')
        features_path = os.path.join(self.models_dir, f'{model_name}_features.json')
        target_path = os.path.join(self.models_dir, f'{model_name}_target.json')

        # Удаление файлов модели и датасетов из локальной файловой системы
        for file_path in [model_path, features_path, target_path]:
            if os.path.exists(file_path):
                os.remove(file_path)

        # Удаление файлов модели и датасетов из MinIO
        for object_name in [f'{model_name}.pkl', f'{model_name}_features.json', f'{model_name}_target.json']:
            try:
                self.minio_client.remove_object("models", object_name)
                self.minio_client.remove_object("datasets", object_name)
            except Exception as e:
                print(f"Error removing object {object_name} from MinIO: {e}")

        return {"message": f"Model and datasets for '{model_name}' were successfully deleted"}

    
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

        # Конвертация и сохранение датасетов в формате JSON
        features_json = json.dumps(train_features)
        target_json = json.dumps(train_target)

        features_path = os.path.join(self.models_dir, f'{model_name}_features.json')
        target_path = os.path.join(self.models_dir, f'{model_name}_target.json')

        with open(features_path, 'w') as f:
            f.write(features_json)
        with open(target_path, 'w') as f:
            f.write(target_json)

        # Сохранение датасетов в Minio
        self.minio_client.fput_object("datasets", f'{model_name}_features.json', features_path)
        self.minio_client.fput_object("datasets", f'{model_name}_target.json', target_path)

        # Отслеживание с помощью DVC
        os.system(f"dvc add {features_path}")
        os.system(f"dvc add {target_path}")

        os.system("git add .dvc/config")
        os.system("git commit -m 'Add dataset files'")

        # Сохранение модели в Minio
        self.save_model(model_name, model)

        return f"Model {model_name} was successfully created!"