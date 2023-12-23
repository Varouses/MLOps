import flask
from flask import jsonify
from flask_restx import Api, Resource, fields
from werkzeug.middleware.proxy_fix import ProxyFix
from storage import ModelStorage
from minio import Minio
import creds
import json


app = flask.Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(app, version='2.0', title='MlOps HW_2', doc='/apidocs/')


minio_url = 'minio:9500'
access_key = creds.ACCESS_KEY  
secret_key = creds.SECRET_KEY 
minio_client = Minio(minio_url, access_key=access_key, secret_key=secret_key, secure=False)


all_models = ModelStorage('./models', minio_client)


create_params = api.model('create', {
    'model_type': fields.Integer(required=True, description='Model type (1: Regression, 2: Classification)', example=1),
    'model_class': fields.String(required=True, description='Model class (LinReg/Lasso for Regression, LogReg/RF for Classification)', example='LinReg'),
    'model_name': fields.String(required=True, description='Model name/id', example='Simple lasso regression'),
    'model_params': fields.Raw(required=False, description='Hyperparameters of the model', example={'alpha': 0.01}),
    'train_features': fields.Raw(required=True, description='Training features in JSON format', example={"X1": [1, 2, 3], "X2": [2, 3, 4]}),
    'train_target': fields.Raw(required=True, description='Training target in array format', example=[1, 2, 1])
})

predict_params = api.model('predict', {
    'model_name': fields.String(required=True, description='Model name/id', example='Simple lasso regression'),
    'save_path': fields.String(required=False, description='Path to save predictions.'),
    }
)

delete_params = api.model('delete', {
    'model_name': fields.String(required=True, description='Model name/id', example='Simple lasso regression')
})


@api.route('/create')
class CreateModel(Resource):
    @api.expect(create_params) 
    def post(self):
        """Train a new model or replace if exists"""
        all_models.create(**api.payload)
        return jsonify({'message': f"Model {api.payload['model_name']} was successfully created."})
    

@api.route('/delete')
class DeleteModel(Resource):
    @api.expect(delete_params)
    def delete(self):
        """Delete model by name"""
        model_name = api.payload['model_name']
        all_models.delete(model_name)
        return jsonify({'message': f"Model {model_name} successfully deleted."})


@api.route('/models')
class ModelsList(Resource):
    def get(self):
        """List all trained models"""
        return jsonify(all_models.get_trained_models())


@api.route('/classes')
class ModelClasses(Resource):
    def get(self):
        """List available model classes"""
        model_classes = {"Regression": ["LinReg", "Lasso"], "Classification": ["LogReg", "RF"]}
        return jsonify(model_classes)


@app.errorhandler(Exception)
def handle_error(error):
    status_code = 500

    if status_code in dir(error):
        status_code = error.status_code

    message = [str(x) for x in error.args]
    response = {
        'error': {
            'type': error.__class__.__name__,
            'message': message
        }
    }
    return jsonify(response), status_code


if __name__ == '__main__':
    app.run(debug=True)
