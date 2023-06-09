# 🍱 Monitor ML Models Managed and Served with BentoML

[BentoML](https://github.com/bentoml/BentoML) is an open-source framework that provides a model registry for versioning ML models and a model serving platform for deploying models to production with support for most major machine learning frameworks, such as scikit-learn, PyTorch, Tensorflow, Keras, XGBoost, and more.

[whylogs](https://github.com/whylabs/whylogs) is an open-source Python library that provides a data profiling tool for monitoring data and ML models in production. 

Both tools are flexible and can be used together to build and monitor AI applications!

#### 🛠️ Workshop Example Overview:

This example shows how to write whylogs data profiles to the [WhyLabs](https://whylabs.ai/) platform for monitoring a machine learning model with BentoML and scikit-learn.

There are two notable files in this example:

- `train.py` - Trains and saves a versioned kNN model with scikit-learn, BentoML, and the iris dataset.
- `service.py` - Creates a model prediction API endpoint with BentoML and writes whylogs data profiles to WhyLabs for ML monitoring.

## Project Setup to Follow Along:

- `python -m venv venv` or `python3 -m venv venv`
- `source venv/bin/activate` 
- `pip install --upgrade pip`
- `pip install -r requirements.txt`

## 1. Run training script:
Run the training script to train and save a versioned kNN model with scikit-learn, BentoML, and the iris dataset.

`python train.py`

you'll see a message similar to "`Model saved: Model(tag="iris_knn:fhfsvifgrsrtjlg6")`". This is the versioned model saved to the BentoML local model store.


## 2. Run BentoML API server:

First update the `WHYLABS_API_KEY`, `WHYLABS_DEFAULT_ORG_ID`, and `WHYLABS_DEFAULT_DATASET_ID` environment variables in `service.py` with your WhyLabs API key, org-id, and project-id.

> Note: A best practice for software deployment is to have these environment variables always set on the machine/environment level (such as per the CI/QA machine, a Kubernetes Pod, etc.) so it is possible to leave code untouched and persist it correctly according to the environment you execute it.

Run the BentoML API server to serve the latest saved model and log the model predictions and data profiles to WhyLabs.
`bentoml serve service:svc --reload`

Use the swagger UI to test the API endpoint at `http://localhost:3000/`

Or use curl to test the API endpoint:

```bash
curl -i \
    --header "Content-Type: application/json" \
    --request POST \
    --data '[[4.9, 3.0, 1.4, 0.2]]' \
    http://localhost:3000/classify
```


## 3. Log training data to WhyLabs as reference profile:

To log the training data as a reference profile, run the following command:

`python service.py`

For more information about refernce profiles, check out the [whylogs documentation](https://whylogs.readthedocs.io/en/latest/examples/integrations/writers/Writing_Reference_Profiles_to_WhyLabs.html).


## Deployment with BentoML

To get your BentoML project ready for deployment, check out the [BentoML documentation](https://docs.bentoml.org/en/latest/).

Add `whylogs[whylabs]` under python packages in the `bentofile.yaml` file to use whylogs and WhyLabs in a production BentoML project.

```yaml
python:
  packages: # Additional pip packages required by the service
    - whylogs[whylabs]
```

## 📚 More Resources

- [Sign up for WhyLabs](https://whylabs.ai/)
- [whylogs Documentation](https://whylogs.readthedocs.io/en/latest/)
- [WhyLabs Documentation](https://docs.whylabs.ai/docs/)
- [BentoML Documentation](https://docs.bentoml.org/en/latest/)
- [WhyLabs Blog](https://whylabs.ai/blog)


- To learn how to deploy your BentoML model to production, check out the [BentoML documentation](https://docs.bentoml.org/en/latest/).
- To learn more about monitoring models in production, check out the [WhyLabs documentation](https://docs.whylabs.ai/docs/).

---------

## 🛠️ How to implement ML monitoring for BentoML with whylogs and WhyLabs:

Next, In `service.py`, we'll get ready to serve and monitor our latest saved model.

To get the WhyLabs API key and org-id, go to the [WhyLabs dashboard](https://hub.whylabsapp.com/) and click on the "Access Tokens" section in the settings menu. Model-ids are found under the "Project Management" tab in the same section.

```python
import atexit
import os

import bentoml
import numpy as np
from bentoml.io import NumpyNdarray, Text

import whylogs as why

# Set WhyLabs environment variables
os.environ["WHYLABS_API_KEY"] = "APIKEY"
os.environ["WHYLABS_DEFAULT_ORG_ID"] = "ORGID"
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "PROJECTID"

# Model Prediction Class Names
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# Load model from BentoML local model store & create a BentoML runner
iris_clf_runner = bentoml.sklearn.get("iris_knn:latest").to_runner()
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

```

> Note: A best practice for software deployment is to have these environment variables always set on the machine/environment level (such as per the CI/QA machine, a Kubernetes Pod, etc.) so it is possible to leave code untouched and persist it correctly according to the environment you execute it.

Now that our bentoML runner and environment variables are set, we can create an API endpoint with BentoML and set up model and data monitoring with whylogs and WhyLabs. We'll Log the data, predicted class, and model probability score.

Create a class in `service.py` to handle startup, shutdown, logging, and prediction logic with Bentoml. This example uses the [rolling log](https://whylogs.readthedocs.io/en/latest/examples/advanced/Log_Rotation_for_Streaming_Data/Streaming_Data_with_Log_Rotation.html) functionality of whylogs to write profiles to WhyLabs at set intervals. The `atexit` module registers a callback function to close the whylogs rolling logger when the service is shut down.

```python
# create a class to handle startup, shutdown, and prediction logic with Bentoml
class MyService:
    def __init__(self):
        # Initialize the whylogs rolling logger with a 5 minute interval
        global logger
        logger = why.logger(
            mode="rolling", interval=5, when="M", base_name="bentoml_predictions"
        )
        logger.append_writer("whylabs")

    def on_exit_callback(self):
        # Close the whylogs rolling logger when the service is shut down
        logger.close()

    # Create a BentoML API endpoint for model predictions & logging
    @svc.api(
        input=NumpyNdarray.from_sample(np.array([4.9, 3.0, 1.4, 0.2], dtype=np.double)),
        output=Text(),
    )
    async def classify(features: np.ndarray):
        results = await iris_clf_runner.predict.async_run([features])
        probs = iris_clf_runner.predict_proba.run([features])
        result = results[0]

        category = CLASS_NAMES[result]
        prob = max(probs[0])

        # create a dict of data & prediction results w/ feature names
        data = {
        "sepal length (cm)": features[0],
        "sepal width (cm)": features[1],
        "petal length (cm)": features[2],
        "petal width (cm)": features[3],
        "class_output": category,
        "proba_output": prob,
        }

        # Log data + model outputs to WhyLabs.ai
        logger.log(data)

        return category


my_service = MyService()
atexit.register(my_service.on_exit_callback)

```

Running `bentoml serve service:svc --reload` will start the BentoML API server locally and log the model predictions and data profiles to WhyLabs.

- To learn how to deploy your BentoML model to production, check out the [BentoML documentation](https://docs.bentoml.org/en/latest/).
- To learn more about monitoring models in production, check out the [WhyLabs documentation](https://docs.whylabs.ai/docs/).

## 🚀 How to run the example:

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train and save model

Train and save a version of the model with BentoML and scikit-learn.

```bash
python train.py
```

### 3. Update WhyLabs environment variables

In `service.py` update the [WhyLabs](https://whylabs.ai/) environment variables with your API key, org-id and project-id.

```python
os.environ['WHYLABS_API_KEY'] = 'APIKEY'
os.environ["WHYLABS_DEFAULT_ORG_ID"] = 'ORGID'
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = 'PROJECTID'
```

### 4. Serve the model with BentoML locally

Run the API server with BentoML to test the API locally.

```bash
bentoml serve service:svc --reload
```

### 5. Test the API locally

With the server running, you can test the API with the Swagger UI, curl, or any HTTP client:

To test with Swagger UI, go to http://localhost:3000 and click on the `classify` endpoint.

To test the API with curl, run the following command:

```bash
curl -i \
    --header "Content-Type: application/json" \
    --request POST \
    --data '[[4.9, 3.0, 1.4, 0.2]]' \
    http://localhost:3000/classify
```

### 6. View the logged data in WhyLabs & Setup Monitoring

In the WhyLabs project, you will see the data profile for the iris dataset and the model predictions being logged. You will need to wait the 5 minutes set in the rolling log before the data is written to WhyLabs.

You can enable ML monitoring for your model and data in the `monitor` section of the WhyLabs dashboard.

![WhyLabs](https://camo.githubusercontent.com/8e9cc18b64b157d4569fa6ed2bd5152200ee7bb1a11e54f858f923a4be635f90/68747470733a2f2f7768796c6162732e61692f5f6e6578742f696d6167653f75726c3d6874747073253341253246253246636f6e74656e742e7768796c6162732e6169253246636f6e74656e74253246696d616765732532463230323225324631312532464672616d652d363839392d2d312d2e706e6726773d3331323026713d3735)

See the [WhyLabs documentation](https://docs.whylabs.ai/docs/) or this [blog post](https://whylabs.ai/blog/posts/ml-monitoring-in-under-5-minutes) for more information on how to use the WhyLabs dashboard.

## Deployment with BentoML

To get your BentoML project ready for deployment, check out the [BentoML documentation](https://docs.bentoml.org/en/latest/).

Add `whylogs[whylabs]` under python packages in the `bentofile.yaml` file to use whylogs and WhyLabs in a production BentoML project.

```yaml
python:
  packages: # Additional pip packages required by the service
    - whylogs[whylabs]
```

## 📚 More Resources

- [Sign up for WhyLabs](https://whylabs.ai/)
- [whylogs Documentation](https://whylogs.readthedocs.io/en/latest/)
- [WhyLabs Documentation](https://docs.whylabs.ai/docs/)
- [BentoML Documentation](https://docs.bentoml.org/en/latest/)
- [WhyLabs Blog](https://whylabs.ai/blog)
