import os
import whylogs as why
from whylogs.api.writer.whylabs import WhyLabsWriter
import pandas as pd


from sklearn import datasets

data_iris = datasets.load_iris(as_frame=True)

# Load training data set
# iris = datasets.load_iris()
X, y = data_iris.data, data_iris.target


# Set WhyLabs environment variables
os.environ['WHYLABS_API_KEY'] = 'APIKEY'
os.environ["WHYLABS_DEFAULT_ORG_ID"] = 'ORGID'
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = 'PROJECTID'

ref_profile = why.log(X).profile()
writer = WhyLabsWriter().option(reference_profile_name="iris_training_profile")
writer.write(file=ref_profile.view())

print("Logged Training Data Profile to WhyLabs")