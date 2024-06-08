import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("linear-regression-experiment")

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """

    dv, lr = data

    # Save the DictVectorizer as an artifact
    with open("dict_vectorizer.pkl", "wb") as f:
        import pickle
        pickle.dump(dv, f)

    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, "model")
        mlflow.log_artifact("dict_vectorizer.pkl", artifact_path="model")