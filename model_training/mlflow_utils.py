import mlflow


def set_or_create_experiment(experiment_name: str) -> str:
    """
    Create a new mlflow experiment with the given name and artifact location.

    Parameters:
    ----------
    experiment_name: (str)
        The name of the experiment to create.

    Returns:
    -------
    experiment_id: (str)
        The id of the created experiment.
    """
    # creating the experiement
    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except Exception:
        experiment_id = mlflow.create_experiment(experiment_name)
    finally:
        # set the experiement as the active experiement
        mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id


def get_mlflow_experiment(
    experiment_id: str = None, experiment_name: str = None
) -> mlflow.entities.Experiment:
    """
    Retrieve the mlflow experiment with the given id or name.

    Parameters:
    ----------
    experiment_id: (str)
        The id of the experiment to retrieve.

    experiment_name: (str)
        The name of the experiment to retrieve.

    Returns:
    -------
    experiment: (mlflow.entities.Experiment)
        The mlflow experiment with the given id or name.
    """
    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        raise ValueError(
            """
                Either experiment_id or experiment_name must be provided.
            """
        )

    return experiment


def register_model(model_name: str, run_id: str, artifact_path: str):
    """
    Register a model.

    Parameters:
    ------
    model_name: (str)
        Name of the model.

    run_id: (str)
        Run ID.

    artifact_path: (str)
        Artifact path.

    return:
    ------
        None.
    """
    # creating the mlflow client
    client = mlflow.tracking.MlflowClient()

    # creating the registry model
    client.create_registered_model(model_name)

    # versioning the model
    client.create_model_version(
        name=model_name, source=f"runs:/{run_id}/{artifact_path}", run_id=run_id
    )
