# mlflow-s1
The task of first sprint: use of mlflow for logging metrics and parameters along with the use of model registry, and fetching the model from registry to integrate it with flask api.

To run this project, follow the following steps;
1. Create and activate virtual environment
    ```
        poetry install
    ```

2.  Run the main file
    ``` python
        python main.py
    ```

    This will start a flask backend server, and we can test it in postman.

All the training code can be found inside model_training folder.
