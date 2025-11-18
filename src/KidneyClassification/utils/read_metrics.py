import os

def load_mlflow_metrics(run_folder="mlruns/0"):
    latest_run = None
    latest_time = 0

    # Find the most recent finished MLflow run
    for run_id in os.listdir(run_folder):
        run_path = os.path.join(run_folder, run_id, "metrics")
        if os.path.isdir(run_path):
            mtime = os.path.getmtime(run_path)
            if mtime > latest_time:
                latest_time = mtime
                latest_run = run_path

    if not latest_run:
        return {}

    def read_value(metric_name):
        file_path = os.path.join(latest_run, metric_name)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return float(f.read().strip())
        return None

    return {
        "accuracy": read_value("accuracy"),
        "loss": read_value("loss"),
        "precision": read_value("precision"),
        "recall": read_value("recall"),
        "f1": read_value("f1")
    }
