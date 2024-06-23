import numpy as np
import optuna
import json
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from optuna.pruners import MedianPruner

from part1 import load_data, preprocess_data

def objective(trial, model_type):
    train_input_path = 'data/training-set-values.csv'
    train_labels_path = 'data/training-set-labels.csv'
    test_input_path = 'data/test-set-values.csv'

    # Load and preprocess data
    train_df, train_labels, _ = load_data(train_input_path, train_labels_path, test_input_path)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels['status_group'])
    num_preprocessing = 'StandardScaler' if model_type == 'LogisticRegression' else 'None'
    cat_preprocessing = 'TargetEncoder'
    preprocessor = preprocess_data(train_df, _, num_preprocessing, cat_preprocessing)

    # Model-specific configurations
    if model_type == 'RandomForestClassifier':
        n_estimators = trial.suggest_int('n_estimators', 50, 400)
        max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 14)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 8)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
    elif model_type == 'LogisticRegression':
        C = trial.suggest_float("C", 1e-10, 1e10, log=True)
        model = LogisticRegression(C=C, max_iter=1000, solver='liblinear', random_state=42)

    # Full pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    score = cross_val_score(pipeline, train_df, y_train, n_jobs=-1, cv=3)
    return np.mean(score)

def main():
    parser = argparse.ArgumentParser(description='Optimize hyperparameters for a specified model.')
    parser.add_argument('--model', type=str, choices=['LogisticRegression', 'RandomForestClassifier'], required=True, help='Model type to optimize.')
    args = parser.parse_args()

    study = optuna.create_study(direction='maximize', pruner=MedianPruner())
    if args.model == 'RandomForestClassifier':
        study.optimize(lambda trial: objective(trial, 'RandomForestClassifier'), n_trials=50)
    else:
        study.optimize(lambda trial: objective(trial, 'LogisticRegression'), n_trials=50)

    best_params = study.best_trial.params
    best_params_filename = f"{args.model}_best_hyperparams.json"
    with open(best_params_filename, 'w') as f:
        json.dump(best_params, f)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:', best_params)

    # Save plots to files
    optimization_history_filename = f"{args.model}_optimization_history.png"
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image('images/'+ optimization_history_filename)

    param_importances_filename = f"{args.model}_param_importances.png"
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image('images/'+ param_importances_filename)

    # Save the study to a file for later analysis
    study.trials_dataframe().to_csv('optuna_study_results.csv', index=False)

if __name__ == "__main__":
    main()
