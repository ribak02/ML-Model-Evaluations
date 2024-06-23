import matplotlib.pyplot as plt
import numpy as np

cv_scores = {
    'Random Forest': [0.71944444, 0.72382155, 0.72390572, 0.71734007, 0.71868687],
    'Logistic Regression': [0.70210438, 0.70765993, 0.70286195, 0.69882155, 0.70227273],
    'Gradient Boosting': [0.70345118, 0.71254209, 0.71607744, 0.6776936, 0.70917508],
    'Hist Gradient Boosting': [0.71877104, 0.71809764, 0.72306397, 0.71321549, 0.72171717],
    'MLP Classifier': [0.6209596, 0.5986532, 0.60984848, 0.63038721, 0.5986532]
}

# Calculate mean CV accuracy for each model
mean_cv_scores = {model: np.mean(scores) for model, scores in cv_scores.items()}

# Calculate standard deviation for CV accuracy for each model
std_cv_scores = {model: np.std(scores) for model, scores in cv_scores.items()}

# Plotting
plt.figure(figsize=(10, 6))
models = list(mean_cv_scores.keys())
mean_scores = list(mean_cv_scores.values())
std_scores = list(std_cv_scores.values())

# Error bar for standard deviation
plt.bar(models, mean_scores, yerr=std_scores, capsize=5, color='skyblue')
plt.xlabel('Model')
plt.ylabel('Mean CV Accuracy')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Saving the plot
plt.savefig('images/model_comparison.png')

# Saving statistics to a CSV file
import pandas as pd
df_stats = pd.DataFrame({
    "Model": models,
    "Mean CV Accuracy": mean_scores,
    "Std Deviation": std_scores
})
df_stats.to_csv('model_cv_stats.csv', index=False)

# Prepare the data for plotting
models = list(cv_scores.keys())
all_scores = np.array(list(cv_scores.values()))

# Plotting the distribution of CV scores for each model
plt.figure(figsize=(14, 6))
plt.boxplot(all_scores.T, labels=models, showmeans=True)
plt.xlabel('Model')
plt.ylabel('CV Accuracy')
plt.title('Distribution of CV Accuracy Scores by Model')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('cv_score_distribution.png')
plt.show()

# Adding ranking based on Mean CV Accuracy
mean_cv_scores = {model: np.mean(scores) for model, scores in cv_scores.items()}
sorted_mean_cv_scores = dict(sorted(mean_cv_scores.items(), key=lambda item: item[1], reverse=True))

print("Ranking of Models by Mean CV Accuracy:")
for rank, (model, score) in enumerate(sorted_mean_cv_scores.items(), start=1):
    print(f"{rank}. {model}: {score:.4f}")

# Saving the detailed statistics and rankings to a CSV file
df_stats = pd.DataFrame({
    "Model": models,
    "Mean CV Accuracy": [mean_cv_scores[model] for model in models],
    "Std Deviation": [np.std(cv_scores[model]) for model in models],
    "Rank": [sorted(models, key=lambda x: mean_cv_scores[x], reverse=True).index(x)+1 for x in models]
})
df_stats.to_csv('model_detailed_stats_and_ranking.csv', index=False)