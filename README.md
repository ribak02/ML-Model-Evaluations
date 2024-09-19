# ML-Model-Evaluations

This project evaluates different machine learning models on the dataset and explores hyper-parameter optimization for improving model performance.

## Data Preprocessing

Data preprocessing transforms the raw dataset into a more suitable format for machine learning models, improving accuracy, efficiency, and effectiveness. Below are the key preprocessing steps undertaken for the "Pump it Up: Data Mining the Water Table" competition dataset.

### Categorical Feature Handling

The dataset contains several categorical features with varying levels of cardinality, and handling them properly is crucial for model performance:

- **OneHotEncoder**: Applied to categorical variables with low cardinality, this method transforms category values into new binary columns, enabling the model to understand the presence or absence of categories.
- **OrdinalEncoder**: Used for ordinal data where the order of categories matters, this encoder converts string labels to integer codes while preserving the inherent order of the categories.
- **TargetEncoder**: Employed for high cardinality features, target encoding replaces a categorical value with a probability blend, helping encode categorical features more effectively without causing data leakage or overfitting.

A custom preprocessing pipeline was created to handle target encoding, including missing value handling and the application of the `TargetEncoder` to prevent leakage.

### Dealing with Missing Values

Missing data can impact model performance significantly, so two strategies were employed:

- **Numerical Data**: A `SimpleImputer` was used to replace missing values with the median, ensuring robustness to outliers.
- **Categorical Data**: Missing values were replaced with the most frequent category using `SimpleImputer`.

### Scaling Numerical Values

Numerical features were scaled using `StandardScaler`, which removes the mean and scales features to unit variance. This step is critical for models like Logistic Regression, where feature scaling influences convergence.

### Datetime Feature Handling

To capture valuable temporal information, a `DateTransformer` was applied to extract the year from datetime features, focusing on long-term trends or effects like the age of water points.

### Other Transformers

- **DataFrameSelector**: A custom transformer to select subsets of data for specific processing pipelines based on their type.
- **CustomPreprocessor**: Designed for categorical data preparation before target encoding, ensuring consistency in handling categorical columns.

### Label Encoding

The target variable (water point functionality) was categorical, and `LabelEncoder` was used to convert it into a numerical format for model training.

## Design

The design of the preprocessing pipeline and model hyper-parameters significantly influenced the performance of machine learning models on the competition dataset.

### Preprocessing Method Parameters

- **Scaling Numerical Values**: Numerical features were scaled using `StandardScaler` to normalize the data, crucial for algorithms like Logistic Regression and MLPClassifier.
- **Encoding Categorical Features**: Depending on feature characteristics:
  - **OneHotEncoder** for low cardinality categorical features.
  - **OrdinalEncoder** for ordinal features.
  - **TargetEncoder** for high cardinality features to reduce dimensionality and overfitting risks.
- **Handling Missing Values**: Numerical features were imputed with the median, while categorical features were filled with the mode (most frequent category).

## Model Hyper-parameters

### RandomForestClassifier

Hyper-parameters like `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf` were optimized:

- `n_estimators` (50-400) controls the number of trees, affecting generalization and computational cost.
- `max_depth` limits tree depth to prevent overfitting.
- `min_samples_split` and `min_samples_leaf` ensure the trees do not grow too complex, improving generalization.

### LogisticRegression

The key hyper-parameter optimized was `C`, the inverse regularization strength:

- Regularization prevents overfitting in high-dimensional datasets.
- `C` was varied over a range from 1e-10 to 1e10 to find the optimal balance between complexity and generalization.

### GradientBoostingClassifier & HistGradientBoostingClassifier

For these models, critical hyper-parameters like `n_estimators` and `learning_rate` were tuned:

- A higher `n_estimators` can improve performance but risks overfitting.
- The `learning_rate` controls the contribution of each tree, with smaller rates requiring more trees but potentially leading to a more robust model.

---

This project illustrates the impact of data preprocessing and hyper-parameter tuning in achieving optimal machine learning performance. Each preprocessing step and hyper-parameter choice was carefully considered to balance accuracy, generalization, and computational efficiency.
