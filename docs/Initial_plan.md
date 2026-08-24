A Python Library for Conditional Feature Importance in Multi-Time Series Forecasting
Introduction
In the domain of time series forecasting, the use of global models to predict multiple time series simultaneously has gained significant traction
 . These models, which are trained on a diverse collection of time series, often demonstrate superior generalization and can adeptly handle scenarios like cold starts for new series
 . However, a persistent challenge is understanding the "why" behind their predictions, a task that falls to feature importance techniques which aim to reveal the variables that most significantly influence the forecast
 .

This research document outlines a plan for implementing a new Python library specifically for conditional feature importance within the context of multi-time series forecasting using global models. A critical consideration is that many features are "conditioned to the series," meaning their values and structure are intrinsically dependent on the specific time series they belong to
 . This dependency introduces complexities that standard feature importance methods often fail to address adequately, as they can create unrealistic data instances by breaking these correlations, leading to misleading insights
 .

The objective here is to explore the foundational concepts, review existing tools with a deeper technical lens, and propose a detailed design and implementation strategy—including working code examples—for a new, specialized Python library to meet this challenge.

Core Concepts
A robust library must be built on a clear understanding of the underlying principles of global modeling and the specific challenges of series-dependent features.

Multi-Time Series Forecasting and Global Models
Multi-Time Series Forecasting: This practice involves predicting future values for a collection of related time series, such as forecasting sales for every product in a store or predicting energy demand across multiple regions
 .

Global Models: Instead of training a separate model for each time series (a local approach), a single "global" model is trained on all the time series data combined
 . This allows the model to learn patterns and relationships that are common across different series
 . Popular architectures for these models include Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTMs), Transformers, and tree-based models like LightGBM
 . Prominent Python libraries like Darts, GluonTS, sktime, and skforecast provide powerful frameworks for building and training these models
 .

Feature Importance in Time Series
Feature importance methods quantify the contribution of each input feature to a model's predictions
 . Common model-agnostic techniques include:

Permutation Feature Importance (PFI): This method measures the increase in a model's prediction error after the values of a single feature are randomly shuffled
 . A large increase in error implies that the model relies heavily on that feature for its accuracy
 . However, standard PFI can be misleading when features are dependent, as it may create unrealistic data instances by breaking these dependencies
 .

Conditional Permutation Feature Importance (cPFI): This technique addresses the limitations of standard PFI by permuting a feature's values only within subgroups where the feature distributions are similar, thus preserving dependencies
 .

SHAP (SHapley Additive exPlanations): Drawing from cooperative game theory, SHAP calculates the marginal contribution of each feature to a specific prediction . It offers both global (average impact) and local (per-prediction) explanations .

Model-Specific Importance: Certain models, particularly tree-based ensembles, provide their own importance metrics based on feature usage (e.g., gain or split count)
 .

The Challenge of Conditional, Series-Dependent Features
The central challenge this research addresses is the proper handling of features that are "conditioned to the series." These are not static, global features but are dynamically generated or have context specific to each time series
 . Key examples include:

Autoregressive Lags: The value of the series at a previous time step (e.g., lag_1 is yesterday's sales for a specific product)
 .

Rolling Statistics: A rolling 7-day average of sales, calculated independently for each product .

Time-Based Features: Features like "day of the week" or "month," which are relative to each series' own timeline
 .

Series Identifiers: A feature explicitly added to the data to tell the model which series a particular data point belongs to. This is crucial for the model to learn series-specific patterns
 .

The problem is that standard feature importance methods can break the inherent structure of these features
 . For instance, randomly shuffling an autoregressive lag_1 feature across the entire dataset would mean pairing a lag from "Product A" with a target from "Product B," creating nonsensical data and rendering the importance score invalid
 . The value of a lag or a rolling mean is only meaningful within the context of its original series.

Existing Libraries and Approaches
Several powerful Python libraries facilitate multi-series forecasting, but their interpretability features require careful consideration for this conditional context.

Darts: A popular and flexible library supporting a wide array of models, from classical ARIMA to deep learning architectures like N-BEATS and Transformers
 . It allows for training on multiple time series with past and future covariates
 . While powerful, its documentation does not focus on built-in methods for conditional feature importance that respect series-dependent feature structures.

sktime & Nixtla Suite: sktime is a comprehensive toolbox for time series analysis, while Nixtla's MLForecast and StatsForecast are optimized for high-performance forecasting . These libraries automate many aspects of feature creation and model training, but like others, applying feature importance in a conditional manner requires a specialized approach not provided out-of-the-box
 .

skforecast: This library excels at using scikit-learn compatible regressors for forecasting and offers robust multi-series capabilities . It is compatible with standard interpretability tools like SHAP and permutation importance
 . Its transparent methodology for preparing data for a global model serves as an excellent foundation for implementing conditional importance techniques. The ForecasterMultiSeries class, for example, generates a training matrix X with a MultiIndex that explicitly tracks the series identifier and timestamp for each row, which is crucial for our purposes .

Designing a New Python Library: Proposed Methodologies and Implementation
A new library should integrate with existing frameworks and directly solve the series-dependency challenge by implementing more advanced, conditional interpretability methods. We will demonstrate the core logic using the skforecast library as a practical foundation.

Setup: Creating a Multi-Series Forecasting Environment
First, we prepare the necessary libraries, data, and a fitted skforecast multi-series forecaster. The training matrix X generated by this forecaster will be the basis for our feature importance calculations .

python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from skforecast.ForecasterMultiSeries import ForecasterMultiSeries
from skforecast.datasets import fetch_dataset

# 1. Load and prepare multi-series data
data = fetch_dataset(name="uci_electricity_demand")
data['date'] = pd.to_datetime(data['date'])
data = data.set_index(['series_id', 'date']).asfreq('H').sort_index()

# Filter for 3 series to keep the example manageable
series_to_use = ['MT_001', 'MT_002', 'MT_003']
data = data[data.index.get_level_values('series_id').isin(series_to_use)]

# 2. Create and train the multi-series forecaster
forecaster = ForecasterMultiSeries(
    regressor=RandomForestRegressor(n_estimators=50, max_depth=10, random_state=123),
    lags=24  # Use 24 lags as predictors
)
forecaster.fit(series=data['demand'])

# 3. Extract the training matrix (X) and target (y)
X, y = forecaster.create_train_X_y()

# 4. Calculate the baseline error for comparison
baseline_pred = forecaster.predict_in_sample()
baseline_mse = mean_squared_error(y, baseline_pred['pred'])
print(f"\nBaseline Model MSE: {baseline_mse}")
Methodology 1: Manual, Dictionary-Based Conditional Permutation Importance
This approach is used when you have predefined groups based on domain knowledge, such as grouping series by geographical location or product category . We define these groups using a Python dictionary and ensure permutations are constrained within these groups.

Step-by-Step Manual Implementation:

Define a Custom Mapping Dictionary: Create a dictionary that maps each unique series identifier to a group label .

python
# Custom dictionary mapping each series_id to a group label
manual_mapping_dict = {
    'MT_001': 'group_A',
    'MT_002': 'group_B',
    'MT_003': 'group_A' # Group MT_001 and MT_003 together
}
Align Group Labels with the Training Matrix: The training matrix X from ForecasterMultiSeries has a MultiIndex containing the series_id (in the 'level' level). We leverage this to create a corresponding array of group labels for every row in X
 .

python
# Get the 'level' (series_id) from the MultiIndex of X
series_ids_for_X = X.index.get_level_values('level')

# Use the .map() method to apply the dictionary to the series_ids
# This creates a pandas Series of group labels perfectly aligned with X
manual_groups = series_ids_for_X.map(manual_mapping_dict)
manual_groups_array = manual_groups.to_numpy()
Implement Permutation and Calculate Importance: We use a function to perform the permutation within the manually defined groups and then calculate the increase in model error
 .

python
def conditional_permutation(X_original: pd.DataFrame, feature_to_permute: str, groups: np.ndarray) -> pd.DataFrame:
    """Permutes the values of a specified feature within conditional groups."""
    rng = np.random.default_rng(123)
    X_permuted = X_original.copy()
    X_permuted['permutation_group'] = groups

    permuted_values = X_permuted.groupby('permutation_group')[feature_to_permute].transform(lambda x: rng.permutation(x))
    X_permuted[feature_to_permute] = permuted_values

    return X_permuted.drop(columns='permutation_group')

# Conditionally permute the feature using the manual groups
X_permuted_manual = conditional_permutation(
    X_original=X,
    feature_to_permute='lag_1',
    groups=manual_groups_array
)

# Predict and calculate the new MSE
permuted_pred_manual = forecaster.regressor.predict(X_permuted_manual.to_numpy())
permuted_mse_manual = mean_squared_error(y, permuted_pred_manual)
importance_score_manual = permuted_mse_manual - baseline_mse

print(f"--- Manual Dictionary-Based PFI Results for 'lag_1' ---")
print(f"Permuted MSE (Manual): {permuted_mse_manual:.4f}")
print(f"Conditional Permutation Importance: {importance_score_manual:.4f}")
Methodology 2: Automated, Tree-Based Conditional Subgroup PFI (cs-PFI)
The core innovation of the proposed library would be an automated implementation of the Conditional Subgroup Permutation Feature Importance (cs-PFI) algorithm
 . This model-agnostic method is more sophisticated because it learns the optimal subgroups for permutation instead of requiring the user to define them
 .

Algorithm & Implementation:

Train a Model to Predict the Feature of Interest: The algorithm conditions the permutation of a feature of interest, Xj (e.g., lag_1), on all other features, X-j
 . To do this, a predictive model, typically a DecisionTreeRegressor, is trained to predict Xj using X-j as predictors
 .

Practical Consideration: The categorical series identifier must be numerically encoded (e.g., via one-hot encoding) for the tree model. To create stable subgroups, the tree should be pruned by setting hyperparameters like max_depth or min_samples_leaf
 .

python
FEATURE_OF_INTEREST = 'lag_1'
y_tree = X[FEATURE_OF_INTEREST]
X_tree_pre = X.drop(columns=[FEATURE_OF_INTEREST]).reset_index()

# One-hot encode the series identifier
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
series_encoded = encoder.fit_transform(X_tree_pre[['level']])
series_encoded_df = pd.DataFrame(series_encoded, columns=encoder.get_feature_names_out(['level']))

# Combine with other features to create the tree's training data
X_tree = pd.concat([
    X_tree_pre.drop(columns=['level', 'date']).reset_index(drop=True),
    series_encoded_df
], axis=1)

# Train the partitioning tree with pruning for stable subgroups
tree_model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=0.05, random_state=123)
tree_model.fit(X_tree, y_tree)
Define Homogeneous Subgroups via Leaf Nodes: The terminal leaf nodes of the trained decision tree define a set of disjoint subgroups
 . We use the tree's .apply() method to get the leaf index for each observation, which serves as our group identifier
 .

python
# Get the leaf node index for each sample; these are our conditional groups
partition_groups = tree_model.apply(X_tree)
Perform Within-Subgroup Permutation: The permutation of feature Xj is then performed with a critical constraint: its values are shuffled only among the observations that fall into the same leaf node (subgroup)
 . This ensures a value is swapped with another from an observation with similar characteristics, preserving the data's structural integrity
 .

Calculate Final Importance Score: The model's prediction error is calculated on this conditionally permuted dataset. The final cs-PFI score is the increase in error
 .

python
# Reuse the conditional_permutation function with the learned groups
X_permuted_tree = conditional_permutation(
    X_original=X,
    feature_to_permute=FEATURE_OF_INTEREST,
    groups=partition_groups
)

# Predict and calculate the new MSE
permuted_pred_tree = forecaster.regressor.predict(X_permuted_tree.to_numpy())
permuted_mse_tree = mean_squared_error(y, permuted_pred_tree)
importance_score_tree = permuted_mse_tree - baseline_mse

print(f"\n--- Automated cs-PFI Results for '{FEATURE_OF_INTEREST}' ---")
print(f"Permuted MSE (Tree-based): {permuted_mse_tree:.4f}")
print(f"Conditional Permutation Importance: {importance_score_tree:.4f}")
Methodology 3: Conditional SHAP
Adapting SHAP requires carefully constructing the "background" dataset used to compute expected values. The background data should reflect the specific context of the prediction being explained.

Algorithm:

To explain a prediction for a specific instance, first identify its series ID.

Construct the background dataset for the shap.KernelExplainer by sampling only from other instances belonging to the same series.

This provides a more accurate and relevant baseline for what feature values are "typical" for that particular series, leading to more meaningful SHAP values.

For a global explanation, SHAP values can be computed for a representative sample of predictions across different time series and then aggregated.

Proposed API Design
A potential API would abstract the complexity of both the manual and automated methods into a simple, user-friendly interface.

python
# Fictional API for the proposed library
import conditional_feature_importance as cfi

# Assume 'global_model' is a trained skforecast or Darts model
# 'X_train' and 'y_train' are the stacked matrices from the global model training

# --- Automated Conditional Subgroup Permutation Importance (cs-PFI) ---
pfi_explainer = cfi.ConditionalPermutationImportance(
    model=global_model,
    metric='mse',
    strategy='auto' # 'auto' uses the tree-based cs-PFI
)
# The explainer internally handles the cs-PFI logic for each feature
pfi_importance = pfi_explainer.compute(X_train, y_train, features=['lag_1', 'day_of_week'])
pfi_explainer.plot(pfi_importance)

# --- Manual Conditional Permutation Importance ---
# 'groups' is the pre-defined group label for each row in X_train
manual_pfi_explainer = cfi.ConditionalPermutationImportance(
    model=global_model,
    metric='mse',
    strategy='manual'
)
manual_importance = manual_pfi_explainer.compute(X_train, y_train, groups=manual_groups_array)
manual_pfi_explainer.plot(manual_importance)

# --- Conditional SHAP ---
shap_explainer = cfi.ConditionalSHAP(
    model=global_model.predict,
    data=X_train,
    series_id_col='_level_skforecast' # Column to identify series
)
# Explain a specific instance with a series-specific background
shap_values = shap_explainer.explain_instance(X_train.iloc[...](asc_slot://start-slot-78))
shap_explainer.plot_instance(shap_values)
Implementation Considerations
Data Structures: The library must efficiently handle standard multi-series data formats, primarily pandas DataFrames with MultiIndex, as produced by libraries like skforecast
 .

Scalability: Feature importance calculations are computationally intensive. The library should leverage parallel processing via tools like joblib or dask to distribute computations, especially for the cs-PFI algorithm.

Extensibility: A modular design will be crucial for adding new conditional importance methods or supporting more forecasting model types in the future.

Executive Summary
The growing adoption of global models for multi-time series forecasting has created a critical need for specialized interpretability tools. Standard feature importance methods are ill-suited for this context because they fail to respect the conditional nature of series-dependent features (like lags and series identifiers), leading to unrealistic data permutations and unreliable insights
 .

This research proposes the development of a new Python library dedicated to conditional feature importance. The feasibility of this proposal is demonstrated through detailed, practical implementations of two core methodologies. The first is a manual, dictionary-based PFI, which allows users to inject domain knowledge by defining explicit permutation groups
 . The second, and the core innovation, is the automated Tree-Based Conditional Subgroup Permutation Feature Importance (cs-PFI) algorithm
 . This advanced, model-agnostic technique addresses feature dependencies by training a decision tree to learn homogeneous data subgroups and then constraining permutations within those groups
 . By including the series identifier as a feature in this process, cs-PFI can naturally handle the series-specific conditioning that is central to the multi-series forecasting problem
 . This is complemented by a proposed conditional approach to SHAP, which uses series-specific background data for more relevant explanations.

By providing a user-friendly API that automates these advanced techniques, integrates with popular forecasting frameworks like skforecast, leverages parallel processing for scalability, and produces clear visualizations, this new library would empower data scientists to gain deeper, more accurate insights into the behavior of their global forecasting models. This advancement will foster more trustworthy models, guide better feature engineering, and ultimately enable more informed, data-driven decision-making.
