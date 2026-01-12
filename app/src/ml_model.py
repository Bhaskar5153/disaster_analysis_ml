# Create necessary directories
import os
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import pickle

# Create 'ml_logs' directory for logging
log_dir = 'ml_logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'disaster_analysis.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting disaster event analysis notebook.")

# Create a dummy data directory and CSV for demonstration if not already present

df = pd.read_csv(r"app\data\disaster_events.csv")
# print(df)

##  Exploratory Data Analysis (EDA)

# Exploratory Data Analysis (EDA) is a crucial step to understand the underlying structure of the data, identify patterns, detect outliers, and uncover relationships between variables. This section will provide an overview of the dataset's characteristics.
if not df.empty:
    logger.info("Starting EDA: Displaying basic statistics and checking for missing values.")

    # Display descriptive statistics for numerical columns
    print("\nDescriptive Statistics:")
    print(df.describe())

    # Check for missing values
    print("\nMissing Values Count:")
    print(df.isnull().sum())
    missing_percentage = df.isnull().sum() / len(df) * 100
    print("\nMissing Values Percentage:")
    print(missing_percentage)

    # Check unique values for categorical columns
    print("\nUnique values for categorical columns:")
    for col in ['disaster_type', 'location', 'aid_provided']:
        if col in df.columns:
            print(f"- {col}: {df[col].nunique()} unique values")
            print(f"  Top 5: {df[col].value_counts().head()}")
        else:
            logger.warning(f"Categorical column '{col}' not found in DataFrame.")

    # Analyze target variable distribution
    print("\nDistribution of Target Variable (is_major_disaster):")
    print(df['is_major_disaster'].value_counts())
    print(df['is_major_disaster'].value_counts(normalize=True) * 100)

    # Check for class imbalance
    if df['is_major_disaster'].value_counts(normalize=True)[0] > 0.75 or df['is_major_disaster'].value_counts(normalize=True)[1] > 0.75:
        logger.warning("Target variable 'is_major_disaster' appears to be imbalanced. This will be addressed during modeling.")
    else:
        logger.info("Target variable 'is_major_disaster' does not show significant imbalance.")

else:
    logger.warning("DataFrame is empty. Skipping EDA.")


##  Preprocessing and Outlier Handling

# This section focuses on preparing the data for machine learning. This includes handling missing values, converting data types, creating new features (feature engineering), encoding categorical variables, and managing outliers to ensure the model performs optimally.

### Handling Missing Values

# We will use different strategies for imputing missing values based on the column type. Numerical columns like `estimated_economic_loss_usd`, `response_time_hours`, and `affected_population` will be imputed with the median, which is robust to outliers.
if not df.empty:
    logger.info("Starting data preprocessing and outlier handling.")

    # Handle missing values
    try:
        # Numerical columns to impute with median
        numerical_cols_to_impute = ['estimated_economic_loss_usd', 'response_time_hours', 'affected_population']
        for col in numerical_cols_to_impute:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Missing values in '{col}' imputed with median: {median_val}")
            elif col not in df.columns:
                logger.warning(f"Column '{col}' not found for imputation.")

        # Categorical columns to impute with mode (if any had NaNs, though none observed in schema)
        categorical_cols_to_impute = ['disaster_type', 'location', 'aid_provided']
        for col in categorical_cols_to_impute:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"Missing values in '{col}' imputed with mode: {mode_val}")
            elif col not in df.columns:
                logger.warning(f"Column '{col}' not found for imputation.")

        print("\nMissing Values After Imputation:")
        print(df.isnull().sum())
    except Exception as e:
        logger.error(f"Error during missing value handling: {e}")

    # Convert 'date' column to datetime and extract features
    try:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        logger.info("Extracted year, month, day, and day of week from 'date' column.")
    except KeyError:
        logger.error("'date' column not found in DataFrame for feature engineering.")
    except Exception as e:
        logger.error(f"Error during date feature engineering: {e}")

    # Outlier Handling (using IQR method for numerical features)
    # Numerical columns for outlier detection
    numerical_features_for_outliers = ['affected_population', 'estimated_economic_loss_usd',
                                       'response_time_hours', 'infrastructure_damage_index', 'severity_level']

    for col in numerical_features_for_outliers:
        if col in df.columns:
            try:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap outliers instead of removing them to retain data points
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                logger.info(f"Outliers in '{col}' capped using IQR method. Lower: {lower_bound:.2f}, Upper: {upper_bound:.2f}")
            except Exception as e:
                logger.error(f"Error handling outliers for column '{col}': {e}")
        else:
            logger.warning(f"Column '{col}' not found for outlier handling.")

    logger.info("Preprocessing and outlier handling completed.")
else:
    logger.warning("DataFrame is empty. Skipping preprocessing and outlier handling.")


##  Visual Representation of EDA (Plotly)

# Visualizations are key to understanding the data at a deeper level. We will use Plotly to create interactive plots that highlight distributions, relationships, and potential issues within the dataset.
if not df.empty:
    logger.info("Generating Plotly visualizations for EDA.")

    # 1. Distribution of Disaster Types
    try:
        fig = px.bar(df['disaster_type'].value_counts().reset_index(),
                     x='index', y='disaster_type',
                     title='Distribution of Disaster Types',
                     labels={'index': 'Disaster Type', 'disaster_type': 'Count'})
        fig.show()
    except Exception as e:
        logger.error(f"Error generating disaster type distribution plot: {e}")

    # 2. Distribution of Severity Levels
    try:
        fig = px.histogram(df, x='severity_level', nbins=10,
                           title='Distribution of Severity Levels',
                           labels={'severity_level': 'Severity Level'})
        fig.show()
    except Exception as e:
        logger.error(f"Error generating severity level distribution plot: {e}")

    # 3. Affected Population Distribution
    try:
        fig = px.histogram(df, x='affected_population', nbins=50,
                           title='Distribution of Affected Population',
                           labels={'affected_population': 'Affected Population'})
        fig.show()
    except Exception as e:
        logger.error(f"Error generating affected population distribution plot: {e}")

    # 4. Estimated Economic Loss Distribution (Box Plot for Outliers)
    try:
        fig = px.box(df, y='estimated_economic_loss_usd',
                     title='Box Plot of Estimated Economic Loss (USD)',
                     labels={'estimated_economic_loss_usd': 'Economic Loss (USD)'})
        fig.show()
    except Exception as e:
        logger.error(f"Error generating economic loss box plot: {e}")

    # 5. Geographic Distribution of Disasters
    try:
        fig = px.scatter_geo(df, lat='latitude', lon='longitude', color='disaster_type',
                             hover_name='location',
                             title='Geographic Distribution of Disaster Events by Type',
                             projection='natural earth')
        fig.show()
    except Exception as e:
        logger.error(f"Error generating geographic distribution plot: {e}")

    # 6. Average Estimated Economic Loss by Disaster Type
    try:
        avg_loss_by_type = df.groupby('disaster_type')['estimated_economic_loss_usd'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(avg_loss_by_type, x='disaster_type', y='estimated_economic_loss_usd',
                     title='Average Estimated Economic Loss by Disaster Type',
                     labels={'disaster_type': 'Disaster Type', 'estimated_economic_loss_usd': 'Average Economic Loss (USD)'})
        fig.show()
    except Exception as e:
        logger.error(f"Error generating average economic loss by disaster type plot: {e}")

    # 7. Target Variable Distribution
    try:
        target_counts = df['is_major_disaster'].value_counts(normalize=True).reset_index()
        target_counts.columns = ['is_major_disaster', 'percentage']
        fig = px.pie(target_counts, names='is_major_disaster', values='percentage',
                     title='Distribution of Major vs. Minor Disasters',
                     labels={'is_major_disaster': 'Major Disaster (1) / Minor Disaster (0)'},
                     hole=0.3)
        fig.update_traces(textinfo='percent+label', marker=dict(colors=['#EF553B', '#636EFB'])) # Customize colors
        fig.show()
    except Exception as e:
        logger.error(f"Error generating target variable distribution plot: {e}")

    logger.info("Plotly visualizations completed.")
else:
    logger.warning("DataFrame is empty. Skipping Plotly visualizations.")

# **Explanation of Plots:**

# 1.  **Distribution of Disaster Types**: This bar chart shows the frequency of each `disaster_type` in the dataset. It helps us understand which types of disasters are most prevalent and informs us about potential class imbalances in this categorical feature. For example, 'Wildfire' or 'Drought' might be more common than 'Volcanic Eruption'.

# 2.  **Distribution of Severity Levels**: This histogram illustrates the frequency of different `severity_level` ratings. It indicates whether most disasters are low, medium, or high severity, providing insight into the overall impact profile of the events.

# 3.  **Distribution of Affected Population**: This histogram displays the distribution of the `affected_population`. It helps us identify typical population impact ranges and highlights any events with exceptionally high or low affected populations. The shape of the distribution can indicate skewness or multiple modes.

# 4.  **Box Plot of Estimated Economic Loss (USD)**: This box plot visualizes the spread and central tendency of `estimated_economic_loss_usd`. The box represents the interquartile range (IQR), the line inside is the median, and the "whiskers" extend to data points within 1.5 times the IQR. Points beyond the whiskers are potential outliers, which we've handled in the preprocessing step. It helps confirm the effectiveness of outlier capping.

# 5.  **Geographic Distribution of Disaster Events by Type**: This scatter plot on a world map shows the `latitude` and `longitude` of each disaster, colored by `disaster_type`. This visualization reveals geographical patterns and clusters of certain disaster types, which can be valuable for understanding regional vulnerabilities.

# 6.  **Average Estimated Economic Loss by Disaster Type**: This bar chart compares the average economic loss across different `disaster_type`s. It helps identify which disaster types are typically more costly, offering insights into their financial impact.

# 7.  **Distribution of Major vs. Minor Disasters**: This pie chart (or bar chart) shows the proportion of major disasters (`is_major_disaster` = 1) versus minor disasters (`is_major_disaster` = 0). It is crucial for checking class imbalance in our target variable, which is important for model selection and evaluation strategies.

## 6. Visual Representation of Correlation and Covariance

# Correlation measures the linear relationship between two variables, indicating both the strength and direction of the relationship. Covariance, while also measuring the relationship, indicates the direction of the linear relationship and is sensitive to the scale of the variables. For feature selection and understanding relationships, correlation (especially Pearson correlation) is generally more informative as it's normalized.

if not df.empty:
    logger.info("Calculating and visualizing correlation matrix.")

    # Select only numerical columns for correlation calculation
    numerical_df = df.select_dtypes(include=np.number)
    
    try:
        correlation_matrix = numerical_df.corr()

        # Plotting the correlation heatmap using Plotly
        fig = px.heatmap(correlation_matrix,
                         x=correlation_matrix.columns,
                         y=correlation_matrix.columns,
                         color_continuous_scale='RdBu_r', # Red-Blue reversed color scale
                         title='Correlation Matrix of Numerical Features')

        # Customize annotations to show correlation values
        fig.update_layout(
            autosize=False,
            width=800,
            height=800,
            xaxis_showgrid=False,
            yaxis_showgrid=False
        )
        fig.update_traces(text=np.round(correlation_matrix.values, 2), texttemplate="%{text}")
        fig.show()
    except Exception as e:
        logger.error(f"Error generating correlation heatmap: {e}")

    # Brief discussion on covariance (no direct plot for clarity, as correlation is preferred)
    try:
        # Covariance matrix calculation (for conceptual understanding)
        # covariance_matrix = numerical_df.cov()
        # logger.info("Covariance matrix calculated (not displayed due to size/interpretability for many features).")
        pass
    except Exception as e:
        logger.error(f"Error calculating covariance matrix: {e}")

    logger.info("Correlation visualization completed.")
else:
    logger.warning("DataFrame is empty. Skipping correlation visualization.")


# **Explanation of the Correlation Heatmap:**

# The heatmap displays the Pearson correlation coefficients between all pairs of numerical features in our dataset.
# *   **Color Scale**: The `RdBu_r` (Red-Blue reversed) color scale is used. Red shades indicate positive correlation (as one variable increases, the other tends to increase). Blue shades indicate negative correlation (as one variable increases, the other tends to decrease). White or light colors indicate weak or no linear correlation.
# *   **Values**: The numbers within each cell represent the correlation coefficient, ranging from -1 to +1.
#     *   `+1`: Perfect positive linear correlation.
#     *   `-1`: Perfect negative linear correlation.
#     *   `0`: No linear correlation.
# *   **Diagonal**: The diagonal elements are always 1, as a variable is perfectly correlated with itself.
# *   **Symmetry**: The matrix is symmetric, meaning the correlation between A and B is the same as between B and A.

# **Key Observations and Interpretation:**
# *   **Target Variable (`is_major_disaster`) Correlations**: We pay close attention to the last row/column corresponding to `is_major_disaster`. This helps identify which features have the strongest positive or negative linear relationship with whether a disaster is major. For instance, a high positive correlation with `severity_level` would suggest higher severity levels are linked to major disasters.
# *   **Feature-Feature Correlations**: We look for highly correlated features (e.g., |correlation| > 0.8). If two features are highly correlated, they might be conveying redundant information. In such cases, we might consider keeping only one of them to reduce multicollinearity, which can be an issue for some models (e.g., Logistic Regression).
# *   **Latitude/Longitude**: Their correlations with other features might indicate regional patterns of certain disaster characteristics.
# *   **Date-related features (`year`, `month`, `day`, `day_of_week`)**: Their correlations can reveal temporal trends. For example, if 'month' is correlated with `is_major_disaster`, it implies certain months are more prone to major disasters.

# **Covariance vs. Correlation**:
# While correlation is good for understanding the strength and direction of linear relationships irrespective of the scale of variables, covariance is scale-dependent. A large covariance only indicates a strong linear relationship if the variables themselves have large values. For this reason, correlation is generally preferred for feature selection and interpretability in most machine learning contexts, as it provides a standardized measure.


## 7. Feature Selection Based on EDA

# Based on our EDA and correlation analysis, we will select relevant features for our model. We aim to choose features that show a strong relationship with the target variable, minimize multicollinearity, and are conceptually relevant.

if not df.empty:
    logger.info("Performing feature selection based on EDA and correlation analysis.")

    # Drop 'event_id' as it's just an identifier and 'date' as we've extracted features
    features_to_drop = ['event_id', 'date']
    
    # Identify highly correlated features that might be redundant (e.g., if two features have >0.9 correlation)
    # For now, let's keep all features that seem relevant. If the correlation matrix showed extremely high correlation
    # between two features (e.g., > 0.95), we might consider dropping one.
    # From the sample data structure, no immediately obvious extremely high multicollinearity without a full matrix.

    # Features selected for the model
    # Include numerical, date-derived, and categorical features
    selected_features = [
        'disaster_type', 'location', 'latitude', 'longitude', 'severity_level',
        'affected_population', 'estimated_economic_loss_usd', 'response_time_hours',
        'aid_provided', 'infrastructure_damage_index',
        'year', 'month', 'day', 'day_of_week'
    ]

    # Ensure all selected features are actually in the DataFrame
    selected_features = [f for f in selected_features if f in df.columns]
    
    # Target variable
    target_variable = 'is_major_disaster'

    logger.info(f"Selected features for modeling: {selected_features}")
    logger.info(f"Target variable: {target_variable}")
else:
    logger.warning("DataFrame is empty. Skipping feature selection.")


##  Separate the Selected Features for Training

# Here, we'll separate our dataset into features (X) and the target variable (y). Then, we will split these into training and testing sets. This allows us to train our model on one part of the data and evaluate its performance on unseen data, providing a more realistic assessment of its generalization capability.

# ### Why selected features are taken:
# *   **`disaster_type`**: Different disaster types inherently have varying probabilities of being 'major'. For example, hurricanes or large-scale floods are often major.
# *   **`location`**: Geographical context can be critical. Some regions are more prone to severe events or have less robust infrastructure/response, influencing whether an event becomes major.
# *   **`latitude`, `longitude`**: These numerical representations of location can capture continuous geographical patterns that categorical `location` might miss.
# *   **`severity_level`**: This is a direct measure of the disaster's intensity, logically strongly correlated with whether it's a major disaster.
# *   **`affected_population`**: A higher number of affected people generally indicates a larger-scale, more significant event, often qualifying it as major.
# *   **`estimated_economic_loss_usd`**: Significant financial damage is a key indicator of a major disaster.
# *   **`response_time_hours`**: While a direct causal link to `is_major_disaster` might not be strong (it's often a consequence), longer response times could correlate with factors making a disaster more severe or prolonged.
# *   **`aid_provided`**: Whether aid was provided could be an indicator of the event's scale and need, although it could also be a result.
# *   **`infrastructure_damage_index`**: High infrastructure damage is a clear sign of a significant, major disaster.
# *   **`year`, `month`, `day`, `day_of_week`**: Temporal features can capture trends, seasonality, or specific times when certain disaster types are more prone to being major. For example, hurricane season months or years with specific climate patterns.

if not df.empty and target_variable in df.columns:
    try:
        X = df[selected_features]
        y = df[target_variable]
        logger.info(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")

        # Explicitly define categorical features (only true categorical columns)
        categorical_features = ['disaster_type', 'location']

        # Clean categorical columns: cast to string and fill missing values
        for col in categorical_features:
            if col in X.columns:
                X[col] = X[col].astype(str).fillna("Unknown")

        # Numerical features: all numeric columns except those not meant for scaling
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        numerical_features = [
            f for f in numerical_features
            if f not in ['year', 'month', 'day', 'day_of_week', 'severity_level']
        ]

        # Build preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'  # keep other features like year, month, severity_level
        )

        logger.info(f"Categorical features for encoding: {categorical_features}")
        logger.info(f"Numerical features for scaling: {numerical_features}")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(
            f"Data split into training (X_train: {X_train.shape}, y_train: {y_train.shape}) "
            f"and testing (X_test: {X_test.shape}, y_test: {y_test.shape}) sets."
        )
        logger.info(f"Training set major disaster ratio: {y_train.value_counts(normalize=True)[1]:.2f}")
        logger.info(f"Test set major disaster ratio: {y_test.value_counts(normalize=True)[1]:.2f}")

    except KeyError as e:
        logger.error(f"Missing a selected feature or target column during data separation: {e}")
        X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
    except Exception as e:
        logger.error(f"An error occurred during data separation or preprocessing setup: {e}")
        X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()
else:
    logger.warning("DataFrame is empty or target variable is missing. Skipping data separation.")
    X_train, X_test, y_train, y_test = pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series()

## 9. Modeling

# Given that our target variable `is_major_disaster` is binary (0 or 1), this is a **binary classification** task. We will choose two appropriate models for this problem:
# 1.  **Logistic Regression**: A simple yet powerful linear model suitable for binary classification. It models the probability of a binary outcome. It's often a good baseline and provides coefficients that are interpretable.
# 2.  **Random Forest Classifier**: An ensemble learning method that builds multiple decision trees and merges their predictions. It's robust to overfitting, handles non-linear relationships well, and can capture complex interactions between features. It's generally a strong performer.

# We will use a `Pipeline` to streamline the preprocessing and model training steps.

if not X_train.empty:
    logger.info("Starting model training for Logistic Regression and Random Forest Classifier.")

    # Initialize models
    log_reg = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced') # 'balanced' to handle potential imbalance
    rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced') # 'balanced' to handle potential imbalance

    # Create pipelines
    try:
        pipeline_log_reg = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('classifier', log_reg)])
        pipeline_rf_clf = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('classifier', rf_clf)])

        logger.info("Training Logistic Regression model...")
        pipeline_log_reg.fit(X_train, y_train)
        logger.info("Logistic Regression model trained.")

        logger.info("Training Random Forest Classifier model...")
        pipeline_rf_clf.fit(X_train, y_train)
        logger.info("Random Forest Classifier model trained.")
    except Exception as e:
        logger.error(f"An error occurred during model training: {e}")
else:
    logger.warning("Training data is empty. Skipping model training.")


## 10. Evaluation Metrics

# For a binary classification task, several metrics are crucial to evaluate model performance comprehensively, especially when dealing with potential class imbalance:

# *   **Accuracy**: The proportion of correctly classified instances out of the total instances.
# *   **Precision**: The proportion of true positive predictions among all positive predictions. It answers: "Of all instances predicted as major disasters, how many actually were major disasters?"
# *   **Recall (Sensitivity)**: The proportion of true positive predictions among all actual positive instances. It answers: "Of all actual major disasters, how many did we correctly identify?"
# *   **F1-Score**: The harmonic mean of Precision and Recall. It provides a single score that balances both precision and recall, useful for imbalanced datasets.
# *   **ROC AUC Score (Receiver Operating Characteristic - Area Under the Curve)**: Measures the ability of the classifier to distinguish between classes. An AUC of 1.0 means perfect separation, while 0.5 means no better than random guessing. It's particularly useful for imbalanced datasets as it considers all possible classification thresholds.


if not X_test.empty:
    logger.info("Evaluating model performance using various metrics.")

    models = {'Logistic Regression': pipeline_log_reg, 'Random Forest Classifier': pipeline_rf_clf}
    results = {}

    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class (1)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)

            results[name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC AUC': roc_auc
            }
            logger.info(f"--- {name} Evaluation ---")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            logger.info(f"ROC AUC: {roc_auc:.4f}")
        except Exception as e:
            logger.error(f"Error evaluating {name} model: {e}")

    # Plotting ROC curves
    try:
        fig = go.Figure()
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

        for name, model in models.items():
            if name in results: # Only plot if evaluation was successful
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={results[name]["ROC AUC"]:.2f})', mode='lines'))

        fig.update_layout(
            title_text='ROC Curve Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            yaxis=dict(scaleanchor="x", scaleratio=1),
            xaxis=dict(constrain='domain')
        )
        fig.show()
    except Exception as e:
        logger.error(f"Error plotting ROC curves: {e}")

    logger.info("Model evaluation completed.")
else:
    logger.warning("Test data is empty. Skipping model evaluation.")



## 12. Residuals and How to Visualize Them (for Classification)

### Residuals Explanation (Adapted for Classification)

# Traditionally, **residuals** are the differences between observed values and the values predicted by a regression model (`residual = actual - predicted`). They are used to check the assumptions of linear regression and assess model fit.

# For classification tasks, the concept of "residuals" isn't directly applicable in the same numerical sense, as our `y_pred` is a class label (0 or 1) rather than a continuous value. Instead, we analyze **prediction errors** or **misclassifications**.

# ### How to Visualize Prediction Errors in Classification

# For classification, the primary tool to visualize and understand prediction errors is the **Confusion Matrix**.

# A **Confusion Matrix** is a table that is used to describe the performance of a classification model on a set of test data for which the true values are known. It allows visualization of the performance of an algorithm.
# It has four key components for binary classification:
# *   **True Positives (TP)**: Correctly predicted positive cases (Actual 1, Predicted 1).
# *   **True Negatives (TN)**: Correctly predicted negative cases (Actual 0, Predicted 0).
# *   **False Positives (FP)**: Incorrectly predicted positive cases (Actual 0, Predicted 1 - Type I error).
# *   **False Negatives (FN)**: Incorrectly predicted negative cases (Actual 1, Predicted 0 - Type II error).

# ### Visualizing the Confusion Matrix

if not X_test.empty:
    logger.info("Visualizing prediction errors using Confusion Matrix.")

    # Use the best performing model (e.g., Random Forest after initial evaluation)
    # For now, let's use Random Forest Classifier as it's generally more robust.
    best_model_name = 'Random Forest Classifier'
    if best_model_name in models and models[best_model_name]:
        try:
            y_pred_best = models[best_model_name].predict(X_test)
            cm = confusion_matrix(y_test, y_pred_best)

            # Create labels for the matrix
            labels = ['Minor Disaster (0)', 'Major Disaster (1)']

            # Plotting the Confusion Matrix using Plotly
            fig = px.imshow(cm,
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=labels,
                            y=labels,
                            color_continuous_scale='Blues',
                            title=f'Confusion Matrix for {best_model_name}')

            fig.update_xaxes(side="bottom")
            fig.update_layout(
                autosize=False,
                width=500,
                height=500,
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                margin=dict(t=50, b=50, l=50, r=50) # Adjust margins
            )
            # Add text annotations
            for i in range(len(labels)):
                for j in range(len(labels)):
                    fig.add_annotation(dict(x=j, y=i, text=str(cm[i, j]), showarrow=False, font=dict(color="black", size=14)))
            
            fig.show()
            logger.info(f"Confusion Matrix for {best_model_name} displayed.")
        except Exception as e:
            logger.error(f"Error generating Confusion Matrix for {best_model_name}: {e}")
    else:
        logger.warning(f"Model '{best_model_name}' not available for Confusion Matrix visualization.")
else:
    logger.warning("Test data is empty. Skipping Confusion Matrix visualization.")



## Overfitting or Underfitting

# Understanding overfitting and underfitting is critical for building robust machine learning models.

# ### Overfitting

# **Definition**: Overfitting occurs when a model learns the training data too well, including its noise and random fluctuations, to the extent that it performs poorly on unseen or new data. The model has essentially memorized the training data rather than learned to generalize from it.

# **Characteristics**:
# *   **High performance on training data** (e.g., high accuracy, low loss).
# *   **Significantly lower performance on test/validation data**.
# *   The model is usually overly complex for the underlying patterns in the data.

# **Causes**:
# *   Too complex a model for the amount of data.
# *   Too many features (some of which might be noise).
# *   Insufficient training data.
# *   Lack of regularization.

# **How to Fix It**:
# 1.  **More Training Data**: Increasing the amount of diverse training data can help the model generalize better.
# 2.  **Feature Selection/Engineering**: Remove irrelevant or redundant features, or combine features to create more meaningful ones.
# 3.  **Simpler Model**: Choose a less complex model (e.g., Logistic Regression instead of a deep neural network for a simple problem).
# 4.  **Regularization**: Add penalties to the loss function (L1 or L2 regularization) that discourage large weights, thereby reducing model complexity. (Logistic Regression and some other models have this built-in).
# 5.  **Cross-Validation**: Use techniques like k-fold cross-validation to get a more reliable estimate of model performance and detect overfitting early.
# 6.  **Early Stopping**: For iterative models, stop training when performance on a validation set starts to degrade.
# 7.  **Ensemble Methods**: Techniques like Random Forests naturally reduce overfitting by averaging predictions from multiple trees.
# 8.  **Dropout (for Neural Networks)**: Randomly ignore a subset of neurons during training to prevent co-adaptation.

# ### Underfitting

# **Definition**: Underfitting occurs when a model is too simple to capture the underlying patterns in the training data, resulting in poor performance on both training and test data. The model fails to learn the relationship between features and the target variable.

# **Characteristics**:
# *   **Poor performance on training data** (e.g., low accuracy, high loss).
# *   **Poor performance on test/validation data**.
# *   The model is usually too simple to capture the complexity of the data.

# **Causes**:
# *   Too simple a model (e.g., a linear model for highly non-linear data).
# *   Insufficient or irrelevant features.
# *   Too much regularization.
# *   Insufficient training time (for iterative models).

# **How to Fix It**:
# 1.  **More Complex Model**: Choose a more flexible or complex model (e.g., a Random Forest, Gradient Boosting, or a neural network for complex problems).
# 2.  **Feature Engineering**: Create new features that might better represent the underlying relationships (e.g., polynomial features, interaction terms).
# 3.  **Reduce Regularization**: If regularization is applied, reduce its strength.
# 4.  **Increase Training Time/Iterations**: For iterative models, allow more training epochs.

# **Diagnosing Overfitting/Underfitting in our models**:
# We can check for overfitting by comparing the performance metrics (e.g., accuracy, F1-score) on the training set versus the test set.
# *   If `Train Score >> Test Score`: Likely overfitting.
# *   If `Train Score ≈ Test Score` and both are low: Likely underfitting.
# *   If `Train Score ≈ Test Score` and both are high: Good fit.

# Our `class_weight='balanced'` setting helps prevent underfitting on the minority class and improve generalization.

## 14. Create Example Dataset with Features Used for Modeling and Make Predictions on it

if not X_train.empty and not X_test.empty:
    logger.info("Diagnosing overfitting/underfitting.")

    models = {'Logistic Regression': pipeline_log_reg, 'Random Forest Classifier': pipeline_rf_clf}

    print("\n--- Overfitting/Underfitting Diagnosis ---")
    for name, model in models.items():
        try:
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            train_f1 = f1_score(y_train, y_train_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            print(f"\nModel: {name}")
            print(f"  Training Accuracy: {train_accuracy:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Training F1-Score: {train_f1:.4f}")
            print(f"  Test F1-Score: {test_f1:.4f}")

            if train_accuracy > test_accuracy + 0.05: # A heuristic for significant difference
                logger.warning(f"{name} might be overfitting. Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
                print("  --> Potential Overfitting: Training performance is significantly higher than test performance.")
            elif train_accuracy < 0.6 and test_accuracy < 0.6: # A heuristic for low performance
                logger.warning(f"{name} might be underfitting. Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
                print("  --> Potential Underfitting: Both training and test performance are low.")
            else:
                logger.info(f"{name} seems to have a good fit. Train Acc: {train_accuracy:.4f}, Test Acc: {test_accuracy:.4f}")
                print("  --> Good Fit: Training and test performance are comparable and satisfactory.")
        except Exception as e:
            logger.error(f"Error during overfitting/underfitting diagnosis for {name}: {e}")
else:
    logger.warning("Training or test data is empty. Skipping overfitting/underfitting diagnosis.")


if not X_train.empty and pipeline_rf_clf: # Assuming Random Forest is a strong candidate
    logger.info("Creating example dataset for prediction.")

    # Schema for the example data, matching the selected_features
    example_schema = {
        'disaster_type': 'Wildfire',
        'location': 'Australia', # Example of a new location not in initial sample
        'latitude': -30.0,
        'longitude': 135.0,
        'severity_level': 9,
        'affected_population': 50000,
        'estimated_economic_loss_usd': 15000000.0,
        'response_time_hours': 10.5,
        'aid_provided': 'Yes',
        'infrastructure_damage_index': 0.98,
        'year': 2026,
        'month': 1,
        'day': 15,
        'day_of_week': 2 # Wednesday
    }

    # Create a DataFrame from the example
    example_df = pd.DataFrame([example_schema])
    logger.info(f"Example data created: {example_df.head()}")

    try:
        # Make a copy of the preprocessor from the trained pipeline to apply to new data
        # The pipeline automatically handles preprocessing
        
        # Ensure the example_df has all columns that were in X_train
        # Handle cases where example_df might be missing columns that were in X_train but not in example_schema
        # This is particularly important for 'remainder'='passthrough' columns
        
        # Get the columns from X_train
        train_cols = X_train.columns.tolist()
        
        # Add missing columns to example_df, filling with default values (e.g., 0 for one-hot encoded features not present)
        for col in train_cols:
            if col not in example_df.columns:
                # For categorical features not present in example_schema, their OHE columns will be created by preprocessor
                # For numerical features, this might not happen unless they are 'remainder'
                # If a 'remainder' numerical feature is missing, it's safer to add it
                if X_train[col].dtype in ['int64', 'float64']:
                    example_df[col] = 0 # Default numerical value
                else:
                    example_df[col] = X_train[col].mode()[0] # Default categorical value (will be handled by OHE)

        # Reorder columns to match X_train, important for some transformers if 'remainder' is 'passthrough'
        example_df = example_df[train_cols]
        
        # Make prediction using the trained pipeline
        prediction = pipeline_rf_clf.predict(example_df)
        prediction_proba = pipeline_rf_clf.predict_proba(example_df)[:, 1]

        print("\n--- Prediction on Example Data ---")
        print(f"Example Data Features:\n{example_df.to_string(index=False)}")
        print(f"Predicted `is_major_disaster`: {prediction[0]}")
        print(f"Probability of being a major disaster: {prediction_proba[0]:.4f}")

        logger.info(f"Prediction for example data completed. Predicted: {prediction[0]}, Probability: {prediction_proba[0]:.4f}")

    except Exception as e:
        logger.error(f"Error making prediction on example data: {e}")
else:
    logger.warning("Model or training data not available. Skipping example prediction.")


## 15. Hyperparameter Tuning on Sample or Small Dataset

# Hyperparameter tuning is crucial for optimizing model performance. We will use `GridSearchCV` to systematically search for the best combination of hyperparameters for our Random Forest Classifier. To keep computation time reasonable for this notebook, we'll consider a limited grid of parameters.

if not X_train.empty and pipeline_rf_clf:
    logger.info("Starting hyperparameter tuning for Random Forest Classifier using GridSearchCV.")

    # Define a smaller parameter grid for demonstration purposes or a subset of the data
    # For a full dataset, this grid could be expanded, or RandomizedSearchCV could be used.
    param_grid_rf = {
        'classifier__n_estimators': [50, 100, 200], # Number of trees in the forest
        'classifier__max_depth': [None, 10, 20], # Maximum depth of the tree
        'classifier__min_samples_split': [2, 5], # Minimum number of samples required to split an internal node
        'classifier__min_samples_leaf': [1, 2] # Minimum number of samples required to be at a leaf node
    }
    
    # We will use the full X_train and y_train, but if it was too large, we would sample a subset.
    # For demonstration, let's use a smaller subset if the full_df size is large, otherwise use all.
    # Here, we will use the full X_train to ensure robustness.

    try:
        # Create GridSearchCV object
        grid_search_rf = GridSearchCV(pipeline_rf_clf, param_grid_rf, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        # Using F1-score as it's good for imbalanced classification

        # Fit GridSearchCV to the training data
        grid_search_rf.fit(X_train, y_train)

        logger.info("Hyperparameter tuning completed.")
        print("\n--- Hyperparameter Tuning Results (Random Forest Classifier) ---")
        print(f"Best parameters found: {grid_search_rf.best_params_}")
        print(f"Best F1-score on validation set: {grid_search_rf.best_score_:.4f}")

        # Get the best model
        best_rf_model = grid_search_rf.best_estimator_
        logger.info("Best Random Forest model obtained from GridSearchCV.")

    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}")
        best_rf_model = None
else:
    logger.warning("Training data or Random Forest pipeline not available. Skipping hyperparameter tuning.")
    best_rf_model = None

# **Explanation of Hyperparameter Tuning and Chosen Hyperparameters:**

# Hyperparameter tuning involves finding the optimal settings for the parameters of the learning algorithm itself (not the model parameters learned during training). These 'hyperparameters' are set before the learning process begins.

# *   **GridSearchCV**: We used `GridSearchCV` for tuning. It performs an exhaustive search over all parameter combinations specified in `param_grid_rf`. For each combination, it trains the model with k-fold cross-validation (`cv=3` means 3 folds) on the training data and evaluates its performance using the specified `scoring` metric (`f1` for F1-score, which is good for our potentially imbalanced dataset). The combination that yields the best average score across the folds is selected as the optimal set of hyperparameters.
# *   **Chosen Hyperparameters for Random Forest**:
#     *   `classifier__n_estimators`: This is the number of trees in the forest. More trees generally lead to better performance but increase computation time. We tested `[50, 100, 200]`.
#     *   `classifier__max_depth`: The maximum depth of each tree. Limiting depth helps control overfitting. `None` means nodes are expanded until all leaves are pure or contain `min_samples_split` samples. We tested `[None, 10, 20]`.
#     *   `classifier__min_samples_split`: The minimum number of samples required to split an internal node. Increasing this can prevent a tree from learning too specific patterns from the training data. We tested `[2, 5]`.
#     *   `classifier__min_samples_leaf`: The minimum number of samples required to be at a leaf node. Similar to `min_samples_split`, it helps in regularization. We tested `[1, 2]`.

# The goal of this tuning is to find a set of hyperparameters that allow the Random Forest model to generalize well to unseen data, balancing bias and variance. The `best_params_` from `GridSearchCV` will provide the optimal combination found within our specified grid.

##  Final Model Selection

# Based on the evaluation metrics from both initial models and the hyperparameter tuning results, we will select the final model.

# Our primary goal is to accurately predict `is_major_disaster`. Considering the potential costs of misclassifying a major disaster (False Negative) or wrongly flagging a minor one as major (False Positive), we aim for a balanced performance, often prioritizing metrics like **F1-Score** and **ROC AUC** as they are robust to class imbalance.

# **Comparison of Models:**
# *   **Logistic Regression**: Often serves as a good baseline. Its simplicity makes it interpretable, but it might not capture complex non-linear relationships.
# *   **Random Forest Classifier (Initial)**: Generally performs better than linear models by capturing complex interactions.
# *   **Random Forest Classifier (Tuned)**: After hyperparameter tuning, this model is expected to have optimized performance, potentially achieving a better balance of precision and recall and a higher F1-score and ROC AUC.

# We will select the model that demonstrated the best overall performance, particularly in terms of F1-score and ROC AUC on the test set. Given its ensemble nature and the tuning process, the **Tuned Random Forest Classifier** is expected to be our final choice.

if best_rf_model and 'Random Forest Classifier' in results:
    logger.info("Comparing tuned and untuned Random Forest models for final selection.")
    
    # Evaluate the untuned Random Forest for direct comparison
    untuned_rf_results = results['Random Forest Classifier']
    
    # Evaluate the tuned Random Forest
    y_pred_tuned = best_rf_model.predict(X_test)
    y_proba_tuned = best_rf_model.predict_proba(X_test)[:, 1]
    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
    tuned_precision = precision_score(y_test, y_pred_tuned)
    tuned_recall = recall_score(y_test, y_pred_tuned)
    tuned_f1 = f1_score(y_test, y_pred_tuned)
    tuned_roc_auc = roc_auc_score(y_test, y_proba_tuned)

    print("\n--- Final Model Selection ---")
    print("\nUntuned Random Forest Classifier:")
    print(f"  Accuracy: {untuned_rf_results['Accuracy']:.4f}")
    print(f"  Precision: {untuned_rf_results['Precision']:.4f}")
    print(f"  Recall: {untuned_rf_results['Recall']:.4f}")
    print(f"  F1-Score: {untuned_rf_results['F1-Score']:.4f}")
    print(f"  ROC AUC: {untuned_rf_results['ROC AUC']:.4f}")

    print("\nTuned Random Forest Classifier (Best from GridSearchCV):")
    print(f"  Accuracy: {tuned_accuracy:.4f}")
    print(f"  Precision: {tuned_precision:.4f}")
    print(f"  Recall: {tuned_recall:.4f}")
    print(f"  F1-Score: {tuned_f1:.4f}")
    print(f"  ROC AUC: {tuned_roc_auc:.4f}")

    if tuned_f1 >= untuned_rf_results['F1-Score'] and tuned_roc_auc >= untuned_rf_results['ROC AUC']:
        final_model = best_rf_model
        final_model_name = 'Tuned Random Forest Classifier'
        logger.info(f"Final model selected: {final_model_name} due to superior performance after tuning.")
    else:
        # Fallback if tuning didn't improve or made it worse (unlikely but possible with small grids)
        final_model = pipeline_rf_clf
        final_model_name = 'Untuned Random Forest Classifier'
        logger.warning(f"Tuned Random Forest did not show significant improvement. Selecting {final_model_name}.")

    print(f"\nConclusion: The **{final_model_name}** is selected as the final model.")
else:
    logger.warning("No best Random Forest model available for final selection. Please re-run tuning.")
    final_model = None


##  Save the Final Model

# It is crucial to save the trained model for future use without needing to retrain it. We will use the `pickle` library to serialize the final selected model and store it in a dedicated `artifacts` folder. This folder will also store the preprocessor for consistent data transformation during inference.
# Create 'artifacts' directory
artifacts_dir = 'artifacts'
if not os.path.exists(artifacts_dir):
    os.makedirs(artifacts_dir)
    logger.info(f"Created artifacts directory: {artifacts_dir}")

if final_model:
    try:
        model_filepath = os.path.join(artifacts_dir, 'final_disaster_classifier.pkl')
        preprocessor_filepath = os.path.join(artifacts_dir, 'preprocessor.pkl')

        # Save the entire pipeline (which includes the preprocessor and the classifier)
        # If we just had the classifier, we'd save the preprocessor separately
        with open(model_filepath, 'wb') as f:
            pickle.dump(final_model, f)
        logger.info(f"Final model (pipeline) saved to {model_filepath}")

        # Also save the preprocessor separately for modularity, although it's part of the pipeline.
        # This can be useful if one needs just the preprocessor for other tasks.
        # For our pipeline structure, it's already encapsulated, but we'll extract and save it explicitly
        # if there were scenarios where only the preprocessor was needed.
        # For simplicity, saving the entire pipeline is sufficient here.

        # If we wanted to save preprocessor separately:
        # with open(preprocessor_filepath, 'wb') as f:
        #     pickle.dump(preprocessor, f)
        # logger.info(f"Preprocessor saved to {preprocessor_filepath}")

        print(f"\nFinal model saved successfully to: {model_filepath}")
        # print(f"Preprocessor saved successfully to: {preprocessor_filepath}")

    except Exception as e:
        logger.error(f"Error saving the final model: {e}")
else:
    logger.warning("No final model available to save.")
