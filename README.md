# oibsip_taskno-5
Advertising Dataset Analysis

Overview

This project focuses on analyzing an advertising dataset to explore the relationships between different advertising channels (TV, Radio, and Newspaper) and their impact on sales. The dataset is loaded from Advertising.csv and consists of 200 records with the following features:

TV: Advertising budget allocated to TV commercials.

Radio: Advertising budget allocated to radio ads.

Newspaper: Advertising budget allocated to newspaper ads.

Sales: Sales figures resulting from the advertising campaigns.

Objectives

Preprocess the data by scaling features for better model performance.

Perform exploratory data analysis (EDA) to uncover trends and patterns.

Build and evaluate predictive models to understand the relationship between advertising channels and sales.

Steps

1. Data Loading

The dataset is read into a Pandas DataFrame using:

adv = pd.read_csv('Advertising.csv')

2. Data Preprocessing

Feature Scaling: StandardScaler is used to normalize the features (TV, Radio, Newspaper).

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

3. Exploratory Data Analysis

Visualizations and descriptive statistics are employed to explore relationships:

Scatter plots for sales vs. each advertising channel.

Heatmap to visualize correlations among features.

Distribution plots to assess data normality.

4. Model Building

Models Implemented:

Linear Regression

Ridge Regression

Lasso Regression

Steps:

Split the dataset into training and testing sets:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

Fit and evaluate models using metrics like Mean Squared Error (MSE) and R-squared.

5. Model Evaluation

Metrics used for evaluation:

Mean Squared Error (MSE)

R-squared (R²)

Cross-validation to validate the robustness of models.

6. Visualization

Plots of actual vs. predicted values.

Residual analysis to check model assumptions.

Libraries Used

Pandas: For data manipulation and analysis.

NumPy: For numerical computations.

Matplotlib & Seaborn: For visualizations.

Scikit-learn: For preprocessing, model building, and evaluation.

Results and Insights

Identified the strongest predictors of sales among the advertising channels.

Visualized the impact of advertising budgets on sales performance.

Demonstrated the effectiveness of regression models in capturing relationships in the data.

How to Run

Ensure Python and required libraries are installed.

Place Advertising.csv in the working directory.

Run the script to preprocess the data, perform analysis, and visualize results.

Future Scope

Explore non-linear models for better predictions.

Conduct feature engineering to create new variables.

Implement advanced techniques like cross-validation and hyperparameter tuning.
