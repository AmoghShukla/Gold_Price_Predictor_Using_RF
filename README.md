# Gold Price Prediction using Random Forest Regressor

## Project Overview
This project focuses on predicting gold prices using a Random Forest Regressor machine learning model. The model is trained on historical data including factors like S&P 500 index (SPX), Crude Oil price (USO), Silver price (SLV), and the Euro to US Dollar exchange rate (EUR/USD) to forecast the price of Gold (GLD).

## Data Source
The dataset used for this project is `gld_price_data.csv`, which contains historical daily prices for various financial instruments:
- **Date**: The date of the observation.
- **SPX**: S&P 500 Index price.
- **GLD**: Gold price (Target variable).
- **USO**: United States Oil Fund price.
- **SLV**: Silver price.
- **EUR/USD**: Euro to US Dollar exchange rate.

## Technologies Used
- Python 3
- Pandas (for data manipulation and analysis)
- NumPy (for numerical operations)
- Matplotlib (for data visualization)
- Seaborn (for enhanced data visualization)
- Scikit-learn (for machine learning model implementation and evaluation)

## Exploratory Data Analysis (EDA)
- **Data Loading and Inspection**: Loaded the `gld_price_data.csv` into a Pandas DataFrame, checked its shape, info, and statistical summary.
- **Missing Values**: Verified that there are no missing values in the dataset.
- **Correlation Analysis**: Explored the correlation between different features and the target variable (GLD) using a heatmap. Noted a strong positive correlation between GLD and SLV, and negative correlations with USO and EUR/USD.
- **Target Variable Distribution**: Visualized the distribution of GLD prices.

## Model Details
### Random Forest Regressor
- **Model Type**: Ensemble Learning (Random Forest).
- **Algorithm**: The Random Forest Regressor is used for its robustness and ability to handle non-linear relationships and interactions between features. It builds multiple decision trees during training and outputs the mean prediction of the individual trees.
- **Hyperparameters**: `n_estimators=100` (number of trees in the forest).

## Model Training and Evaluation
- **Feature Selection**: The 'Date' column and 'GLD' (target) were dropped from the feature set (`X`).
- **Data Splitting**: The dataset was split into training (80%) and testing (20%) sets using `train_test_split` with a `random_state=2` for reproducibility.
- **Training**: The `RandomForestRegressor` model was trained on the `X_train` and `Y_train` data.
- **Prediction**: Predictions were made on the `X_test` data.
- **Evaluation Metric**: The R-squared error (`r2_score`) was used to evaluate the model's performance. A high R-squared value indicates a good fit of the model to the data.
  - **R Squared Error**: `0.9893` (indicating a strong performance).
- **Visualization**: A plot comparing the actual gold prices (`Y_test`) against the predicted gold prices (`test_data_prediction`) was generated to visually assess the model's accuracy.

## How to Run the Project
1.  **Clone the Repository**: 
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Ensure you have the dataset**: Place the `gld_price_data.csv` file in the root directory of the project, or update the path in the code.
3.  **Install Dependencies**: Make sure you have all the necessary Python libraries installed. You can install them using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
4.  **Open in Google Colab**: Upload the `.ipynb` notebook to Google Colab.
5.  **Run Cells**: Execute the cells sequentially from top to bottom to perform data loading, preprocessing, model training, and evaluation.
