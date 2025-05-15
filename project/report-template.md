# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Haider Ali

# Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
The output needed to have only two columns, the datetime column and the count column. Similarly, when saving the file, the index should be set to `False`. 

### What was the top ranked model that performed?
## AutoGluon Leaderboard – Initial Training Run

![Validation-score bar chart for all trained models](img/predictor_leaderboard.png)
![Validation-score bar chart for all trained models_with engineered_features](img/predictor_leaderboard_feature_engineered.png)

The shortest (highest) bar belongs to **`WeightedEnsemble_L3`**, making it the top-ranked model on the validation set. This is true for both original and engineered features.


## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
## Exploratory Data Analysis (EDA) & Feature Engineering

### 1 · Key findings from the EDA  
| Evidence | Insight |
|----------|---------|
| ![Histogram of `count`](img/hist_count.png) | **Target (`count`) is right-skewed** – many hours with small rentals, long tail of busy periods. |
| ![Scatter-matrix](img/scatter_matrix.png) | **Positive trend:** `temp` / `atemp` ⟶ higher bike demand.<br>**Weak negatives:** `windspeed`, mild for `humidity`. |
| ![Correlation heat-map](img/corr_matrix.png) | `temp`, `atemp`, `hour_sin/cos`, `rush_hour` show strongest (linear) association with `count`. Almost no multicollinearity issues except `temp` ≃ `atemp`. |
| ![Histogram of `workingday`](img/hist_workingday.png) | Binary flags (`workingday`, `holiday`) are **highly imbalanced** → treat as categorical. |

### 2 · Additional features created  
| New column | Type | Why it helps |
|------------|------|--------------|
| `hour`, `dayofweek`, `month`, `year` | `category` | Exposes daily, weekly, monthly, yearly seasonality. |
| `is_weekend` | int (0/1) | Single flag for Sat/Sun demand patterns. |
| `rush_hour`  | int (0/1) | Captures 07-09 h & 16-18 h commuting spikes. |
| `hour_sin`, `hour_cos` | float | Cyclical encoding lets linear / NN models “see” that hour = 23 is next to 0. |
| `season`, `weather` | converted to `category` | Prevents AutoGluon treating them as ordinal magnitudes. |

### 3 · Implementation snippet
```python
for df in (train_data, test):
    # decomposed datetime
    df['hour']       = df['datetime'].dt.hour.astype('uint8')
    df['dayofweek']  = df['datetime'].dt.dayof_week.astype('uint8')
    df['month']      = df['datetime'].dt.month.astype('uint8')
    df['year']       = df['datetime'].dt.year.astype('uint16')

    # weekend / rush-hour flags
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype('int8')
    df['rush_hour']  = (
        df['hour'].between(7, 9) | df['hour'].between(16, 18)
    ).astype('int8')

    # cyclical hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # ensure categorical dtypes
    for col in ['season', 'weather', 'hour', 'dayofweek', 'month', 'year']:
        df[col] = df[col].astype('category')
```

### How much better did your model preform after adding additional features and why do you think that is?
TODO: Add your explanation

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
TODO: Add your explanation

### If you were given more time with this dataset, where do you think you would spend more time?
TODO: Add your explanation

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--|--|--|--|
|initial|?|?|?|?|
|add_features|?|?|?|?|
|hpo|?|?|?|?|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](img/model_test_score.png)

## Summary
TODO: Add your explanation
