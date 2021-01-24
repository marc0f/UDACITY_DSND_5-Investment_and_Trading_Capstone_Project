Machine Learning Engineer Nanodegree

## Capstone Project
Marco Fagiani  
January 20st, 2020

## I. Definition
### Project Overview
Investment firms, hedge funds and even individuals have been using financial models to better understand market behavior and make profitable investments and trades. A wealth of  information is available in the form of historical stock prices and company performance data, suitable for machine learning algorithms to process.

The goal of this project is to build a stock price predictor that takes daily trading data over a certain date range as input, and outputs projected estimates for given query dates. The selected model should guarantee a percentage error of the prediction (in the training phase) lower than the 5%. Note that the inputs will contain multiple metrics, such as opening price (Open), highest price the stock traded at (High), how many stocks were traded (Volume) and closing price adjusted for stock splits and dividends (Adjusted Close); this system only needs to predict the Adjusted Close price.

In summary the project is divided in three main parts: 

- data retrieving;
- data modeling;
-  data visualization. 

The data retrieving is taking care of collect data from trusted source, clean them, combine with previous data and make them available to consumption of the others parts. 

The data modeling's main task is to perform analysis and to produce the model that will be used to predict the market returns for different time spans. This part is mainly developed to be run on demand by the user to update the models, using the same hyper-parameters as well as to provide all the methods to investigate and explore other solutions. 

Finally, the data visualization part provides a dashboard for the user to display historical data as well as the predictions achieved using up-to-date data. 



### Problem Statement

In order to produce any model the first step required is the collection of the data that will be used. Once a trusted source is selected the data must be analyzed to understand any issues within the data itself. Common problems are the presence of gaps in the data, fields reporting the same constant value, or fields with missing data. Moreover, begin the main goal to provide up-to-date prediction, the data must be updated with newest one on daily basis, avoiding the download of the whole required range every time.  

Aiming to produce a prediction of the future returns for different time spans, one thing to understand is the amount o data to be used, especially to train the model. Few data will produce a model with both poor performance and not representing the general phenomena, but only a specific path in the data. On the other hand, using the maximum amount of data available, without a proper split in train and test sets, could lead to produce poor performance (example, due to over-fitting the model on the training data), lack of specificity of the model as well as really long training times. Moreover, the extraction of meaningful features is another important step. Select the right feature, even just dropping of the original one, can help to reduce the number of used data (thus a faster training phase), as well as pointing to the development of features that maximize the knowledge.

Finally, the results must be presented to the final user, in yet comprehensive but understandable manner.



### Metrics
In this project have been adopted 3 metrics: mean squared error (MSE), mean absolute error (MAE) and mean absolute percentage error. Defined as:


$$
MSE = \frac{1}{n} \sum^{n}_{i=1} (y_i - \hat{y}_i)^2\\
MAE = \frac{\sum^{n}_{i=1} |y_i - \hat{y}_i|}{n}\\
MAPE = \frac{1}{n}\sum^{n}_{i=1}|\frac{y_i - \hat{y}_i}{y_i}|
$$
where, given a time-series $Y$ of $n$ elements, $y_i$ is the i-th actual value and $\hat{y}_i$ is the corresponding predicted value.

The MSE and the MAE are metrics commonly used to measure the regression performance, and thus have been kept to have a wide pool of metrics to perform the models evaluation. The MAPE has been selected because the target of the project is to select a model capable to achieve a prediction percentage error lower than 5%.




## II. Analysis
### Data Exploration

The project has been performed over the following set of stocks:

- Best Value Stocks 
  - NRG Energy Inc. ([NRG](https://www.investopedia.com/markets/quote?tvwidgetsymbol=NRG))
  - Vornado Realty Trust ([VNO](https://www.investopedia.com/markets/quote?tvwidgetsymbol=VNO))
  - MGM Resorts International ([MGM](https://www.investopedia.com/markets/quote?tvwidgetsymbol=MGM))
- Fastest Growing Stocks
  - AmerisourceBergen Corp. ([ABC](https://www.investopedia.com/markets/quote?tvwidgetsymbol=ABC))
  - MGM Resorts International (MGM)
  - Align Technology Inc. ([ALGN](https://www.investopedia.com/markets/quote?tvwidgetsymbol=ALGN))
- Stocks with the Most Momentum
  - DexCom Inc. ([DXCM](https://www.investopedia.com/markets/quote?tvwidgetsymbol=DXCM))
  - NVIDIA Corp. ([NVDA](https://www.investopedia.com/markets/quote?tvwidgetsymbol=NVDA))
  - Regeneron Pharmaceuticals Inc. ([REGN](https://www.investopedia.com/markets/quote?tvwidgetsymbol=REGN))
  - S&P 500

Considered the Top Stock till mid-2020 ([Investopedia](https://www.investopedia.com/top-stocks-4581225)).

The stock data are retrieved using the Yahoo! Finance [API](https://pypi.org/project/yfinance/) and leveraging over the historical end-point to obtain the following data:

```
             Open   High    Low  Close  Adj Close   Volume  Dividends  Stock Splits
Date                                                                               
2009-12-31  23.90  24.00  23.61  23.61      20.67  1024900        0.0             0
2010-01-04  23.78  24.13  23.70  23.87      20.89  1683700        0.0             0
2010-01-05  23.96  24.28  23.81  24.24      21.22  3473400        0.0             0
2010-01-06  24.25  24.79  24.11  24.77      21.68  2719300        0.0             0
2010-01-07  24.79  25.24  24.73  24.91      21.80  3200800        0.0             0
```

Basically, the query is performed from the current UTC datetime back till the required range is obtained. The raw data is then stored in a dedicated CSV, for each symbol. In further queries are performed, if the CSV already exists, at first the data are read from it and only for the missing date range a new query toward the provider is performed. In case of a new query, the new data are integrate with the CSV one, and re-stored into the same CSV.

#### Data cleaning

As said, the raw data are stored in to CSV. A cleaning procedure is performed only on the data before the utilization in the training, test, prediction phases. The only abnormality in the data is the presence of two columns, Dividends and Stock Splits, composed of only 0 values. Therefore, these columns are dropped in a dedicated clean function that detects columns with all equal values.

```python
def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()


def clean_data(data):
    """ remove nans or constant values columns, or drop equals columns"""

    for col_name in list(data.columns):
        if is_unique(data[col_name]):
            logger.info(f"Column {col_name} has unique values..removed.")
            data.drop(columns=col_name, inplace=True)
```



### Exploratory Visualization

Here below is reported the last 6 months of prices (open, close, high, low, adj) and volume for the symbol NRG. 

![NRG Prices](imgs/visual_NRG_1.png)



![NRG Volume](imgs/visual_NRG_2.png)

#### Dashboard

The developed dashboard displays the expected returns (predictions) a head of 1, 7, 14, and 28 days for each symbol. Also a visualization of the past data and the predictions is provided in graphs.  

![Dashboard](imgs/dashboard_1.png)



### Algorithms and Techniques

#### Data Normalization

Once the data is clean (see below), the next step is to normalize the data. The normalization process can affects the data range (i.e., simple compression or expansion to get data in the [0, 1] range) or alters the data statistics, mean and variance.

##### Min-Max normalization (Normalizer)

Min-max normalization is one of the most common ways to normalize data. For every feature, the minimum value of that feature gets transformed into a 0, the maximum value gets transformed into a 1, and every other value gets transformed into a decimal between 0 and 1. 


$$
\hat{x} = \frac{x - x_{min}}{(x_{max} - x_{min})}
$$


##### Mean and Variance Normalization (StandardScaler)

This normalization ensures that the input is transformed to obtain an mean of the output data approximately zero, while the standard deviation (as well as the variance) is in a range close to unity. Given with $\mu, \sigma$ the mean and the variance of the input data, the output data is:
$$
\hat{x} = \frac{x- \mu}{\sigma}
$$


#### Support Vector Regression

The Support Vector Regression approach (SVR or SVM regression) is derived from the Support Vector Machine technique. The SVMs are binary classifiers that discriminate whether an input vector x belongs to class +1 or to class −1 based on the following discriminant function:
$$
f(x) = \sum^{N}_{i=1} \alpha_i \cdot t_i \cdot K(\mathbf{x}, \mathbf{x_i}) + d
$$
where $t_i \in {+1, -1}, $ $\alpha_i > 0$ and $\sum^{N}_{i=1} \alpha_i · t_i = 0$. The terms $\mathbf{x}_i$ are the “support vectors” and $d$ is a bias term that together with the $\alpha_i$ is determined during the training process of the SVM. The input vector $\mathbf{x}$ is classified as +1 if f(x) ≥ 0 and −1 if $f(x) < 0$. The kernel function $K(·, ·)$ can assume different forms.

Differently from the SVM, the solution of the optimization problem of a linear model (in the feature space) for the SVR is given by:
$$
f(x) = \sum^{N}_{i=1} (\alpha_i - \alpha_i^{*}) \cdot K(\mathbf{x}, \mathbf{x_i}) + d
$$
where, $\sum^{N}_{i=1} (\alpha_i - \alpha_i^{*}) = 0$ with $\alpha_i, \alpha_i^{*} \in [0, C]$.

The kernel types to be evaluated are the linear and the radial basis function (RBF). A linear kernel expresses a normal dot product of any two given observations, thus $K(x, xi) = (x \times xi)$. Instead, the RBF can map an input space in infinite dimensional space, where $K(u, v) = \exp^{−\gamma (x − x_i^2 )}$. The $\gamma$ and $C$ parameters, and the chosen kernel are crucial to obtain the best performance.  



#### Data Validation

The experiments has been divided in tow main groups: model validation and model finalization.

##### Model validation

In the model validation the data has been split using a standard 70-30 approach, thus 70% of the data has been associated to the train set, and the remaining 30% to the test set. This split has been adopted to perform the validation of the the data normalization to adopt, the length of the train period, and extra  features selection. Moreover, the regression technique parameters have not been optimized in this phase, and the default ones have been adopted: kernel set to RBF, $\gamma$ set to 'scale' (thus value set as $1 / (n\_features * X.var())$), C set to 1, $\epsilon$ set to 1.

In order to select the normalization to use, among none, Mean and Variance Normalization and Min-Max normalization, the data from 2016/01/01 to 2020/08/31 has been adopted.

Once the normalization has been selected, the same time range has been adopted to selected some extra feature to be introduced, extracted from the data, to achieve better results. The extra features selected are the difference for each input data, between the data at the time $t$ and the value of the previous 1, 7, 14, 28 days. Thus generating 4 set of features providing difference to different time lags.

 



steps:

- simple split 70%/30% with default parameters to evaluate:
  - train length
  - normalization type
  - feature selection and evaluation
- finally, CV 70/30 over set of parameters
- 





##### Model finalization

Cross-validation and grid search over the hyper-parameters



###################

In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing



```python
data = get_daily_historical(symbol, start_date, end_date)
data = clean_data(data)
samples, targets = prepare_data(data, delays=prediction_horizons, diffs=prediction_horizons)
```



```python


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    # ('scaler', Normalizer()),
    ('regres', MultiOutputRegressor(SVR(), n_jobs=num_cpus))
    # ('regres', SVR())
])
```



#####

In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:

- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_



### Model validation

Symbols: NRG





#### Normalizations

The adopted data range goes from 2016/01/01 to 2020/08/31, 56 months using as features the available from the raw data: Open, High, Low, Close, Volume and Dividends.

Different normalization techniques can be compared only over MAPE metric.

##### No Normalization Results

| Lags | MSE      | MAE   | MAPE   |
| :--- | -------- | ----- | ------ |
| 1    | 99.540   | 8.000 | 44.344 |
| 7    | 100.571. | 7.968 | 44.133 |
| 14   | 102.507  | 7.997 | 44.266 |
| 28   | 105.087  | 8.111 | 44.794 |



##### Mean and Variance Normalization Results

| Lags | MSE   | MAE   | MAPE  |
| :--- | ----- | ----- | ----- |
| 1    | 0.627 | 0.558 | 2.416 |
| 7    | 2.332 | 1.105 | 4.676 |
| 14   | 4.838 | 1.503 | 6.436 |
| 28   | 7.993 | 2.048 | 8.356 |



##### Min and Max Normalization Results

| Lags | MSE     | MAE   | MAPE   |
| :--- | ------- | ----- | ------ |
| 1    | 113.634 | 9.060 | 52.394 |
| 7    | 113.680 | 8.996 | 52.263 |
| 14   | 112.249 | 8.935 | 50.924 |
| 28   | 110.058 | 8.734 | 49.794 |



Selected normalization: Mean and Variance Normalization



#### Data ranges and Features Selection

##### 6 months

###### No extra features

| Lags | MSE   | MAE   | MAPE  |
| :--- | ----- | ----- | ----- |
| 1    | 0.738 | 0.692 | 2.176 |
| 7    | 2.166 | 1.085 | 3.423 |
| 14   | 3.751 | 1.636 | 5.019 |
| 28   | 6.922 | 1.821 | 5.292 |

###### Features lags 1, 7, 14, 28

| Lags | MSE   | MAE   | MAPE  |
| :--- | ----- | ----- | ----- |
| 1    | 0.536 | 0.576 | 1.827 |
| 7    | 1.855 | 1.136 | 3.669 |
| 14   | 1.606 | 1.006 | 3.103 |
| 28   | 7.767 | 1.734 | 4.688 |



##### 12 months

###### No extra features

| Lags | MSE   | MAE   | MAPE  |
| :--- | ----- | ----- | ----- |
| 1    | 1.507 | 0.803 | 2.622 |
| 7    | 5.874 | 1.542 | 5.183 |
| 14   | 9.936 | 2.307 | 7.804 |
| 28   | 7.116 | 1.882 | 5.830 |

###### Features lags 1, 7, 14, 28

| Lags | MSE   | MAE   | MAPE  |
| :--- | ----- | ----- | ----- |
| 1    | 2.632 | 1.127 | 3.731 |
| 7    | 2.019 | 1.122 | 3.483 |
| 14   | 1.677 | 1.064 | 3.295 |
| 28   | 2.838 | 1.350 | 4.171 |



##### 18 months

###### No extra features

| Lags | MSE    | MAE   | MAPE  |
| :--- | ------ | ----- | ----- |
| 1    | 2.399  | 0.899 | 2.896 |
| 7    | 9.236  | 1.930 | 6.596 |
| 14   | 13.532 | 2.281 | 7.953 |
| 28   | 16.852 | 3.020 | 9.669 |

###### Features lags 1, 7, 14, 28

| Lags | MSE   | MAE   | MAPE  |
| :--- | ----- | ----- | ----- |
| 1    | 0.632 | 0.648 | 1.905 |
| 7    | 3.944 | 1.347 | 4.179 |
| 14   | 4.363 | 1.425 | 4.294 |
| 28   | 6.176 | 2.011 | 5.951 |



##### 24 months

###### No extra features

| Lags | MSE   | MAE   | MAPE  |
| :--- | ----- | ----- | ----- |
| 1    | 0.427 | 0.540 | 1.551 |
| 7    | 1.210 | 0.864 | 2.497 |
| 14   | 2.419 | 1.181 | 3.440 |
| 28   | 5.593 | 1.648 | 4.635 |

###### Features lags 1, 7, 14, 28

| Lags | MSE    | MAE   | MAPE   |
| :--- | ------ | ----- | ------ |
| 1    | 0.939  | 0.693 | 2.0546 |
| 7    | 4.398  | 1.376 | 4.326  |
| 14   | 5.599  | 1.584 | 4.863  |
| 28   | 10.911 | 2.574 | 7.411  |



#### Model Finalization

The grid search has been performed over the following ranges for each hyper-parameter:

- C: [0.2, 0.4, 0.6, 0.8, 1. , 1.2, 1.4, 1.6, 1.8]
- $\epsilon$: [0.02, 0.04, 0.06, 0.08, 0.1 , 0.12, 0.14, 0.16, 0.18]
- kernel: ['linear', 'rbf']

Cross-validation: 5 folds over 6 months 

##### NRG

| Lags | MSE   | MAE   | MAPE  |
| :--- | ----- | ----- | ----- |
| 1    | 0.532 | 0.584 | 1.841 |
| 7    | 1.660 | 1.083 | 3.479 |
| 14   | 1.534 | 0.968 | 3.000 |
| 28   | 7.211 | 1.688 | 4.574 |

Selected hyper-parameters: 

- C: 1.8
- $\epsilon$: 0.18
- kernel: RBF

##### VNO

| Lags | MSE    | MAE   | MAPE  |
| :--- | ------ | ----- | ----- |
| 1    | 2.827  | 1.318 | 3.811 |
| 7    | 6.474  | 1.852 | 5.305 |
| 14   | 7.113  | 2.043 | 5.336 |
| 28   | 10.499 | 2.179 | 6.444 |

Selected hyper-parameters: 

- C: 1.8
- $\epsilon$: 0.18
- kernel: RBF

##### MGM

| Lags | MSE    | MAE   | MAPE  |
| :--- | ------ | ----- | ----- |
| 1    | 1.771  | 1.100 | 4.639 |
| 7    | 5.232  | 1.771 | 7.383 |
| 14   | 9.548  | 2.430 | 9.745 |
| 28   | 11.086 | 2.523 | 9.380 |

Selected hyper-parameters: 

- C: 0.2
- $\epsilon$: 0.12
- kernel: linear

##### ABC

| Lags | MSE   | MAE   | MAPE  |
| :--- | ----- | ----- | ----- |
| 1    | 5.359 | 1.738 | 1.710 |
| 7    | 6.926 | 2.121 | 2.115 |
| 14   | 6.491 | 1.762 | 1.736 |
| 28   | 8.738 | 2.392 | 2.341 |

Selected hyper-parameters: 

- C: 1.8
- $\epsilon$: 0.02
- kernel: RBF

##### ALGN

| Lags | MSE      | MAE    | MAPE  |
| :--- | -------- | ------ | ----- |
| 1    | 140.075  | 8.920  | 2.165 |
| 7    | 2229.026 | 32.542 | 7.476 |
| 14   | 3233.779 | 38.610 | 8.555 |
| 28   | 970.298  | 18.748 | 4.198 |

Selected hyper-parameters: 

- C: 1.8
- $\epsilon$: 0.02
- kernel: RBF

##### DXCM

| Lags | MSE     | MAE    | MAPE  |
| :--- | ------- | ------ | ----- |
| 1    | 321.172 | 14.799 | 4.208 |
| 7    | 906.46  | 24.372 | 6.862 |
| 14   | 981.718 | 26.242 | 6.977 |
| 28   | 825.052 | 9.877  | 5.272 |

Selected hyper-parameters: 

- C: 0.4
- $\epsilon$: 0.18
- kernel: linear

##### NVDA

| Lags | MSE     | MAE    | MAPE  |
| :--- | ------- | ------ | ----- |
| 1    | 518.028 | 18.032 | 3.464 |
| 7    | 425.651 | 15.256 | 2.908 |
| 14   | 272.346 | 12.261 | 2.307 |
| 28   | 221.789 | 10.938 | 2.042 |

Selected hyper-parameters: 

- C: 1.8
- $\epsilon$: 0.02
- kernel: RBF

##### REGN

| Lags | MSE      | MAE    | MAPE  |
| :--- | -------- | ------ | ----- |
| 1    | 173.962  | 9.363  | 1.657 |
| 7    | 534.684  | 18.562 | 3.365 |
| 14   | 1058.514 | 26.511 | 4.866 |
| 28   | 914.402  | 6.744  | 4.961 |

Selected hyper-parameters: 

- C: 1.8
- $\epsilon$: 0.18
- kernel: linear

##### ^GSPC

| Lags | MSE       | MAE     | MAPE  |
| :--- | --------- | ------- | ----- |
| 1    | 3390.300  | 50.805  | 1.489 |
| 7    | 14818.349 | 90.067  | 2.669 |
| 14   | 23919.770 | 115.385 | 3.362 |
| 28   | 16985.906 | 95.058  | 2.725 |

Selected hyper-parameters: 

- C: 1.8

- $\epsilon$: 0.18

- kernel: linear

  

### Model Evaluation and Validation (as above)
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
