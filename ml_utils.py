## for data
import numpy as np
import pandas as pd

## for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

## for statistical tests
import scipy
import scipy.stats as stats

## for machine learning
from sklearn import preprocessing, impute, utils, linear_model, feature_selection, model_selection, metrics, decomposition, cluster, ensemble, set_config, tree, neighbors
from imblearn import over_sampling, pipeline
import xgboost as xgb

## feature encoding
from category_encoders.hashing import HashingEncoder

## save file/model
import joblib

###############################################################################
###############################################################################


def outliers_subplot(data,col1,col2):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fig.tight_layout(w_pad=5.0)

    sns.set_style("whitegrid")
    custom_palette = sns.color_palette("muted")

    sns.boxplot(y=data[col1], ax=ax[0], color=custom_palette[0])
    sns.boxplot(y=data[col2], ax=ax[1], color=custom_palette[1])
  
    ax[0].set_ylabel("Income", fontsize=14)
    ax[1].set_ylabel("Age", fontsize=14)

    for i in range(2):
        ax[i].grid(False)
        ax[i].spines["right"].set_visible(False)
        ax[i].spines["top"].set_visible(False)
    
    return fig, ax

def num_dist(data):
    var_group = data.columns
    plt.figure(figsize=(12,7), dpi=400)

    for j,i in enumerate(var_group):
        
        mini = data[i].min()
        maxi = data[i].max()
        ran = data[i].max() - data[i].min()
        mean = data[i].mean()
        median = data[i].median()
        st_dev = data[i].std()
        skew = data[i].skew()
        kurt = data[i].kurtosis()

        points = mean-st_dev, mean+st_dev

        plt.subplot(1,2,j+1)
        
        sns.distplot(data[i], hist=True, kde=True)
        
        sns.lineplot(x=points, y=[0,0], color='black', label="std_dev")
        sns.scatterplot(x=[mini,maxi], y=[0,0], color='orange', label="min/max")
        sns.scatterplot(x=[mean], y=[0], color='red', label="mean")
        sns.scatterplot(x=[median], y=[0], color='blue', label="median")
        plt.xlabel('{}'.format(i), fontsize=20)
        plt.ylabel('density')
        plt.title('std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}'.format((round(points[0],2),round(points[1],2)),round(kurt,2),round(skew,2),(round(mini,2),round(maxi,2),round(ran,2)),round(mean,2),round(median,2)))
        sns.despine(top=True, right=True)
        ymin, ymax = plt.gca().get_ylim()
        plt.grid(False)
        plt.ylim(min(ymin, -0.05 * ymax), ymax)
    plt.tight_layout()
    
def cat_dist(data, x, hue, label1, label2, palette):
    sns.set_style("whitegrid")
    ax = sns.countplot(data=data, x=x, hue=hue, palette=palette)
    sns.despine(top=True, right=True)
    plt.xlabel(f"{x}", fontsize=15)
    plt.ylabel("Count", fontsize=15)
    plt.legend([label1, label2])
    plt.grid(False)

    for i in ax.patches:
        ax.text(i.get_x()+0.1, i.get_height()+3000, int(i.get_height()), fontsize=11)
        
def target_dist(data,col,label1,label2):
    mpl.rcParams['font.size'] = 11
    r = data.groupby(col)[col].count()
    fig, ax = plt.subplots(figsize=(3,3))
    ax.pie(r, explode=[0.05, 0.1], labels=[label1, label2], radius=1.5, autopct='%1.1f%%', shadow=True, startangle=45,
           colors=['#66b3ff', '#ff9999'])
    ax.set_aspect('equal')
    ax.set_frame_on(False)
    
def num_corr(data):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    correlation = data.corr()
    mask = np.zeros_like(correlation, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(correlation, annot=True, mask=mask, cmap='coolwarm', annot_kws={"size": 11})
    sns.despine(left=True, bottom=True)
    plt.grid(False)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.title("Correlation Matrix", fontsize=15, fontweight='bold')    
    
def chi_square_test(data, col_x, col_target):
    crosstab = pd.crosstab(data[col_x], data[col_target])
    stat, p, dof, expected = stats.chi2_contingency(crosstab)
    alpha = 0.05
    print("p-value is: ", p)
    if p <= alpha:
        print('Dependent (reject H0)\nThis feature is dependent on the target variable.')
    else:
        print('Independent (H0 holds true)\nThis feature is independent on the target variable.')

model_df={}
def model_val(model,X,y):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.20, random_state = 42)
   
    pipe = pipeline.Pipeline(steps = [("smote", over_sampling.SMOTE(random_state = 42)), 
                      ("standardscaler", preprocessing.StandardScaler()),
                      (f"{model}", model)])
    
    pipe.fit(X_train, y_train)
    
    score = model_selection.cross_validate(pipe, X_train, y_train)
    cross_val_score = model_selection.cross_validate(pipe, X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    
    cv_scores = cross_val_score['test_score']
    print(f"{model} cross validation score is {np.mean(cv_scores)}")
    print(f"{model} accuracy score is {metrics.accuracy_score(y_test,y_pred)}")
    print(f"{model} precision score is {metrics.precision_score(y_test, y_pred)}")
    print(f"{model} recall score is {metrics.recall_score(y_test, y_pred)}")
    print(f"{model} f1 score is {metrics.f1_score(y_test, y_pred)}")
    model_df[model] = round(np.mean(cv_scores)*100,2)
    set_config(display='diagram')
    display(pipe)
    joblib.dump(pipe, 'trained_model_w_pipe.joblib')