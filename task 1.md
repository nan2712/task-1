```python
import numpy as np
import pandas as pd
```

pip install scikit-learn


```python
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
%matplotlib inline
```


```python
import pandas as pd
```


```python
dataset = pd.read_csv(r"C:\Users\perum\Desktop\spam.csv",encoding='latin1')


```


```python
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>v1</th>
      <th>v2</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#removing unnamed columns
dataset = dataset.drop('Unnamed: 2',axis=1)
dataset = dataset.drop('Unnamed: 3',axis=1)
dataset = dataset.drop('Unnamed: 4',axis=1)
```


```python
dataset = dataset.rename(columns = {'v1':'label','v2':'message'})
```


```python
dataset.groupby('label').describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">message</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ham</th>
      <td>4825</td>
      <td>4516</td>
      <td>Sorry, I'll call later</td>
      <td>30</td>
    </tr>
    <tr>
      <th>spam</th>
      <td>747</td>
      <td>653</td>
      <td>Please call our customer service representativ...</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
count_Class=pd.value_counts(dataset["label"], sort= True)
count_Class.plot(kind = 'bar',color = ["green","red"])
plt.title('Bar Plot')
plt.show();
```


    
![png](output_9_0.png)
    



```python
import numpy as np
```


```python
f = feature_extraction.text.CountVectorizer(stop_words = 'english')
X = f.fit_transform(dataset["message"])
np.shape(X)
```




    (5572, 8404)




```python
# Classifying spam and not spam msgs as 1 and 0

dataset["label"]=dataset["label"].map({'spam':1,'ham':0})
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, dataset['label'], test_size=0.70, random_state=42)
```


```python
dataset.isnull()
dataset.isnull().sum().sum()
dataset.dropna(inplace=True)
```


```python
list_alpha = np.arange(1/100000, 20, 0.11)
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))
count = 0
dataset.replace([np.inf,-np.inf],np.nan,inplace=True)
dataset.fillna(999,inplace=True)
```


```python
for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count]= bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
    count = count + 1 
```


```python
matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns = 
             ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
models.head(n=10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alpha</th>
      <th>Train Accuracy</th>
      <th>Test Accuracy</th>
      <th>Test Recall</th>
      <th>Test Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00001</td>
      <td>0.998803</td>
      <td>0.961805</td>
      <td>0.913793</td>
      <td>0.820998</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.11001</td>
      <td>0.998803</td>
      <td>0.966163</td>
      <td>0.946360</td>
      <td>0.826087</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.22001</td>
      <td>0.999402</td>
      <td>0.967444</td>
      <td>0.938697</td>
      <td>0.837607</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.33001</td>
      <td>0.999402</td>
      <td>0.968726</td>
      <td>0.938697</td>
      <td>0.844828</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.44001</td>
      <td>0.999402</td>
      <td>0.971546</td>
      <td>0.929119</td>
      <td>0.867621</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.55001</td>
      <td>0.998803</td>
      <td>0.976160</td>
      <td>0.925287</td>
      <td>0.899441</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.66001</td>
      <td>0.998803</td>
      <td>0.976160</td>
      <td>0.919540</td>
      <td>0.903955</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.77001</td>
      <td>0.997606</td>
      <td>0.977698</td>
      <td>0.917625</td>
      <td>0.915870</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.88001</td>
      <td>0.997606</td>
      <td>0.977954</td>
      <td>0.909962</td>
      <td>0.924125</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.99001</td>
      <td>0.997606</td>
      <td>0.978980</td>
      <td>0.902299</td>
      <td>0.938247</td>
    </tr>
  </tbody>
</table>
</div>




```python
matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns = 
             ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
models.head(n=10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alpha</th>
      <th>Train Accuracy</th>
      <th>Test Accuracy</th>
      <th>Test Recall</th>
      <th>Test Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00001</td>
      <td>0.998803</td>
      <td>0.961805</td>
      <td>0.913793</td>
      <td>0.820998</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.11001</td>
      <td>0.998803</td>
      <td>0.966163</td>
      <td>0.946360</td>
      <td>0.826087</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.22001</td>
      <td>0.999402</td>
      <td>0.967444</td>
      <td>0.938697</td>
      <td>0.837607</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.33001</td>
      <td>0.999402</td>
      <td>0.968726</td>
      <td>0.938697</td>
      <td>0.844828</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.44001</td>
      <td>0.999402</td>
      <td>0.971546</td>
      <td>0.929119</td>
      <td>0.867621</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.55001</td>
      <td>0.998803</td>
      <td>0.976160</td>
      <td>0.925287</td>
      <td>0.899441</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.66001</td>
      <td>0.998803</td>
      <td>0.976160</td>
      <td>0.919540</td>
      <td>0.903955</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.77001</td>
      <td>0.997606</td>
      <td>0.977698</td>
      <td>0.917625</td>
      <td>0.915870</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.88001</td>
      <td>0.997606</td>
      <td>0.977954</td>
      <td>0.909962</td>
      <td>0.924125</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.99001</td>
      <td>0.997606</td>
      <td>0.978980</td>
      <td>0.902299</td>
      <td>0.938247</td>
    </tr>
  </tbody>
</table>
</div>




```python
best_index = models['Test Precision'].idxmax()
models.iloc[best_index, :]
```




    alpha             10.670010
    Train Accuracy     0.977259
    Test Accuracy      0.962574
    Test Recall        0.720307
    Test Precision     1.000000
    Name: 97, dtype: float64




```python
rf = RandomForestClassifier(n_estimators=100,max_depth=None,n_jobs=-1)
rf_model = rf.fit(X_train,y_train)
```


```python
y_pred=rf_model.predict(X_test)
precision,recall,fscore,support =score(y_test,y_pred,pos_label=1, average ='binary')
print('Precision : {} / Recall : {} / fscore : {} / Accuracy: {}'.format(round(precision,3),round(recall,3),round(fscore,3),round((y_pred==y_test).sum()/len(y_test),3)))
```

    Precision : 0.995 / Recall : 0.718 / fscore : 0.834 / Accuracy: 0.962
    


```python
pip install tensorflow
```

    Requirement already satisfied: tensorflow in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (2.10.0)
    Requirement already satisfied: absl-py>=1.0.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (2.0.0)
    Requirement already satisfied: astunparse>=1.6.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (1.6.3)
    Requirement already satisfied: flatbuffers>=2.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (23.5.26)
    Requirement already satisfied: gast<=0.4.0,>=0.2.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (0.4.0)
    Requirement already satisfied: google-pasta>=0.1.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (0.2.0)
    Requirement already satisfied: h5py>=2.9.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (3.7.0)
    Requirement already satisfied: keras-preprocessing>=1.1.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (1.1.2)
    Requirement already satisfied: libclang>=13.0.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (16.0.6)
    Requirement already satisfied: numpy>=1.20 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (1.21.6)
    Requirement already satisfied: opt-einsum>=2.3.2 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (3.3.0)
    Requirement already satisfied: packaging in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (23.2)
    Requirement already satisfied: protobuf<3.20,>=3.9.2 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (3.19.6)
    Requirement already satisfied: setuptools in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (68.2.2)
    Requirement already satisfied: six>=1.12.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (1.16.0)
    Requirement already satisfied: termcolor>=1.1.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (2.3.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (4.7.1)
    Requirement already satisfied: wrapt>=1.11.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (1.14.1)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (0.31.0)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (1.48.1)
    Requirement already satisfied: tensorboard<2.11,>=2.10 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (2.10.1)
    Requirement already satisfied: tensorflow-estimator<2.11,>=2.10.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (2.10.0)
    Requirement already satisfied: keras<2.11,>=2.10.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (2.10.0)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from astunparse>=1.6.0->tensorflow) (0.42.0)
    Requirement already satisfied: google-auth<3,>=1.6.3 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorboard<2.11,>=2.10->tensorflow) (2.24.0)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorboard<2.11,>=2.10->tensorflow) (0.4.6)
    Requirement already satisfied: markdown>=2.6.8 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorboard<2.11,>=2.10->tensorflow) (3.5.1)
    Requirement already satisfied: requests<3,>=2.21.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorboard<2.11,>=2.10->tensorflow) (2.31.0)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorboard<2.11,>=2.10->tensorflow) (0.6.0)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorboard<2.11,>=2.10->tensorflow) (1.8.1)
    Requirement already satisfied: werkzeug>=1.0.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorboard<2.11,>=2.10->tensorflow) (2.2.3)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow) (5.3.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow) (0.3.0)
    Requirement already satisfied: rsa<5,>=3.1.4 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow) (4.9)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.11,>=2.10->tensorflow) (1.3.1)
    Requirement already satisfied: importlib-metadata>=4.4 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from markdown>=2.6.8->tensorboard<2.11,>=2.10->tensorflow) (4.11.4)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow) (2.1.0)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow) (2023.11.17)
    Requirement already satisfied: MarkupSafe>=2.1.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from werkzeug>=1.0.1->tensorboard<2.11,>=2.10->tensorflow) (2.1.1)
    Requirement already satisfied: zipp>=0.5 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.11,>=2.10->tensorflow) (3.15.0)
    Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow) (0.5.1)
    Requirement already satisfied: oauthlib>=3.0.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.11,>=2.10->tensorflow) (3.2.2)
    Note: you may need to restart the kernel to use updated packages.
    


```python
pip install tensorflow
```

    Requirement already satisfied: tensorflow in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (2.10.0)
    Requirement already satisfied: absl-py>=1.0.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (2.0.0)
    Requirement already satisfied: astunparse>=1.6.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (1.6.3)
    Requirement already satisfied: flatbuffers>=2.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (23.5.26)
    Requirement already satisfied: gast<=0.4.0,>=0.2.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (0.4.0)
    Requirement already satisfied: google-pasta>=0.1.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (0.2.0)
    Requirement already satisfied: h5py>=2.9.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (3.7.0)
    Requirement already satisfied: keras-preprocessing>=1.1.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (1.1.2)
    Requirement already satisfied: libclang>=13.0.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (16.0.6)
    Requirement already satisfied: numpy>=1.20 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (1.21.6)
    Requirement already satisfied: opt-einsum>=2.3.2 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (3.3.0)
    Requirement already satisfied: packaging in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (23.2)
    Requirement already satisfied: protobuf<3.20,>=3.9.2 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (3.19.6)
    Requirement already satisfied: setuptools in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (68.2.2)
    Requirement already satisfied: six>=1.12.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (1.16.0)
    Requirement already satisfied: termcolor>=1.1.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (2.3.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (4.7.1)
    Requirement already satisfied: wrapt>=1.11.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (1.14.1)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (0.31.0)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (1.48.1)
    Requirement already satisfied: tensorboard<2.11,>=2.10 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (2.10.1)
    Requirement already satisfied: tensorflow-estimator<2.11,>=2.10.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (2.10.0)
    Requirement already satisfied: keras<2.11,>=2.10.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorflow) (2.10.0)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from astunparse>=1.6.0->tensorflow) (0.42.0)
    Requirement already satisfied: google-auth<3,>=1.6.3 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorboard<2.11,>=2.10->tensorflow) (2.24.0)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorboard<2.11,>=2.10->tensorflow) (0.4.6)
    Requirement already satisfied: markdown>=2.6.8 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorboard<2.11,>=2.10->tensorflow) (3.5.1)
    Requirement already satisfied: requests<3,>=2.21.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorboard<2.11,>=2.10->tensorflow) (2.31.0)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorboard<2.11,>=2.10->tensorflow) (0.6.0)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorboard<2.11,>=2.10->tensorflow) (1.8.1)
    Requirement already satisfied: werkzeug>=1.0.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from tensorboard<2.11,>=2.10->tensorflow) (2.2.3)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow) (5.3.2)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow) (0.3.0)
    Requirement already satisfied: rsa<5,>=3.1.4 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow) (4.9)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.11,>=2.10->tensorflow) (1.3.1)
    Requirement already satisfied: importlib-metadata>=4.4 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from markdown>=2.6.8->tensorboard<2.11,>=2.10->tensorflow) (4.11.4)
    Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow) (2.1.0)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from requests<3,>=2.21.0->tensorboard<2.11,>=2.10->tensorflow) (2023.11.17)
    Requirement already satisfied: MarkupSafe>=2.1.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from werkzeug>=1.0.1->tensorboard<2.11,>=2.10->tensorflow) (2.1.1)
    Requirement already satisfied: zipp>=0.5 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.11,>=2.10->tensorflow) (3.15.0)
    Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.11,>=2.10->tensorflow) (0.5.1)
    Requirement already satisfied: oauthlib>=3.0.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.11,>=2.10->tensorflow) (3.2.2)
    Note: you may need to restart the kernel to use updated packages.
    


```python
pip install dataset
```

    Requirement already satisfied: dataset in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (1.6.2)Note: you may need to restart the kernel to use updated packages.
    
    Requirement already satisfied: sqlalchemy<2.0.0,>=1.3.2 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from dataset) (1.4.50)
    Requirement already satisfied: alembic>=0.6.2 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from dataset) (1.12.1)
    Requirement already satisfied: banal>=1.0.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from dataset) (1.0.6)
    Requirement already satisfied: Mako in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from alembic>=0.6.2->dataset) (1.2.4)
    Requirement already satisfied: typing-extensions>=4 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from alembic>=0.6.2->dataset) (4.7.1)
    Requirement already satisfied: importlib-metadata in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from alembic>=0.6.2->dataset) (4.11.4)
    Requirement already satisfied: importlib-resources in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from alembic>=0.6.2->dataset) (6.0.0)
    Requirement already satisfied: greenlet!=0.4.17 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from sqlalchemy<2.0.0,>=1.3.2->dataset) (3.0.1)
    Requirement already satisfied: zipp>=0.5 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from importlib-metadata->alembic>=0.6.2->dataset) (3.15.0)
    Requirement already satisfied: MarkupSafe>=0.9.2 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from Mako->alembic>=0.6.2->dataset) (2.1.1)
    


```python
pip install keras
```

    Requirement already satisfied: keras in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (2.10.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
```


```python
vocab_size = 400
oov_tok = ""
max_length = 250
embedding_dim = 16
encode = ({'ham': 0, 'spam': 1} )

```


```python
X = dataset['message']
Y = dataset['label']
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X)
# convert to sequence of integers
X = tokenizer.texts_to_sequences(X)
```


```python
X = np.array(X,dtype=object)
y = np.array(Y,dtype=object)
```


```python
X = pad_sequences(X, maxlen=max_length)
```


```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding (Embedding)       (None, 250, 16)           6400      
                                                                     
     global_average_pooling1d (G  (None, 16)               0         
     lobalAveragePooling1D)                                          
                                                                     
     dense (Dense)               (None, 24)                408       
                                                                     
     dense_1 (Dense)             (None, 1)                 25        
                                                                     
    =================================================================
    Total params: 6,833
    Trainable params: 6,833
    Non-trainable params: 0
    _________________________________________________________________
    


```python
num_epochs = 50
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.20, random_state=7)
history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_test,y_test), verbose=2)
```

    Epoch 1/50
    140/140 - 2s - loss: 0.5311 - accuracy: 0.8429 - val_loss: 0.3830 - val_accuracy: 0.8700 - 2s/epoch - 16ms/step
    Epoch 2/50
    140/140 - 1s - loss: 0.3818 - accuracy: 0.8649 - val_loss: 0.3668 - val_accuracy: 0.8700 - 599ms/epoch - 4ms/step
    Epoch 3/50
    140/140 - 1s - loss: 0.3692 - accuracy: 0.8649 - val_loss: 0.3531 - val_accuracy: 0.8700 - 690ms/epoch - 5ms/step
    Epoch 4/50
    140/140 - 1s - loss: 0.3520 - accuracy: 0.8649 - val_loss: 0.3329 - val_accuracy: 0.8700 - 655ms/epoch - 5ms/step
    Epoch 5/50
    140/140 - 1s - loss: 0.3203 - accuracy: 0.8647 - val_loss: 0.2880 - val_accuracy: 0.8700 - 612ms/epoch - 4ms/step
    Epoch 6/50
    140/140 - 1s - loss: 0.2597 - accuracy: 0.8681 - val_loss: 0.2174 - val_accuracy: 0.8843 - 580ms/epoch - 4ms/step
    Epoch 7/50
    140/140 - 1s - loss: 0.1954 - accuracy: 0.9183 - val_loss: 0.1655 - val_accuracy: 0.9462 - 537ms/epoch - 4ms/step
    Epoch 8/50
    140/140 - 1s - loss: 0.1568 - accuracy: 0.9495 - val_loss: 0.1389 - val_accuracy: 0.9480 - 551ms/epoch - 4ms/step
    Epoch 9/50
    140/140 - 1s - loss: 0.1341 - accuracy: 0.9587 - val_loss: 0.1201 - val_accuracy: 0.9570 - 540ms/epoch - 4ms/step
    Epoch 10/50
    140/140 - 1s - loss: 0.1193 - accuracy: 0.9632 - val_loss: 0.1065 - val_accuracy: 0.9641 - 534ms/epoch - 4ms/step
    Epoch 11/50
    140/140 - 1s - loss: 0.1069 - accuracy: 0.9677 - val_loss: 0.0944 - val_accuracy: 0.9668 - 569ms/epoch - 4ms/step
    Epoch 12/50
    140/140 - 1s - loss: 0.0980 - accuracy: 0.9688 - val_loss: 0.0854 - val_accuracy: 0.9704 - 607ms/epoch - 4ms/step
    Epoch 13/50
    140/140 - 1s - loss: 0.0904 - accuracy: 0.9702 - val_loss: 0.0793 - val_accuracy: 0.9749 - 599ms/epoch - 4ms/step
    Epoch 14/50
    140/140 - 1s - loss: 0.0846 - accuracy: 0.9724 - val_loss: 0.0723 - val_accuracy: 0.9767 - 619ms/epoch - 4ms/step
    Epoch 15/50
    140/140 - 1s - loss: 0.0782 - accuracy: 0.9737 - val_loss: 0.0666 - val_accuracy: 0.9767 - 608ms/epoch - 4ms/step
    Epoch 16/50
    140/140 - 1s - loss: 0.0735 - accuracy: 0.9755 - val_loss: 0.0630 - val_accuracy: 0.9803 - 587ms/epoch - 4ms/step
    Epoch 17/50
    140/140 - 1s - loss: 0.0690 - accuracy: 0.9762 - val_loss: 0.0617 - val_accuracy: 0.9812 - 589ms/epoch - 4ms/step
    Epoch 18/50
    140/140 - 1s - loss: 0.0657 - accuracy: 0.9791 - val_loss: 0.0568 - val_accuracy: 0.9812 - 587ms/epoch - 4ms/step
    Epoch 19/50
    140/140 - 1s - loss: 0.0622 - accuracy: 0.9812 - val_loss: 0.0550 - val_accuracy: 0.9821 - 596ms/epoch - 4ms/step
    Epoch 20/50
    140/140 - 1s - loss: 0.0600 - accuracy: 0.9794 - val_loss: 0.0526 - val_accuracy: 0.9812 - 591ms/epoch - 4ms/step
    Epoch 21/50
    140/140 - 1s - loss: 0.0589 - accuracy: 0.9812 - val_loss: 0.0515 - val_accuracy: 0.9830 - 581ms/epoch - 4ms/step
    Epoch 22/50
    140/140 - 1s - loss: 0.0566 - accuracy: 0.9803 - val_loss: 0.0497 - val_accuracy: 0.9830 - 585ms/epoch - 4ms/step
    Epoch 23/50
    140/140 - 1s - loss: 0.0545 - accuracy: 0.9818 - val_loss: 0.0487 - val_accuracy: 0.9839 - 563ms/epoch - 4ms/step
    Epoch 24/50
    140/140 - 1s - loss: 0.0526 - accuracy: 0.9823 - val_loss: 0.0475 - val_accuracy: 0.9848 - 586ms/epoch - 4ms/step
    Epoch 25/50
    140/140 - 1s - loss: 0.0509 - accuracy: 0.9821 - val_loss: 0.0471 - val_accuracy: 0.9839 - 555ms/epoch - 4ms/step
    Epoch 26/50
    140/140 - 1s - loss: 0.0500 - accuracy: 0.9832 - val_loss: 0.0531 - val_accuracy: 0.9821 - 583ms/epoch - 4ms/step
    Epoch 27/50
    140/140 - 1s - loss: 0.0484 - accuracy: 0.9843 - val_loss: 0.0459 - val_accuracy: 0.9865 - 582ms/epoch - 4ms/step
    Epoch 28/50
    140/140 - 1s - loss: 0.0467 - accuracy: 0.9843 - val_loss: 0.0456 - val_accuracy: 0.9874 - 563ms/epoch - 4ms/step
    Epoch 29/50
    140/140 - 1s - loss: 0.0460 - accuracy: 0.9847 - val_loss: 0.0463 - val_accuracy: 0.9857 - 568ms/epoch - 4ms/step
    Epoch 30/50
    140/140 - 1s - loss: 0.0448 - accuracy: 0.9838 - val_loss: 0.0436 - val_accuracy: 0.9883 - 588ms/epoch - 4ms/step
    Epoch 31/50
    140/140 - 1s - loss: 0.0436 - accuracy: 0.9859 - val_loss: 0.0476 - val_accuracy: 0.9830 - 617ms/epoch - 4ms/step
    Epoch 32/50
    140/140 - 1s - loss: 0.0429 - accuracy: 0.9863 - val_loss: 0.0430 - val_accuracy: 0.9883 - 626ms/epoch - 4ms/step
    Epoch 33/50
    140/140 - 1s - loss: 0.0429 - accuracy: 0.9868 - val_loss: 0.0443 - val_accuracy: 0.9883 - 581ms/epoch - 4ms/step
    Epoch 34/50
    140/140 - 1s - loss: 0.0409 - accuracy: 0.9870 - val_loss: 0.0441 - val_accuracy: 0.9892 - 602ms/epoch - 4ms/step
    Epoch 35/50
    140/140 - 1s - loss: 0.0391 - accuracy: 0.9865 - val_loss: 0.0453 - val_accuracy: 0.9830 - 645ms/epoch - 5ms/step
    Epoch 36/50
    140/140 - 1s - loss: 0.0400 - accuracy: 0.9859 - val_loss: 0.0424 - val_accuracy: 0.9892 - 595ms/epoch - 4ms/step
    Epoch 37/50
    140/140 - 1s - loss: 0.0394 - accuracy: 0.9872 - val_loss: 0.0434 - val_accuracy: 0.9892 - 600ms/epoch - 4ms/step
    Epoch 38/50
    140/140 - 1s - loss: 0.0379 - accuracy: 0.9863 - val_loss: 0.0434 - val_accuracy: 0.9848 - 587ms/epoch - 4ms/step
    Epoch 39/50
    140/140 - 1s - loss: 0.0381 - accuracy: 0.9886 - val_loss: 0.0524 - val_accuracy: 0.9857 - 589ms/epoch - 4ms/step
    Epoch 40/50
    140/140 - 1s - loss: 0.0370 - accuracy: 0.9877 - val_loss: 0.0418 - val_accuracy: 0.9865 - 588ms/epoch - 4ms/step
    Epoch 41/50
    140/140 - 1s - loss: 0.0367 - accuracy: 0.9883 - val_loss: 0.0419 - val_accuracy: 0.9865 - 588ms/epoch - 4ms/step
    Epoch 42/50
    140/140 - 1s - loss: 0.0356 - accuracy: 0.9883 - val_loss: 0.0440 - val_accuracy: 0.9901 - 554ms/epoch - 4ms/step
    Epoch 43/50
    140/140 - 1s - loss: 0.0337 - accuracy: 0.9892 - val_loss: 0.0410 - val_accuracy: 0.9892 - 543ms/epoch - 4ms/step
    Epoch 44/50
    140/140 - 1s - loss: 0.0343 - accuracy: 0.9879 - val_loss: 0.0415 - val_accuracy: 0.9901 - 539ms/epoch - 4ms/step
    Epoch 45/50
    140/140 - 1s - loss: 0.0338 - accuracy: 0.9899 - val_loss: 0.0411 - val_accuracy: 0.9883 - 574ms/epoch - 4ms/step
    Epoch 46/50
    140/140 - 1s - loss: 0.0328 - accuracy: 0.9895 - val_loss: 0.0412 - val_accuracy: 0.9901 - 594ms/epoch - 4ms/step
    Epoch 47/50
    140/140 - 1s - loss: 0.0322 - accuracy: 0.9897 - val_loss: 0.0412 - val_accuracy: 0.9901 - 581ms/epoch - 4ms/step
    Epoch 48/50
    140/140 - 1s - loss: 0.0320 - accuracy: 0.9892 - val_loss: 0.0412 - val_accuracy: 0.9901 - 630ms/epoch - 4ms/step
    Epoch 49/50
    140/140 - 1s - loss: 0.0310 - accuracy: 0.9901 - val_loss: 0.0419 - val_accuracy: 0.9883 - 606ms/epoch - 4ms/step
    Epoch 50/50
    140/140 - 1s - loss: 0.0310 - accuracy: 0.9886 - val_loss: 0.0479 - val_accuracy: 0.9892 - 572ms/epoch - 4ms/step
    


```python
results = model.evaluate(X_test, y_test)
loss = results[0]
accuracy = results[1]


print(f"[+] Accuracy: {accuracy*100:.2f}%")
```

    35/35 [==============================] - 0s 3ms/step - loss: 0.0479 - accuracy: 0.9892
    [+] Accuracy: 98.92%
    


```python
pip install Keras_Preprocessing
```

    Requirement already satisfied: Keras_Preprocessing in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (1.1.2)
    Requirement already satisfied: numpy>=1.9.1 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from Keras_Preprocessing) (1.21.6)
    Requirement already satisfied: six>=1.9.0 in c:\users\perum\anaconda3\envs\my-env\lib\site-packages (from Keras_Preprocessing) (1.16.0)
    Note: you may need to restart the kernel to use updated packages.
    


```python
from keras_preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences

```


```python
#Defining the function
def get_predictions(txts):
    txts = tokenizer.texts_to_sequences(txts)
    txts = sequence.pad_sequences(txts, maxlen=max_length)
    preds = model.predict(txts)
    if(preds[0] > 0.5):
        print("SPAM MESSAGE")
        
    else:
        print('NOT SPAM')
```


```python
txts=["You have won a free ticket to las vegas. Contact now"]

get_predictions(txts)
```

    1/1 [==============================] - 5s 5s/step
    SPAM MESSAGE
    


```python
txts=["Hey there call me asap!!"]

get_predictions(txts)
```

    1/1 [==============================] - 0s 47ms/step
    NOT SPAM
    


```python

```
