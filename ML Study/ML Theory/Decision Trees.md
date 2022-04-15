# Decision Trees

## Tables of Content
* [Decision Tree](#decision-tree)
* [Random Forest](#random-forest)
* [AdaBoost](#adaboost)
* [Gradient Boosting Decision Tree](#gradient-boosting-decision-tree)
* [XGBoost](#xgboost)


## Decision Tree
- [Decision Tree In Python. An example of how to implement a… | by Cory Maklin | Towards Data Science](https://towardsdatascience.com/decision-tree-in-python-b433ae57fb93)

It is a model useful to enhance our decision making abilities, it uses logic and math to create rules.
It helps to select the discriminant variable in a yes/no question.

The top of the tree is the **root node**, there are also **intermediate nodes** or just nodes and finally at the bootom **leaves**. The leaves contain the classification.

The algorithm evaluates how much each feature can perform classification. In order to decide which is the better one **impurity** is used.
Impurity referes to the fact that none of the leaves have 100% correct classification. One way to measure it is through **gini** criterion.
$$left\ impurty = 1-(\frac{PredicPos}{LeftLeafSample})^2-(\frac{PredicNeg}{LeftLeafSample})^2$$
Same for right impurity. While impurity for the node
$$Impurity = 1 - (\frac{\#sampleLeft}{totSample})^2 - (\frac{\#sampleRight}{totSample})^2$$
$$Informtion\ Gain = Impurity - \frac{\#sampleLeft}{totSample}*left\ impurity - \frac{\#sampleRight}{totSample}*right\ impurity$$
This process is performed for every feature. The feature with the highest information gain is selected.
Then for each node the process is repeated. Every branch can have different number of samples.

In order to stop splits it is used the **minimum impurity decrease**, a node is split if the split introduces a decrease of the impurity greater than or equal to the selected value.

### Code
``` python
from sklearn.datasets import load_iris  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix  
from sklearn.tree import export_graphviz  
from sklearn.externals.six import StringIO   
from IPython.display import Image   
from pydot import graph_from_dot_data  
import pandas as pd  
import numpy as np

iris = load_iris()  
X = pd.DataFrame(iris.data, columns=iris.feature_names)  
y = pd.Categorical.from_codes(iris.target, iris.target_names)

# Encoding of categorical variables in order to build a confusion matric later
y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Instance of decision tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Tree visualization
dot_data = StringIO()
export_graphviz(dt, out_file=dot_data, feature_names=iris.feature_names)(graph, ) = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

# Model predictions
y_pred = dt.predict(X_test)                

# metrics
species = np.array(y_test).argmax(axis=1)  
predictions = np.array(y_pred).argmax(axis=1)  
confusion_matrix(species, predictions)
```

Note normalization is not required and they can be applied both to classification and regression problems. However, they don't generalize well.

## Random Forest
- [Random Forest In Python. Random forest is one of the most… | by Cory Maklin | Towards Data Science](https://towardsdatascience.com/random-forest-in-python-154d78aad254)

Random Forest can be applied to both regression and classification problems.
The reason why it is such a popular algorithm is model explainability, this is a key feature in many businesses, furthermpre it is strictly required in issuing loans or insurances.

### Theory
The random forest works by aggregating the predictions made by multiple decision trees of varying depth. Every tree is trained on a subset of the dataset called **bootstrapped dataset**.
The portion of samples left out during the construction of each tree is the **Out-Of-Bag (OOB) dataset** and it is used to evaluate the performances of the model.

For each tree i(of the forest) a random predefined number of features are selected as candidate, this will result in a larger variance between trees (instead of having just the features which are highly correlated with target label).

In classification problems the label of a sample is predicted taking the majority of the predictions made by each individual decision tree in the forest. In regression tasks is the average.

### Code
``` python
from sklearn.ensemble import RandomForestClassifier  
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix  
import pandas as pd  
import numpy as np  
from sklearn.tree import export_graphviz  
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from pydot import graph_from_dot_data

iris = load_iris()  
X = pd.DataFrame(iris.data, columns=iris.feature_names)  
y = pd.Categorical.from_codes(iris.target, iris.target_names)

# Since the RandomForestCassifier can't handle categorical data each species mut be encoded as a number
y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

rf = RandomForestClassifier(criterion='entropy', oob_score=True, random_state=1)
rf.fit(X_train, y_train)

# estimators_ contains an array of the decisiontree objects that make up the forest
dt = rf.estimators_[0]

# Visualize a given decision tree (we choose to visualize dt)
dot_data = StringIO()
export_graphviz(dt, out_file=dot_data, feature_names=iris.feature_names)
(graph, ) = graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

# The forest is evaluated using OOB
rf.oob_score_

# And of course against testing set
y_pred = rf.predict(X_test)
species = np.array(y_test).argmax(axis=1)  
predictions = np.array(y_pred).argmax(axis=1)
confusion_matrix(species, predictions)
```

The decision criteria is entropy (information gain):
$$Entropy = - \sum p_i log_2(p_i)$$
The impurity $p_i$ for the node is equal to the fraction of samples in the left child + the fraction of samples in the right child. This will produce an information gain which will be used to make the split with the largest information gain.

## AdaBoost
### Theory
- https://towardsdatascience.com/machine-learning-part-17-boosting-algorithms-adaboost-in-python-d00faac6c464?gi=7d414be50b4e

The general idea behind boosting methods is to train predictors sequentially, each trying to correct its predecessor.

AdaBoost and Gradient Boosting are the two most popular algorithms.

AdaBoost is similar to random forest, they bost uses the predictions made by each tree to make the final classification.

**In AdaBoost the decision trees have a depth of 1 (i.e. 2 leaves)**, also the predictions made by each tree does not have the same impact on the final prediction made by the model.

Consider the following example, you want to classify people as attractive or not and you have this dataset:

![[AdaBoost_dataset.png]]
*This picture already has weights, which are not part of the original dataset*

Steps:
1. Each sample is associated with a weight that indicates how important is with regard to the classification. Initially all have same weight (1/number of samples)
2. Decision tree with depth = 1 is built **for each feature**, than the predictions made by each tree are compared with the true label. The tree who got the best results becomes the next tree in the forest. Following an example of tree:
![[AdaBoost_tree.png]]
3. Calculate the significance of the tree in the final classification. Total error is the sum of the weights of samples incorrectly classified, in our case since there is only 1 incorrect it would be = 1/8$$significance = \frac{1}{2}log(\frac{1-totalError}{totalError})$$
4. update the sample weights, the goal is to decrease weights of correct samples and increase wrong ones (the goal is to have 0 miss classified therefore if you make an error the sample will be weighted more) $$wrong\ sample:\ newWeight = sampleWeight\ * e^{significance} $$$$correct\ sample:\ newWeight = sampleWeight\ * e^{-significance} $$Since the sum of the new weights would be different from 1 the weights are normalized.
5. Form a new dataset, the higher the weight the higher is the probability to be part of the new dataset, furthermore the dataset can contain multiple copies of samples if its weights are far greater than the correct one. Therefore when we check all tree again (step 2 again) this time the best tree will have a good amount of samples correctly classified which were wrong the first time.
6. Repeat step 2 to 5 a number of time specified by an hyperparameter called number of estimators.
7. Now you have your forest since you selected the best tree in each iteration, now use the forest to make predictions.
   The sample goes through each tree, the trees are divided in 2 groups (according to their prediction) and each tree is associated with his significance. The group with higher sum of significance is the winning one.
![[AdaBoost_final.png]]

AdaBoost does not take the average (or majority if classification task) of the predictions made by each tree, here each tree contributes to the final prediction with certain weight (its significance).

### Code
``` python
 
from sklearn.ensemble import AdaBoostClassifier  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.datasets import load_breast_cancer  
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix  
from sklearn.preprocessing import LabelEncoder  

breast_cancer = load_breast_cancer()
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)  
y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)

# Categorical feature are encoded as numbers, malignant 1, benignant 0
encoder = LabelEncoder()  
binary_encoded_y = pd.Series(encoder.fit_transform(y))

# train test split
train_X, test_X, train_y, test_y = train_test_split(X, binary_encoded_y, random_state=1)

# max depth=1 will create trees with 1 decision node and 2 leaves, n_estimators is the number of trees (iteration 2-5)
classifier = AdaBoostClassifier(  
    DecisionTreeClassifier(max_depth=1),  
    n_estimators=200)
classifier.fit(train_X, train_y)

predictions = classifier.predict(test_X)

confusion_matrix(test_y, predictions)
```

## Gradient Boosting Decision Tree
- [Gradient Boosting Decision Tree Algorithm Explained | by Cory Maklin | Towards Data Science](https://towardsdatascience.com/machine-learning-part-18-boosting-algorithms-gradient-boosting-in-python-ef5ae6965be4)

### Theory
Gradient Boosting is similar to AdaBoost, they both use an ensemble of decision trees to predict a target label. However, unlike AdaBoost, the Gradient Boost trees have a depth larger than 1.

Gradient boost is typical used with 8 to 32 leaves.

We want to predict the price of a house given certain features:
![[GradBoost_dataset.png]]
*the dataset has already residuals, which were not present in the original dataset*

1. calculate the average of the target label, this will be a baseline to approach the correct solution
2. calculate residuals
3. build a decision tree in order to predict the residuals, every leaf will have a prediction about the value of the residual ![[GradBoost_tree.png]] If there are more residual than leaves some residuals will be in the same leaf. If this happen the mean is computed and it is put inside the leaf (taking the place of the 2 values, in the example leaf1= -338, leaf4 = 512).
4. Predict the target label using all of the trees within the ensamble.
   In particular, each sample passes through the decision nodes of the tree until it reach a leaf.
   The tree will predict a residual which will give the price of the house when combined with the average. Also, a learning rate is introduced in order to lower variance while mantaining the same bias, this will lead to better results on test set.
   The learning rate is multiplied to the predicted residual.
   Therefore multiple trees are needed to make a prediction, each tree will make a small step toward the result.
   i.e. $predicted Price = 688 + 0.1*(-338) = 654.2$  (row 2)
5. Compute the new residuals. $newResidual=actualPrice-predictedPrice=350-654.2=-304.2$  the new residuals will be used in leaves like done in step 3
6. Repeat 3 to 5 until the number of estimator is reached
7. Once trained use all of the trees in the ensamble to make a final prediction![[GradBoost_result.png]]

### Code
``` python
from sklearn.ensemble import GradientBoostingRegressor  
import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error  
from sklearn.datasets import load_boston  
from sklearn.metrics import mean_absolute_error

boston = load_boston()  
X = pd.DataFrame(boston.data, columns=boston.feature_names)  
y = pd.Series(boston.target)
X_train, X_test, y_train, y_test = train_test_split(X, y)

regressor = GradientBoostingRegressor(  
	max_depth=2,  # number of leaves
	n_estimators=3,  # total number of trees in the ensamble
	learning_rate=1.0  # scaling the contribution of each tree
)  
regressor.fit(X_train, y_train)

# staged_predict() measures the validation error at each stage of the training (with 1 tree, with 2, ecc) to find the optimal number of trees
errors = [mean_squared_error(y_test, y_pred) for y_pred in regressor.staged_predict(X_test)]
best_n_estimators = np.argmin(errors)

# We train again the forest using the best hyperparameter combination
best_regressor = GradientBoostingRegressor(  
	max_depth=2,  
	n_estimators=best_n_estimators,  
	learning_rate=1.0  
)  
best_regressor.fit(X_train, y_train)

y_pred = best_regressor.predict(X_test)
mean_absolute_error(y_test, y_pred)
```

## XGBoost
- [XGBoost Python Example. XGBoost is short for Extreme Gradient… | by Cory Maklin | Towards Data Science](https://towardsdatascience.com/xgboost-python-example-42777d01001e)

Extreme Gradient Boost make use of regularization parameters that helps against overfit.

### Theory
1. If we build a model which aims to predict house price given their square footage we could start using the average as a dumb model. In case of classification it could be just 0.5.
2. For each sample we calculate the residual $residual = actual\ value - predicted\ value$ 
   (i.e. -30.000, -24.000, 2.500, -3.000, 30.000)
3. A threshold along the features (in our case only square footage) is chosen![[xgboost_threshold.png]] And on the base of the threshold a tree is built![[xgboost_tree.png]] The value inside leaves are the residual.
4. The gain is calculated, it is the improvement in accuracy brough about by the split. $$Gain = \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} - \gamma$$
   - $G_L = \sum{residuals\ in\ left\ child}$
   - $G_R = \sum{residuals\ in\ right\ child}$ 
   - $H_L = numbers\ residuals\ in\ left\ child$
   - $H_R = numbers\ residuals\ in\ right\ child$
	
	While $\lambda$ and $\gamma$ are hyperparameters. The first is needed as regularization parameter that reduces the predicion's [sensitivity](Metrics.md#recall) to individual observations, while the second is the minimum loss reduction required to make a further partition on a leaf node of the tree.
5. The same is done for another partition setting threshold after the second sample in graph showed at step 3. The same is done for every possible threshold and the one that gave the maximum gain is selected. In our case threshold = 1000 and his tree is used. Also, for his leaves another threshold is searched.![[xgboost_threshold2.png]]![[xgboost_tree2.png]] If the gain is negative the split is bad and it would be better to leave the tree as it is. Of course, every split is tried, if the split is positive another node and leaves are created (in our case 1300 yeld negative gain, so is bad, but 1600 is good and must be chosen). Every split is tried until a leaf contain a single value or the split would yeld negative gain.
6. Every leaf can contain only one value therefore the 2 or more residuals are combined following this formula $$output=\frac{\sum{residuals}}{\#residuals +\lambda}$$
7. A prediction is computed $$Prediction = Initial\ Prediction + Learning\ Rate*Prediction$$ where the prediction on the right of the formula is the output of the tree (the residual inside the leaf selected), this yelds new residuals: (-16.500, -10.500, -2.375, 3.125, 15.000)
8. Steps 3 to 7 are repeated until is reached the number of estimators.
9. The prediction is computed as $$Prediction = Initial\ Prediction+Learning\ Rate*Prediction_1+Learning\ Rate*Prediction_2 + ....$$
### Code
XGBoost isn't included in Scikit-Learn package.

``` python
import pandas as pd  
import xgboost as xgb  
from sklearn.datasets import load_boston  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import mean_squared_error

boston = load_boston()  
X = pd.DataFrame(boston.data, columns=boston.feature_names)  
y = pd.Series(boston.target)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# XGBRegressor is initialized, lambda and gamma are selected, also number of estimators
regressor = xgb.XGBRegressor(  
	n_estimators=100,  
	reg_lambda=1,  
	gamma=0,  
	max_depth=3  
)

regressor.fit(X_train, y_train)

# It is also possible to check the importance gave to each features
pd.DataFrame(regressor.feature_importances_.reshape(1, -1), columns=boston.feature_names)

# The model is used to make predictions
y_pred = regressor.predict(X_test)
mean_squared_error(y_test, y_pred)
```

*Study how it gives feature importance*