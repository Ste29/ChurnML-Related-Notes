# Metrics
## Table of contents
* [Classification](#classification)
	* [Accuracy](#accuracy)
	* [Precision](#precision)
	* [Recall](#recall)
		* [Precision-Recall Curve](#precision-recall-curve)
	* [F1 Score](#f1-score)
* [Regression](#clustering)
* [Clustering](#clustering)

![[metrics.png]]
*List of metrics from sklearn*


Lift???


## Classification
### Accuracy
It is the number of correctly classified vs total samples. In binary classification can be summarized as:

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$
However, when the dataset is highly imbalanced the results can be misleading.
I.e. In a situation with 95 negatives and 5 positives predicting all negatives would give 95% accuracy.
In these cases it would be better to use **balanced accuracy** which normalize true positives and true negatives predictions by the number of positives and negative samples.
$$
BalancedAccuracy = \frac{recall+specificity}{2}
$$
In the specified example bACC would be 0.5 (the maximum is 1), which is equivalent to a random classifier.
This metric can be used wether or not the dataset is balanced, the only important thing is that positive and negative values have the same importance.

### Precision
![[Precisionrecall.svg.png]]
Precision is about how precise/accurate your model is. Among the predicted positives, how many are actual positives?
$$
Precision = \frac{TP}{TP + FP} = \frac{TP}{Predicted Positive}
$$
When to use it:
- If the cost of false positive is high, this is the right metric

i.e. in email detection it is really important to not classify a real email as spam (false positive! true positives are spam)

### False Positive Rate
It is the probability of false alarm.
$$FPR = \frac{FP}{N} = \frac{FP}{FP+TN} = 1 - specificity = 1 - \frac{TN}{N}$$

### Recall
Also called sensitivity or true positive rate. It is the probability of detection.
Allows you to measure how many positives you spot among **all** the existing positives
$$
Recall = \frac{TP}{TP + FN} = \frac{TP}{P}
$$
When to use it:
- If the cost associated to a false negative is high

i.e. if you miss a fraudolent transaction (Actual positive --> False negative) the consequences can be harsh

The opposite of recall is called **specificity** or true negative rate.

#### ROC
![[Roc_curve.svg.png]]
Receiver operating characteristic, can be used to illustrate the diaggnostic ability of a binary classifier.
Area Under Curve (AUC) come from curve True Positive Rate (recall)/ False Positive Rate (fall-out)
AUC can take values between 0 and 1, where for a random classifier it takes the value 0.5 . The larger the value, the better.
The area between the ROC curve and the no-discrimination line multiplied by two is called the _Gini coefficient_. It should not be confused with the [measure of statistical dispersion also called Gini coefficient](https://en.wikipedia.org/wiki/Gini_coefficient "Gini coefficient").


#### Precision-Recall Curve
![[Precision-Recall-curve-vs-ROC.png]]
This curve is the more informative on imbalanced data, it is better than ROC.
It can take values between 0 and 1, for a random classifier it takes the value equal to the imbalance of the minority class (in this case 0.1-0.2). The larger the better.

### F1 Score
Also called harmonic mean of precision and recall.

$$
F = 2*\frac{precision*recall}{precision+recall}
$$
You can weight more recall with F2 metric and precision more with F0.5.
It is a solid metric in case of imbalanced data.