# Correlation
## Pearson's Correlation
It measures the **linear** association between 2 variables. Please note that zero correlation does not mean zero association (there can be some non-linear relationship).

- +1 complete positive correlation
- 0 no linear correlation
- -1 complete negative correlation

$$
Pearson = \frac{\sigma_{xy}}{\sigma_x\sigma_y} = \frac{\sum_i{(x_i - \bar{x}})(y_i - \bar{y})}{\sqrt{\sum_j{(x_j - \bar{x})^2} \sum_k{(y_k - \bar{y})^2}}}
$$
- $\sigma_{xy}$ = Covariance, how much those variables changes togheter, if their dependent on each other
- $\sigma_{x}$ = Standard deviation or $\sqrt{\sigma_{x}^2}$ , which is variance, it gives indication of how much sparse is a variable (distant from its mean)

A simple Python implementation:

``` python
Score1 = [15,12,8,8,7,7,7,6,5,3] 
Score2 = [10,25,17,11,13,17,20,13,9,15]

# Mean of the series
m1 = (sum(Score1)/len(Score1))
m2 = (sum(Score2)/len(Score1))

# Covariance
Score_1 = [x-m1 for x in Score1]
Score_2 = [x-m2 for x in Score2]

n1 = sum([x*y for x,y in zip(Score_1,Score_2)])

# Standard deviations
Score1 = sum([(x-m1)**2 for x in Score1])
Score2 = sum([(x-m2)**2 for x in Score2])

n2 = (Score1**(1/2))*(Score2**(1/2))

# Pearson Correlation
r = n1/n2

print (round(r,3))
```

Also, note that the slope of the regression line is 
$$
Slope = \frac{\sigma_{xy}}{\sigma_x^2} = \frac{COV(X,Y)}{VAR(X)} = \frac{\sum_i{(x_i - \bar{x}})(y_i - \bar{y})}{\sum_j{(x_j - \bar{x})^2}}
$$
Studying expected values:
https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/more-on-regression/v/covariance-and-the-regression-line

## Covariance
$$S_{xy} = \frac{\sum_i{(x_i - \bar{x}})(y_i - \bar{y})}{n-1}$$
Sample covariance, if you have the whole population it is just divided N (and it is written as $\sigma_{xy}$). Of corse the more the samples the more the formulas converge.

It represents how much two variables changes togheter. Also note the number given by the formula means nothing, it's the sign which is important.

The expcted value in many cases is the mean, or in continuous distribution is probability weighted sum.

## Regression Line

A linear regression lets you use one variable to predict another variable’s value. The formula is $y = mx + b$ 
Once you have the slope, you can easily calculate b as $b = \bar{y} - slope*\bar{x}$, then if you want to know the most probable value of y given an x you just apply this equation with m and b known.

Also, note that in a normal Ordinary Least Squared regression you should remove the mean, this is a standardization and it is important because otherwise regression using x as fixed variable would be different from regression using y as fixed variable.

Pearson(X,Y) is equal to y,x thanks to commutativity.

If you want to get the slope of the line for x as indipendent you do: $\frac{COV(X,Y)}{VAR(X)}$, if you want to use y as indipendent $\frac{COV(Y, X)}{VAR(Y)}$
Of course, if variables are standardized var(y)=var(x)

Lastly, if someone asks you to compute the most probable value of x given y, this time you consider the line which has y as indipendent variable, compute slope and intercept, then you get x. (x regressed on y)