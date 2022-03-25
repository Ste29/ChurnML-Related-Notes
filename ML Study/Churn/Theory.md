# Theory

## Domande:
- % Churn?
- Come hai gestito il fatto il dataset fosse fortemente sbilanciato?
- Insight generali?
- Modello scelto?
- metrica? precision recall?
- Come sono state scelte le feature?

## Links

- https://medium.com/adobetech/gaining-a-deeper-understanding-of-churn-using-data-science-workspace-18a2190e0cf3

## Understanding Churn
**L'articolo non si riferisce a Sisal**

Highly competitive markets (i.e. gaming, hotel, casino chains operate in) needs to understand customers, their behavior (CLC) and the features which lead to churn.
The goal was to understand which customer were highly unlikely to make a booking over a six month period, then sending personalized offers to those customers.
![[aep_theory.jpg]]

Over a 18 month window 95% of customers re-book within six months, therefore with only 6 months we could understand one complete purchase cycle for customers

**Our data included half a million customers and with a churn rate of 53%.**

- Useful features were:
	- Amount of plays
	- Days since last bet (compared with median bet cycle for that customer)
- Insight:
	- A customer who made frequent bookings in a one-month period was more likely to churn out. Consistent engagement of the customer over longer durations can help prevent churn. (It is not a good pattern if someone starts to play massively in a short period, consistency is the key)

The model was trained over one cycle, 6 months, using this as a benchmark it was studied if the customer would return to make a booking in the next 6 months.
Also, the data across time periods was studied to make sure there was no major seasonality.

- Evaluation parameters:
	- Recall of the model: # of churners the model is able to identify
	- Precision of the model: # of churners over the total predicted churns
	- Stability of the metrics over time