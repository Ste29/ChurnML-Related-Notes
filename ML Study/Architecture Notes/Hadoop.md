# Hadoop

In order to manage big data a lot of compute power is needed, there are 2 options:
- Vertical scaling (more ram, hdds, cores)
- Horizontal scaling (distributing computing)

Processor speeds, however, can't go over 4GHz in the near future, also, the cost for better components doesn't scale linearly. Therefore horizontal scaling is the cheaper approach.

Hadoop is divided in:
- HDFS
- YARN
- MapReduce

## HDFS (Hadoop Distributed File System)
Traditionally data storage and computations are separated, however moving a lot of data over the network is expensive.
Therefore processing data where it is stored could be a better solution.

Moving the code is easier than petabytes of data.

However, how data stored on different nodes can be accessed? HDFS is implemented as a master slave architecture, it consists of a NameNode (master) and one or more DataNodes (slaves).
The NameNode tells clients which node to send data to or which nodes contains certain data.

The client then connect to the DataNode and begin transferring data without any further NameNode involvement.

By default, HDFS stores three copies of the files in the cluster in order to manage failures. The files are read-only, the files always will have that content. It is only possible to create files and delete files, therefore synchronized changes are not necessary.


## Yarn
Yet Another Resource Negotiator, it manages compute resources. It spreads out the workloads over the cluster, tells each computer what it should run and how many resources use.

## MapReduce

Map apply some function to every element in the list and return the result as a list. The order of map functions doesn't matter.

If the list is very long the action would take some time.

[Apache Hadoop — What Is YARN | HDFS | MapReduce | by Cory Maklin | Towards Data Science](https://towardsdatascience.com/big-data-what-is-apache-hadoop-3dafda16c98e)

[ARIMA Model Python Example — Time Series Forecasting | by Cory Maklin | Towards Data Science](https://towardsdatascience.com/machine-learning-part-19-time-series-and-autoregressive-integrated-moving-average-model-arima-c1005347b0d7)

[Linear Regression In Python. An example of how to implement linear… | by Cory Maklin | Towards Data Science](https://towardsdatascience.com/linear-regression-in-python-64679ab58fc7)

[Singular Value Decomposition Example In Python | by Cory Maklin | Towards Data Science](https://to

[Cory Maklin – Medium](https://medium.com/@corymaklin)wardsdatascience.com/singular-value-decomposition-example-in-python-dab2507d85a0)

[Credit Card Fraud Detection (Machine Learning Models using PySpark) | by Shubham Sharma | Medium](https://medium.com/@shsharma14/credit-card-fraud-detection-ml-using-pyspark-8bb32394cf7b)