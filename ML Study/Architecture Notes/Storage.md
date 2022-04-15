# Storage
## Table of Contents
* [OLTP vs OLAP](#oltp-vs-olap)
	* [OLTP](#oltp)
	* [OLAP](#olap)
* [Data Warehouse](#data-warehouse)
* [Data Lakehouse](#data-lakehouse)
	* [Time Travel](#time-travel)

## OLTP vs OLAP
- [OLTP vs OLAP. Let’s say you decide to build a… | by Cory Maklin | Feb, 2022 | Medium](https://medium.com/@corymaklin/oltp-vs-olap-90dce52d4fa0)
- [martinfowler.com](https://martinfowler.com/)

In order to perform statistics over your application items and not put pressure on source systems the correct way is to store data in a database. It is possible to perform operational updates of the application's state and create reports, however since the 2 tasks are different it is safe to create different schemas and data access patterns --> reporting database and operational database. 
Using different databases does not put load over the one used for operational tasks, while using different schemas allow to perform certain operation faster saving analysts time.

The operational DB is an OLTP system, the reporting one is an OLAP system.

### OLTP
Online Transaction Processing is characterized by a large number of ACID compliant CRUD (Create, Read, Update, Delete) transactions.
Its performances are measured as number of transactions per second and they store application relevant information (everything your app need to know in order to work).

It is a row oriented DB, therefore everycolumn is added consequentialy (thus the head of the hard disk can write everything without being lifted, *la testina scrive senza sollevarsi cosa che accaderebbe in un db column oriented*).

### OLAP
Online Analytical Processing is characterized by a low volume of read only transactions.
Its performances are measured as query response times.

The data inside OLAP is derived from OLTP database at predefined intervals.

They stores data using star schemas (often used in data mart, it is a table wich references various tables) in order to use less joins as possibles to analyze the data.

Columnar databases are well suited for OLAP systems.
I.e. determining the total quantity of products sold in one month, in this case the needle of the hard drive would have just to search the month and then scan every data since they are stored consequently.



## Data Warehouse
- [Data Warehouses. The term Data Warehouse was first… | by Cory Maklin | Feb, 2022 | Medium](https://medium.com/@corymaklin/data-warehouses-98c9d3788159)

Coined in 1970, it is adatabase management system (DBMS) the stores all enterprise's data. It is a single source of truth for all business related query.

A reporting db is needed to accomodate different types of data access pattern. The goal is to avoid undue stress on operational systems creating a new db with its own schema.
The problem is that each department team will create its tables (maybe all tables regarding clients, but focusing on different aspects and sampled from operational systems in different time slots, different hours, therefore customers can have multiple tables everyone with a slightly different information) and intermediate tables with different values for what it could be the same feature.
What is the right version of te data to use? This will result in poor decisions.

Data warehouses solve the problem of data integrity. It stores the entire enterprise's data in one central location, therefore you don't have multiple customer tables for each department, but just one.

However, data warehouses aren't suite for unstructured data because they are relational database. The users therefore can't use machine learning to analyse images, audio, textual data...
Also, they have problems managing the amount and velocity of data created nowadays (does not scale). It's expensive, vendors charge the amount of data stored at rest, it is far more expensive then cloud storage.

Therefore datalake were introduced as landing point. Once arrived on the data lake only the structured data would go to data warehouse via etl pipelines.

Remember: it is important that each team does not create its own reporting database otherwise you will have inconsistencies. Data warehouses provides data marts in the form of star schemas.

## Data Lakehouse
- [Data Lakehouses. In the previous article, we discussed… | by Cory Maklin | Feb, 2022 | Medium](https://medium.com/@corymaklin/data-lakehouses-abc791e08300)

![[data_lakehouse.png]]
In order to overcome some of the problems about data warehouses you could write data into data lakes and then send them to data warehouses. DWH would support bi, reports and data science, data lake would support data science and machine learning tasks.
Also data scientist could use every kind of data (also unstructured ones) while analysts could continue to use SQL.

The problem is that continue to extract data from DL to DWH requires a lot of effort, every new ETL step could introduce bugs and errors, therefore it was hard mantaining consistent DWH and DL.

The solution is to use a DWH which uses a DL as the underlying file system. It does not store data into a traditional DBMS, it uses something like **Apache Iceberg & Delta Lake** (or Apache Hudi?) to create tables on top of raw data which can then be queried using **Spark or Trino**.

Spark can read&write directly in Parquet over an S3 storage using SQL, however, its **transaction are not ACID** compliant. Therefore, failures could create files half modified, duplicates and other inconsistencies. This is where Delta Lake can help using log in DBMS while Apache Iceberg uses metadata.

### Data Lakehouse Time Travel
- [Date Lakehouse Time Travel. It’s Tuesday afternoon, you’re sitting… | by Cory Maklin | Feb, 2022 | Medium](https://medium.com/@corymaklin/date-warehouse-time-travel-8c0d527c9f58)
- [Data Science DC Nov 2021 Meetup: Apache Iceberg - An Architectural Look Under the Covers - YouTube](https://www.youtube.com/watch?v=N4gAi_zpN88)
- https://iceberg.apache.org/blogs/
- [Docker, Spark, and Iceberg: The Fastest Way to Try Iceberg! (tabular.io)](https://tabular.io/blog/docker-spark-and-iceberg/)

If a Spark job fails while running on a table, or is mistekenly run on a table and a "reset" is needed it is possible to leverage on **Apache Iceberg**.
An open source poject created by Netflix.

It is similar to Hadoop Hbase, which provides Bigtable-like capabilities on top of HDFS; here Iceberg provides Bigtable-like capabilities on top of any distributed file system (S3, GCS, ABS, HDFS, ...).

It provides atomic commits, concurrent writes, schema evolution, hidden partitioning and time travel.
It is supported by Spark, Flink Dremio, Amazon Athena, Amazon EMR and Trino.

When a query against a table is done, the engine calls Iceberg API which then read from/write to metadata files that are used to track the data files.
Whenever data in an Iceberg table is overwritten, it actually keeps the old version of the data and uses the metadata files to give the appearance that it's no longer there. This process enables Iceberg to "travel back in time" (a rollback to a previous snapshot).

*The first link has a little tutorial on how to run apache iceberg locally and see how you can use different snapshot in your notebooks.
You can use it togheter with 4th link.*
