# RayDB: Accelerating Databases with Ray Tracing Cores

---

### Requirements

---

- NVCC. Tested on 12.2.140.
- CMake. Tested on 3.16.3.
- GCC/G++. Tested on 8.4.0.
- OptiX. Tested on 7.1. The project already includes OptiX SDK 7.1. A RTX-capable GPU (Turing architecture and later) from Nvidia. Tested on RTX 4090.

### Generate input data

---

Checkout the star schema benchmark repository and compile the data generator:

```bash
git clone https://github.com/vadimtk/ssb-dbgen.git
cd ssb-dbgen
make
```

Generate table data. Parameter **`-s`** specifies the scale factor. We take SF=20 as an example：

```bash
./dbgen -s 20 -T c
./dbgen -s 20 -T l
./dbgen -s 20 -T p
./dbgen -s 20 -T s
```

Create tables and import data into tables in a database. We take MonetDB as an example. Execute following commands in the MonetDB client:

```bash
CREATE TABLE customer
(
C_CUSTKEY       INT,
C_NAME          STRING,
C_ADDRESS       STRING,
C_CITY          STRING,
C_NATION        STRING,
C_REGION        STRING,
C_PHONE         STRING,
C_MKTSEGMENT    STRING
);

CREATE TABLE lineorder
(
LO_ORDERKEY             INT,
LO_LINENUMBER           INT,
LO_CUSTKEY              INT,
LO_PARTKEY              INT,
LO_SUPPKEY              INT,
LO_ORDERDATE            DATE,
LO_ORDERPRIORITY        STRING,
LO_SHIPPRIORITY         INT,
LO_QUANTITY             INT,
LO_EXTENDEDPRICE        INT,
LO_ORDTOTALPRICE        INT,
LO_DISCOUNT             INT,
LO_REVENUE              INT,
LO_SUPPLYCOST           INT,
LO_TAX                  INT,
LO_COMMITDATE           DATE,
LO_SHIPMODE             STRING
);

CREATE TABLE part
(
P_PARTKEY       INT,
P_NAME          STRING,
P_MFGR          STRING,
P_CATEGORY      STRING,
P_BRAND         STRING,
P_COLOR         STRING,
P_TYPE          STRING,
P_SIZE          INT,
P_CONTAINER     STRING
);

CREATE TABLE supplier
(
S_SUPPKEY       INT,
S_NAME          STRING,
S_ADDRESS       STRING,
S_CITY          STRING,
S_NATION        STRING,
S_REGION        STRING,
S_PHONE         STRING
);

COPY INTO customer
FROM 'customer.tbl' ON CLIENT
USING DELIMITERS ',', E'\n', '"';

COPY INTO part
FROM 'part.tbl' ON CLIENT
USING DELIMITERS ',', E'\n', '"';

COPY INTO supplier
FROM 'supplier.tbl' ON CLIENT
USING DELIMITERS ',', E'\n', '"';

COPY INTO lineorder
FROM 'lineorder.tbl' ON CLIENT
USING DELIMITERS ',', E'\n', '"';
```

Join all tables to get a single denormalized flat table:

```bash
CREATE TABLE lineorder_flat 
AS SELECT 
l.LO_ORDERKEY AS LO_ORDERKEY,
l.LO_LINENUMBER AS LO_LINENUMBER,
l.LO_CUSTKEY AS LO_CUSTKEY,
l.LO_PARTKEY AS LO_PARTKEY,
l.LO_SUPPKEY AS LO_SUPPKEY,
l.LO_ORDERDATE AS LO_ORDERDATE,
l.LO_ORDERPRIORITY AS LO_ORDERPRIORITY,
l.LO_SHIPPRIORITY AS LO_SHIPPRIORITY,
l.LO_QUANTITY AS LO_QUANTITY,
l.LO_EXTENDEDPRICE AS LO_EXTENDEDPRICE,
l.LO_ORDTOTALPRICE AS LO_ORDTOTALPRICE,
l.LO_DISCOUNT AS LO_DISCOUNT,
l.LO_REVENUE AS LO_REVENUE,
l.LO_SUPPLYCOST AS LO_SUPPLYCOST,
l.LO_TAX AS LO_TAX,
l.LO_COMMITDATE AS LO_COMMITDATE,
l.LO_SHIPMODE AS LO_SHIPMODE,
c.C_NAME AS C_NAME,
c.C_ADDRESS AS C_ADDRESS,
c.C_CITY AS C_CITY,
c.C_NATION AS C_NATION,
c.C_REGION AS C_REGION,
c.C_PHONE AS C_PHONE,
c.C_MKTSEGMENT AS C_MKTSEGMENT,
s.S_NAME AS S_NAME,
s.S_ADDRESS AS S_ADDRESS,
s.S_CITY AS S_CITY,
s.S_NATION AS S_NATION,
s.S_REGION AS S_REGION,
s.S_PHONE AS S_PHONE,
p.P_NAME AS P_NAME,
p.P_MFGR AS P_MFGR,
p.P_CATEGORY AS P_CATEGORY,
p.P_BRAND AS P_BRAND,
p.P_COLOR AS P_COLOR,
p.P_TYPE AS P_TYPE,
p.P_SIZE AS P_SIZE,
p.P_CONTAINER AS P_CONTAINER 
FROM lineorder AS l 
INNER JOIN customer AS c ON c.C_CUSTKEY = l.LO_CUSTKEY 
INNER JOIN supplier AS s ON s.S_SUPPKEY = l.LO_SUPPKEY 
INNER JOIN part AS p ON p.P_PARTKEY = l.LO_PARTKEY;
```

Export the data needed for queries. Make sure the current path is the root of our project repository:

```bash
COPY select LO_EXTENDEDPRICE as subrevenue,LO_ORDERDATE,LO_DISCOUNT,LO_QUANTITY,LO_DISCOUNT as discount from lineorder_flat INTO 'ssb_data/q1dot1/file_q1dot1.csv' ON CLIENT USING DELIMITERS ',' , E'\n' , '"';

COPY select LO_REVENUE,"year"(LO_ORDERDATE) AS Y, P_BRAND,P_CATEGORY,S_REGION from lineorder_flat INTO 'ssb_data/q2dot1/file_q2dot1.csv' ON CLIENT USING DELIMITERS ',' , E'\n' , '"';

COPY select LO_REVENUE,"year"(LO_ORDERDATE) AS Y, C_NATION, S_NATION, C_REGION, S_REGION, LO_ORDERDATE  from lineorder_flat INTO 'ssb_data/q3dot1/file_q3dot1.csv' ON CLIENT USING DELIMITERS ',' , E'\n' , '"';

COPY select LO_REVENUE,"year"(LO_ORDERDATE) AS Y, C_CITY, S_CITY, C_NATION, S_NATION, LO_ORDERDATE  from lineorder_flat INTO 'ssb_data/q3dot2/file_q3dot2.csv' ON CLIENT USING DELIMITERS ',' , E'\n' , '"';

COPY select LO_REVENUE - LO_SUPPLYCOST, "year"(LO_ORDERDATE) AS Y, C_NATION, C_REGION, S_REGION, P_MFGR from lineorder_flat INTO 'ssb_data/q4dot1/file_q4dot1.csv' ON CLIENT USING DELIMITERS ',' , E'\n' , '"';

COPY select LO_REVENUE - LO_SUPPLYCOST, "year"(LO_ORDERDATE) AS Y,S_NATION, P_CATEGORY, C_REGION, S_REGION, LO_ORDERDATE, P_MFGR  from lineorder_flat INTO 'ssb_data/q4dot2/file_q4dot2.csv' ON CLIENT USING DELIMITERS ',' , E'\n' , '"';

COPY select LO_REVENUE - LO_SUPPLYCOST, "year"(LO_ORDERDATE) AS Y, S_CITY, P_BRAND, S_NATION, LO_ORDERDATE, P_CATEGORY  from lineorder_flat INTO 'ssb_data/q4dot3/file_q4dot3.csv' ON CLIENT USING DELIMITERS ',' , E'\n' , '"';

```

Preprocess the data. Make sure the current path is the root of our project repository:

```bash
./script/data_preprocess.sh
```

### Run experiments

---

Make sure the current path is the root of our project repository:

```bash
mkdir build
mkdir log
python script/run.py
```