from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, radians, sin, cos, asin, sqrt, sum as _sum, to_timestamp
#_sum to avoid build in funtion sum everytime
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
import os
import findspark

#Additonal part because I had troubles with pyspark. The only way it worked was when I specified Local IP, used findspark and created an environment
# with python 3.10 (I use 3.12 usually)

#os.environ["SPARK_LOCAL_IP"] = "IP"
#os.environ["PYSPARK_PYTHON"] = r"C:\.....\pyspark_env\python.exe"
#os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\....\pyspark_env\python.exe"

findspark.init()

#again I had to specify local with in the session builder
spark = (
     SparkSession.builder
    .master("local[1]")
    .appName("VesselDistanceCalculator")
    .getOrCreate()
)

#standard data reading
df = spark.read.csv(r"C:\....\Desktop\aisdk-2024-05-04.csv", header=True, inferSchema=True)

#removed Na values in the collumns
df = (
     df.select("MMSI", "# Timestamp", "Latitude", "Longitude")
       .dropna(subset=["MMSI", "# Timestamp", "Latitude", "Longitude"])
       .withColumn("Latitude", col("Latitude").cast(DoubleType()))
       .withColumn("Longitude", col("Longitude").cast(DoubleType()))
)

#timestamp was converted to appropriate for spark with specification
df = df.withColumn("Timestamp", to_timestamp(col("# Timestamp"), "dd/MM/yyyy HH:mm:ss"))

#window for grouping by MMSI and order by time
windowSpec = Window.partitionBy("MMSI").orderBy("Timestamp")

#adds previous location for distance calculation
df = (
     df.withColumn("prev_lat", lag("Latitude").over(windowSpec))
       .withColumn("prev_lon", lag("Longitude").over(windowSpec))
)

#removed rows with no previous location 
df = df.filter(col("prev_lat").isNotNull() & col("prev_lon").isNotNull())

#converts to radians
df = (
     df.withColumn("lat1", radians(col("prev_lat")))
       .withColumn("lon1", radians(col("prev_lon")))
       .withColumn("lat2", radians(col("Latitude")))
       .withColumn("lon2", radians(col("Longitude")))
)

#Haversine distance calculation between two gps points
df = (
    df.withColumn("dlat", col("lat2") - col("lat1"))
      .withColumn("dlon", col("lon2") - col("lon1"))
      .withColumn(
          "a",
          sin(col("dlat") / 2) ** 2 +
          cos(col("lat1")) * cos(col("lat2")) * sin(col("dlon") / 2) ** 2
      )
      .withColumn("c", 2 * asin(sqrt(col("a"))))
      .withColumn("segment_km", 6371 * col("c"))  # Earth radius in km
)

#filtering for high distance jumps
df = df.filter(col("segment_km") <= 50)

#sum distance by MMSI
distance_by_mmsi = (
   df.groupBy("MMSI") \
    .agg(_sum("segment_km").alias("total_distance_km"))
)

#order distances to show top 1
longest_vessel = distance_by_mmsi.orderBy(col("total_distance_km").desc()).limit(1)

#show the results
longest_vessel.show(truncate=False)

#shut down spark session (so there are no "zombie" session in the background)
spark.stop()