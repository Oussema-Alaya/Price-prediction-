import kagglehub
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Étape 1 : Téléchargement et vérification des données
# Téléchargement des données avec kagglehub
path = kagglehub.dataset_download("ghassen1302/property-prices-in-tunisia")
print("Chemin vers les fichiers du dataset :", path)

# Vérifiez le contenu du dossier pour trouver le fichier CSV
print("Contenu du dossier :", os.listdir(path))

# Nom du fichier CSV
csv_path = f"{path}/Property Prices in Tunisia.csv"  # Nom exact du fichier

# Étape 2 : Initialisation de Spark
spark = SparkSession.builder \
    .appName("Property Prices Analysis") \
    .getOrCreate()

# Chargement des données
try:
    data = spark.read.csv(csv_path, header=True, inferSchema=True)
    print("Dataset chargé avec succès")
except Exception as e:
    print("Erreur lors du chargement du dataset :", e)
    raise

# Vérifiez la structure des données
data.show(5)
data.printSchema()

# Étape 3 : Analyse exploratoire
# Statistiques descriptives
data.describe().show()

# Vérification des valeurs nulles
print("Valeurs nulles par colonne :")
data.select([col(c).isNull().alias(c) for c in data.columns]).groupby().sum().show()

# Analyse des catégories uniques
data.select("category", "type", "city", "region").distinct().show()

# Étape 4 : Préparation des données pour le Machine Learning
# Encodage des colonnes catégoriques
indexers = [
    StringIndexer(inputCol=col, outputCol=f"{col}_index").fit(data)
    for col in ["category", "type", "city", "region"]
]

# Combinaison des colonnes numériques et encodées
feature_columns = ["room_count", "bathroom_count", "size",
                   "category_index", "type_index", "city_index", "region_index"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Pipeline de transformation
pipeline = Pipeline(stages=indexers + [assembler])

# Préparation des données cibles
data = data.withColumn("price", col("price").cast("double"))

# Appliquer le pipeline
data_prepared = pipeline.fit(data).transform(data)
data_prepared.select("features", "price").show()

# Étape 5 : Création et évaluation du modèle
# Séparation des données en ensemble d'entraînement et de test
train_data, test_data = data_prepared.randomSplit([0.8, 0.2], seed=42)

# Création du modèle de régression linéaire
lr = LinearRegression(featuresCol="features", labelCol="price")
lr_model = lr.fit(train_data)

# Évaluation sur les données de test
predictions = lr_model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE) : {rmse}")

predictions.select("features", "price", "prediction").show()

# Étape 6 : Fonction de prédiction pour une nouvelle propriété
def predict_price(room_count, bathroom_count, size, category, type_, city, region):
    from pyspark.sql import Row

    # Convertir les inputs en DataFrame Spark
    input_data = spark.createDataFrame([Row(
        room_count=room_count,
        bathroom_count=bathroom_count,
        size=size,
        category=category,
        type=type_,
        city=city,
        region=region
    )])

    # Appliquer le pipeline et prédire
    input_prepared = pipeline.fit(data).transform(input_data)
    prediction = lr_model.transform(input_prepared)
    return prediction.select("prediction").collect()[0][0]

# Étape 7 : Interface utilisateur
while True:
    print("=== Prédiction de prix ===")
    room_count = int(input("Nombre de chambres : "))
    bathroom_count = int(input("Nombre de salles de bain : "))
    size = float(input("Taille (m²) : "))
    category = input("Catégorie (e.g., appartement) : ")
    type_ = input("Type (e.g., vente) : ")
    city = input("Ville : ")
    region = input("Région : ")

    predicted_price = predict_price(room_count, bathroom_count, size, category, type_, city, region)
    print(f"Prix prédit : {predicted_price:.2f} TND")
