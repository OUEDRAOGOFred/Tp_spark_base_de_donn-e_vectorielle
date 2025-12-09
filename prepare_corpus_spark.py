"""
Préparation du corpus médical avec Apache Spark
Étape 1 : Construction et nettoyage du corpus (VERSION SPARK)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, length, regexp_replace, trim, lower, udf, monotonically_increasing_id,
    when, lit
)
from pyspark.sql.types import StringType, IntegerType
import pandas as pd
import os
import re
from tqdm import tqdm

# Configuration
DATA_DIR = "BD quest_resp medecine"
OUTPUT_FILE = "docs_medical.csv"
TARGET_SIZE = (500, 2000)

# Fichiers médicaux
MEDICAL_FILES = [
    'CancerQA.csv',
    'Diabetes_and_Digestive_and_Kidney_DiseasesQA.csv',
    'Disease_Control_and_PreventionQA.csv',
    'Genetic_and_Rare_DiseasesQA.csv',
    'growth_hormone_receptorQA.csv',
    'Heart_Lung_and_BloodQA.csv',
    'MedicalQuestionAnswering.csv',
    'Neurological_Disorders_and_StrokeQA.csv',
    'OtherQA.csv',
    'SeniorHealthQA.csv'
]

def create_spark_session():
    """
    Créer une session Spark avec configuration optimisée
    """
    print("🔧 Création de la session Spark...")
    
    spark = SparkSession.builder \
        .appName("MedicalCorpusPreparation") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"✅ Spark Session créée: {spark.version}")
    return spark

def clean_text_spark(text):
    """
    Fonction de nettoyage du texte pour UDF Spark
    """
    if text is None:
        return ""
    
    # Supprimer URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Supprimer emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Supprimer caractères de contrôle
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normaliser espaces
    text = re.sub(r'\s+', ' ', text)
    
    # Normaliser ponctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r',{2,}', ',', text)
    
    return text.strip()

def categorize_source(source_file):
    """
    Catégoriser la source médicale
    """
    category_map = {
        'Cancer': 'Oncology',
        'Diabetes': 'Endocrinology',
        'Disease_Control': 'Public Health',
        'Genetic': 'Genetics',
        'growth_hormone': 'Endocrinology',
        'Heart_Lung': 'Cardiology',
        'Medical': 'General Medicine',
        'Neurological': 'Neurology',
        'Other': 'General',
        'SeniorHealth': 'Geriatrics'
    }
    
    for key, value in category_map.items():
        if key in source_file:
            return value
    return 'General'

def load_medical_data_spark(spark):
    """
    Charger tous les fichiers médicaux avec Spark
    """
    print("\n📁 Chargement des fichiers médicaux avec Spark...")
    
    all_dataframes = []
    
    for filename in tqdm(MEDICAL_FILES, desc="Chargement"):
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️  Fichier non trouvé: {filename}")
            continue
        
        try:
            # Lire avec Spark
            df = spark.read.csv(
                filepath,
                header=True,
                inferSchema=True,
                mode="DROPMALFORMED",
                encoding="UTF-8"
            )
            
            # Détecter les colonnes
            columns = df.columns
            question_col = None
            answer_col = None
            
            # Chercher les colonnes question/answer
            for col_name in columns:
                col_lower = col_name.lower()
                if 'question' in col_lower or 'query' in col_lower or 'text' in col_lower:
                    if question_col is None:
                        question_col = col_name
                if 'answer' in col_lower or 'response' in col_lower or 'content' in col_lower:
                    if answer_col is None:
                        answer_col = col_name
            
            # Si non détecté, prendre les deux premières colonnes
            if question_col is None or answer_col is None:
                if len(columns) >= 2:
                    question_col = columns[0]
                    answer_col = columns[1]
                else:
                    print(f"⚠️  Colonnes insuffisantes dans {filename}")
                    continue
            
            # Renommer et ajouter source
            df = df.select(
                col(question_col).alias("Question"),
                col(answer_col).alias("Answer")
            ).withColumn("source_file", lit(filename.replace('.csv', '')))
            
            all_dataframes.append(df)
            print(f"✅ {filename}: {df.count()} documents")
            
        except Exception as e:
            print(f"❌ Erreur: {filename}: {str(e)}")
    
    if not all_dataframes:
        raise ValueError("Aucune donnée chargée !")
    
    # Union de tous les DataFrames
    combined_df = all_dataframes[0]
    for df in all_dataframes[1:]:
        combined_df = combined_df.union(df)
    
    print(f"\n📊 Total documents bruts: {combined_df.count()}")
    
    return combined_df

def clean_corpus_spark(spark, df):
    """
    Nettoyer le corpus avec Spark
    """
    print("\n🧹 Nettoyage du corpus avec Spark...")
    
    initial_count = df.count()
    print(f"   Départ: {initial_count} documents")
    
    # Créer UDF pour le nettoyage
    clean_text_udf = udf(clean_text_spark, StringType())
    
    # Appliquer le nettoyage
    df = df.withColumn("Question", clean_text_udf(col("Question"))) \
           .withColumn("Answer", clean_text_udf(col("Answer")))
    
    # Filtrer les vides
    df = df.filter(
        (col("Question") != "") & 
        (col("Answer") != "") &
        (col("Question").isNotNull()) &
        (col("Answer").isNotNull())
    )
    
    print(f"   Après filtrage vides: {df.count()} documents")
    
    # Supprimer doublons
    df = df.dropDuplicates(["Question", "Answer"])
    print(f"   Après suppression doublons: {df.count()} documents")
    
    # Ajouter longueurs
    df = df.withColumn("question_length", length(col("Question"))) \
           .withColumn("answer_length", length(col("Answer")))
    
    # Filtrer par longueur
    df = df.filter(
        (col("question_length") >= 20) &
        (col("answer_length") >= 50) &
        (col("answer_length") <= 2000)
    )
    
    print(f"   Après filtrage longueur: {df.count()} documents")
    
    return df

def add_metadata_spark(spark, df):
    """
    Ajouter métadonnées avec Spark
    """
    print("\n🏷️  Ajout des métadonnées avec Spark...")
    
    # Ajouter ID
    df = df.withColumn("doc_id", monotonically_increasing_id())
    
    # Catégoriser source
    categorize_udf = udf(categorize_source, StringType())
    df = df.withColumn("medical_category", categorize_udf(col("source_file")))
    
    # Complexité basée sur longueur
    df = df.withColumn(
        "complexity",
        when(col("answer_length") < 200, "Simple")
        .when(col("answer_length") < 500, "Moderate")
        .when(col("answer_length") < 1000, "Complex")
        .otherwise("Very Complex")
    )
    
    return df

def balance_and_save(spark, df, target_size=1500):
    """
    Équilibrer et sauvegarder avec Spark
    """
    print(f"\n⚖️  Équilibrage du corpus (cible: {target_size})...")
    
    total_count = df.count()
    
    if total_count > target_size:
        # Échantillonnage stratifié
        fraction = target_size / total_count
        df = df.sampleBy("source_file", fractions={
            source: fraction 
            for source in df.select("source_file").distinct().rdd.flatMap(lambda x: x).collect()
        }, seed=42)
        
        # Ajuster pour avoir exactement target_size
        current_count = df.count()
        if current_count > target_size:
            df = df.limit(target_size)
        
        print(f"   Échantillonné à: {df.count()} documents")
    
    # Réindexer les IDs
    df = df.drop("doc_id")
    df = df.withColumn("doc_id", monotonically_increasing_id())
    
    # Statistiques
    print(f"\n📊 Statistiques finales:")
    print(f"   Documents: {df.count()}")
    print(f"   Sources: {df.select('source_file').distinct().count()}")
    print(f"   Catégories: {df.select('medical_category').distinct().count()}")
    
    print("\n📈 Distribution par catégorie:")
    df.groupBy("medical_category").count().orderBy("count", ascending=False).show()
    
    print("\n📊 Distribution par complexité:")
    df.groupBy("complexity").count().orderBy("count", ascending=False).show()
    
    # Convertir en Pandas et sauvegarder
    print(f"\n💾 Sauvegarde dans {OUTPUT_FILE}...")
    
    # Réorganiser colonnes
    df = df.select(
        "doc_id", "Question", "Answer", "source_file",
        "medical_category", "complexity",
        "question_length", "answer_length"
    )
    
    # Convertir en Pandas pour sauvegarde CSV
    pandas_df = df.toPandas()
    pandas_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    
    print(f"✅ Corpus sauvegardé: {len(pandas_df)} documents")
    
    return df

def main():
    """
    Pipeline complet avec Apache Spark
    """
    print("=" * 60)
    print("🏥 PRÉPARATION DU CORPUS MÉDICAL AVEC APACHE SPARK")
    print("=" * 60)
    
    # Créer session Spark
    spark = create_spark_session()
    
    try:
        # Étape 1: Charger les données
        df = load_medical_data_spark(spark)
        
        # Étape 2: Nettoyer
        df = clean_corpus_spark(spark, df)
        
        # Étape 3: Ajouter métadonnées
        df = add_metadata_spark(spark, df)
        
        # Étape 4: Équilibrer et sauvegarder
        final_df = balance_and_save(spark, df, target_size=1500)
        
        print("\n" + "=" * 60)
        print("✅ PRÉPARATION TERMINÉE AVEC SUCCÈS (SPARK) !")
        print("=" * 60)
        
        # Afficher quelques statistiques Spark
        print("\n📊 Statistiques Spark:")
        print(f"   Partitions: {final_df.rdd.getNumPartitions()}")
        
    finally:
        # Arrêter Spark
        print("\n🛑 Arrêt de la session Spark...")
        spark.stop()

if __name__ == "__main__":
    main()
