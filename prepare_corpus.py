"""
Script de préparation du corpus médical
Étape 1 : Construction et nettoyage du corpus
"""

import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from tqdm import tqdm

# Configuration
DATA_DIR = "BD quest_resp medecine"
OUTPUT_FILE = "docs_medical.csv"
TARGET_SIZE = (500, 2000)  # Taille cible du corpus

# Fichiers médicaux à utiliser (exclure les fichiers non-médicaux)
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

def clean_text(text):
    """
    Nettoyage avancé du texte
    """
    if pd.isna(text):
        return ""
    
    # Convertir en string
    text = str(text)
    
    # Supprimer les caractères spéciaux et multiples espaces
    text = re.sub(r'\s+', ' ', text)
    
    # Supprimer les URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Supprimer les emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Supprimer les caractères de contrôle
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Supprimer les multiples points, virgules, etc.
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r',{2,}', ',', text)
    
    # Trim
    text = text.strip()
    
    return text

def load_medical_data():
    """
    Charger et combiner tous les fichiers médicaux
    """
    all_data = []
    
    print("📁 Chargement des fichiers médicaux...")
    
    for filename in tqdm(MEDICAL_FILES):
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️  Fichier non trouvé: {filename}")
            continue
        
        try:
            # Lire le CSV
            df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip')
            
            # Normaliser les noms de colonnes
            df.columns = df.columns.str.strip()
            
            # Détecter les colonnes question/answer (plusieurs formats possibles)
            question_cols = ['Question', 'question', 'query', 'Query', 'text', 'Text']
            answer_cols = ['Answer', 'answer', 'response', 'Response', 'content', 'Content']
            
            question_col = None
            answer_col = None
            
            for col in question_cols:
                if col in df.columns:
                    question_col = col
                    break
            
            for col in answer_cols:
                if col in df.columns:
                    answer_col = col
                    break
            
            if question_col is None or answer_col is None:
                # Essayer de détecter automatiquement
                if len(df.columns) >= 2:
                    question_col = df.columns[0]
                    answer_col = df.columns[1]
                else:
                    print(f"⚠️  Colonnes non détectées dans {filename}")
                    continue
            
            # Créer un DataFrame standardisé
            standardized = pd.DataFrame({
                'Question': df[question_col],
                'Answer': df[answer_col],
                'source_file': filename.replace('.csv', '')
            })
            
            all_data.append(standardized)
            print(f"✅ {filename}: {len(standardized)} documents chargés")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement de {filename}: {str(e)}")
    
    # Combiner tous les DataFrames
    if not all_data:
        raise ValueError("Aucune donnée médicale chargée !")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total documents bruts: {len(combined_df)}")
    
    return combined_df

def clean_and_filter_corpus(df):
    """
    Nettoyer et filtrer le corpus
    """
    print("\n🧹 Nettoyage du corpus...")
    
    # Appliquer le nettoyage
    df['Question'] = df['Question'].apply(clean_text)
    df['Answer'] = df['Answer'].apply(clean_text)
    
    # Filtrer les entrées vides
    initial_size = len(df)
    df = df[(df['Question'] != '') & (df['Answer'] != '')]
    print(f"   Suppression des vides: {initial_size} → {len(df)}")
    
    # Supprimer les doublons exacts
    initial_size = len(df)
    df = df.drop_duplicates(subset=['Question', 'Answer'])
    print(f"   Suppression des doublons: {initial_size} → {len(df)}")
    
    # Filtrer les textes trop courts (moins de 20 caractères)
    initial_size = len(df)
    df = df[(df['Question'].str.len() >= 20) & (df['Answer'].str.len() >= 50)]
    print(f"   Filtrage textes courts: {initial_size} → {len(df)}")
    
    # Filtrer les textes trop longs (plus de 2000 caractères pour la réponse)
    initial_size = len(df)
    df = df[df['Answer'].str.len() <= 2000]
    print(f"   Filtrage textes longs: {initial_size} → {len(df)}")
    
    # Réinitialiser l'index
    df = df.reset_index(drop=True)
    
    return df

def balance_corpus(df, target_size=1500):
    """
    Équilibrer le corpus pour avoir une distribution raisonnable par source
    """
    print(f"\n⚖️  Équilibrage du corpus (cible: {target_size} documents)...")
    
    # Statistiques par source
    source_counts = df['source_file'].value_counts()
    print("\n📊 Distribution par source:")
    print(source_counts)
    
    # Si on a déjà la bonne taille, on échantillonne
    if len(df) > target_size:
        # Échantillonnage stratifié pour garder la diversité
        samples_per_source = {}
        total_sources = len(source_counts)
        
        for source in source_counts.index:
            # Proportionnel mais avec un minimum
            proportion = source_counts[source] / len(df)
            samples = max(int(target_size * proportion), 20)
            samples_per_source[source] = min(samples, source_counts[source])
        
        # Ajuster pour atteindre exactement target_size
        while sum(samples_per_source.values()) > target_size:
            # Réduire la source avec le plus de documents
            max_source = max(samples_per_source, key=samples_per_source.get)
            samples_per_source[max_source] -= 1
        
        # Échantillonner
        sampled_dfs = []
        for source, n_samples in samples_per_source.items():
            source_df = df[df['source_file'] == source]
            sampled = source_df.sample(n=n_samples, random_state=42)
            sampled_dfs.append(sampled)
        
        df = pd.concat(sampled_dfs, ignore_index=True)
        print(f"✅ Échantillonnage: {len(df)} documents")
    
    elif len(df) < TARGET_SIZE[0]:
        print(f"⚠️  Attention: corpus petit ({len(df)} < {TARGET_SIZE[0]})")
    
    return df

def add_metadata(df):
    """
    Ajouter des métadonnées utiles
    """
    print("\n🏷️  Ajout des métadonnées...")
    
    # ID unique pour chaque document
    df['doc_id'] = range(len(df))
    
    # Longueurs
    df['question_length'] = df['Question'].str.len()
    df['answer_length'] = df['Answer'].str.len()
    
    # Catégorie médicale (basée sur le fichier source)
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
    
    df['medical_category'] = df['source_file'].apply(
        lambda x: next((v for k, v in category_map.items() if k in x), 'General')
    )
    
    # Complexité (basée sur la longueur de la réponse)
    df['complexity'] = pd.cut(
        df['answer_length'],
        bins=[0, 200, 500, 1000, float('inf')],
        labels=['Simple', 'Moderate', 'Complex', 'Very Complex']
    )
    
    return df

def save_corpus(df, output_file):
    """
    Sauvegarder le corpus nettoyé
    """
    print(f"\n💾 Sauvegarde du corpus dans {output_file}...")
    
    # Réorganiser les colonnes
    column_order = [
        'doc_id', 'Question', 'Answer', 'source_file', 
        'medical_category', 'complexity',
        'question_length', 'answer_length'
    ]
    
    df = df[column_order]
    
    # Sauvegarder
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✅ Corpus sauvegardé: {len(df)} documents")
    print(f"\n📊 Statistiques finales:")
    print(f"   - Nombre total de documents: {len(df)}")
    print(f"   - Sources différentes: {df['source_file'].nunique()}")
    print(f"   - Catégories médicales: {df['medical_category'].nunique()}")
    print(f"\n📈 Distribution par catégorie:")
    print(df['medical_category'].value_counts())
    print(f"\n📊 Distribution par complexité:")
    print(df['complexity'].value_counts())
    
    return df

def main():
    """
    Pipeline complet de préparation du corpus
    """
    print("=" * 60)
    print("🏥 PRÉPARATION DU CORPUS MÉDICAL")
    print("=" * 60)
    
    # Étape 1: Charger les données
    df = load_medical_data()
    
    # Étape 2: Nettoyer et filtrer
    df = clean_and_filter_corpus(df)
    
    # Étape 3: Équilibrer le corpus
    df = balance_corpus(df, target_size=1500)
    
    # Étape 4: Ajouter les métadonnées
    df = add_metadata(df)
    
    # Étape 5: Sauvegarder
    final_df = save_corpus(df, OUTPUT_FILE)
    
    print("\n" + "=" * 60)
    print("✅ PRÉPARATION TERMINÉE AVEC SUCCÈS !")
    print("=" * 60)
    
    return final_df

if __name__ == "__main__":
    main()
