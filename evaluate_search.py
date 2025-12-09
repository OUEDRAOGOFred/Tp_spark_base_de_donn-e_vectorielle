"""
Script d'évaluation du moteur de recherche
Calcul des métriques: Recall@K, MRR@K, Precision@K, NDCG@K
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import time
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CORPUS_FILE = "docs_medical.csv"
INDEX_FILE = "medical_faiss.index"
EMBEDDINGS_FILE = "embeddings_medical.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class SearchEvaluator:
    """Évaluation du moteur de recherche"""
    
    def __init__(self):
        print("🔧 Initialisation de l'évaluateur...")
        
        # Charger les ressources
        self.corpus = pd.read_csv(CORPUS_FILE)
        self.index = faiss.read_index(INDEX_FILE)
        self.embeddings = np.load(EMBEDDINGS_FILE)
        self.model = SentenceTransformer(MODEL_NAME)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        print(f"✅ Corpus: {len(self.corpus)} documents")
        print(f"✅ Index: {self.index.ntotal} vecteurs")
    
    def create_test_queries(self, n_queries=100):
        """
        Créer des requêtes de test à partir du corpus
        """
        print(f"\n📝 Création de {n_queries} requêtes de test...")
        
        # Échantillonner aléatoirement des documents
        test_indices = np.random.choice(len(self.corpus), n_queries, replace=False)
        
        test_queries = []
        for idx in test_indices:
            doc = self.corpus.iloc[idx]
            test_queries.append({
                'query': doc['Question'],
                'ground_truth_id': int(doc['doc_id']),
                'ground_truth_idx': idx,
                'category': doc['medical_category']
            })
        
        return test_queries
    
    def search(self, query, top_k=10, use_reranking=False):
        """Effectuer une recherche"""
        # Encoder la requête
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Recherche FAISS
        search_k = 50 if use_reranking else top_k
        scores, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            
            doc = self.corpus.iloc[idx]
            results.append({
                'doc_id': int(doc['doc_id']),
                'idx': idx,
                'score': float(score),
                'answer': doc['Answer']
            })
        
        # Re-ranking si demandé
        if use_reranking and len(results) > 0:
            pairs = [[query, r['answer']] for r in results]
            cross_scores = self.cross_encoder.predict(pairs)
            
            for i, score in enumerate(cross_scores):
                results[i]['cross_score'] = float(score)
            
            results = sorted(results, key=lambda x: x['cross_score'], reverse=True)
        
        return results[:top_k]
    
    def calculate_recall_at_k(self, results, ground_truth_id, k=10):
        """Recall@K: Le document pertinent est-il dans les top-K?"""
        doc_ids = [r['doc_id'] for r in results[:k]]
        return 1.0 if ground_truth_id in doc_ids else 0.0
    
    def calculate_mrr_at_k(self, results, ground_truth_id, k=10):
        """MRR@K: Mean Reciprocal Rank"""
        doc_ids = [r['doc_id'] for r in results[:k]]
        if ground_truth_id in doc_ids:
            rank = doc_ids.index(ground_truth_id) + 1
            return 1.0 / rank
        return 0.0
    
    def calculate_precision_at_k(self, results, ground_truth_id, k=10):
        """Precision@K"""
        doc_ids = [r['doc_id'] for r in results[:k]]
        relevant = sum(1 for doc_id in doc_ids if doc_id == ground_truth_id)
        return relevant / k
    
    def calculate_ndcg_at_k(self, results, ground_truth_id, k=10):
        """NDCG@K: Normalized Discounted Cumulative Gain"""
        doc_ids = [r['doc_id'] for r in results[:k]]
        
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(doc_ids, 1):
            if doc_id == ground_truth_id:
                dcg += 1.0 / np.log2(i + 1)
        
        # IDCG (le meilleur cas: document pertinent en position 1)
        idcg = 1.0 / np.log2(2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate(self, test_queries, k_values=[1, 5, 10, 20], use_reranking=False):
        """
        Évaluer le système sur toutes les requêtes de test
        """
        print(f"\n📊 Évaluation sur {len(test_queries)} requêtes...")
        print(f"   Re-ranking: {'Activé' if use_reranking else 'Désactivé'}")
        
        results_data = []
        latencies = []
        
        for query_data in tqdm(test_queries, desc="Évaluation"):
            start_time = time.time()
            
            # Recherche
            results = self.search(
                query_data['query'],
                top_k=max(k_values),
                use_reranking=use_reranking
            )
            
            latency = time.time() - start_time
            latencies.append(latency)
            
            # Calculer les métriques pour chaque K
            metrics = {
                'query': query_data['query'],
                'ground_truth_id': query_data['ground_truth_id'],
                'category': query_data['category'],
                'latency': latency
            }
            
            for k in k_values:
                metrics[f'recall@{k}'] = self.calculate_recall_at_k(
                    results, query_data['ground_truth_id'], k
                )
                metrics[f'mrr@{k}'] = self.calculate_mrr_at_k(
                    results, query_data['ground_truth_id'], k
                )
                metrics[f'precision@{k}'] = self.calculate_precision_at_k(
                    results, query_data['ground_truth_id'], k
                )
                metrics[f'ndcg@{k}'] = self.calculate_ndcg_at_k(
                    results, query_data['ground_truth_id'], k
                )
            
            results_data.append(metrics)
        
        # Créer un DataFrame avec les résultats
        results_df = pd.DataFrame(results_data)
        
        # Calculer les moyennes
        summary = {
            'n_queries': len(test_queries),
            'use_reranking': use_reranking,
            'avg_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99)
        }
        
        for k in k_values:
            summary[f'recall@{k}'] = results_df[f'recall@{k}'].mean()
            summary[f'mrr@{k}'] = results_df[f'mrr@{k}'].mean()
            summary[f'precision@{k}'] = results_df[f'precision@{k}'].mean()
            summary[f'ndcg@{k}'] = results_df[f'ndcg@{k}'].mean()
        
        return results_df, summary
    
    def print_summary(self, summary):
        """Afficher le résumé des résultats"""
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ DE L'ÉVALUATION")
        print("=" * 60)
        
        print(f"\n🔍 Configuration:")
        print(f"   Nombre de requêtes: {summary['n_queries']}")
        print(f"   Re-ranking: {'Activé' if summary['use_reranking'] else 'Désactivé'}")
        
        print(f"\n⏱️  Latence:")
        print(f"   Moyenne: {summary['avg_latency']*1000:.2f} ms")
        print(f"   Écart-type: {summary['std_latency']*1000:.2f} ms")
        print(f"   Min/Max: {summary['min_latency']*1000:.2f} / {summary['max_latency']*1000:.2f} ms")
        print(f"   P50/P95/P99: {summary['p50_latency']*1000:.2f} / {summary['p95_latency']*1000:.2f} / {summary['p99_latency']*1000:.2f} ms")
        
        print(f"\n📈 Métriques de Qualité:")
        
        # Trouver tous les K utilisés
        k_values = sorted([int(k.split('@')[1]) for k in summary.keys() if k.startswith('recall@')])
        
        print(f"\n   Recall@K:")
        for k in k_values:
            print(f"      @{k:2d}: {summary[f'recall@{k}']:.4f}")
        
        print(f"\n   MRR@K:")
        for k in k_values:
            print(f"      @{k:2d}: {summary[f'mrr@{k}']:.4f}")
        
        print(f"\n   Precision@K:")
        for k in k_values:
            print(f"      @{k:2d}: {summary[f'precision@{k}']:.4f}")
        
        print(f"\n   NDCG@K:")
        for k in k_values:
            print(f"      @{k:2d}: {summary[f'ndcg@{k}']:.4f}")
        
        print("\n" + "=" * 60)
    
    def plot_results(self, summary_baseline, summary_reranking):
        """Créer des graphiques de comparaison"""
        print("\n📊 Génération des graphiques...")
        
        k_values = [1, 5, 10, 20]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparaison: Baseline vs Re-ranking', fontsize=16, fontweight='bold')
        
        metrics = ['recall', 'mrr', 'precision', 'ndcg']
        titles = ['Recall@K', 'MRR@K', 'Precision@K', 'NDCG@K']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            baseline_values = [summary_baseline[f'{metric}@{k}'] for k in k_values]
            reranking_values = [summary_reranking[f'{metric}@{k}'] for k in k_values]
            
            x = np.arange(len(k_values))
            width = 0.35
            
            ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
            ax.bar(x + width/2, reranking_values, width, label='Re-ranking', alpha=0.8)
            
            ax.set_xlabel('K')
            ax.set_ylabel(metric.upper())
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(k_values)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
        print("✅ Graphique sauvegardé: evaluation_metrics.png")
        
        # Graphique de latence
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Moyenne', 'P50', 'P95', 'P99']
        baseline_latencies = [
            summary_baseline['avg_latency'] * 1000,
            summary_baseline['p50_latency'] * 1000,
            summary_baseline['p95_latency'] * 1000,
            summary_baseline['p99_latency'] * 1000
        ]
        reranking_latencies = [
            summary_reranking['avg_latency'] * 1000,
            summary_reranking['p50_latency'] * 1000,
            summary_reranking['p95_latency'] * 1000,
            summary_reranking['p99_latency'] * 1000
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, baseline_latencies, width, label='Baseline', alpha=0.8)
        ax.bar(x + width/2, reranking_latencies, width, label='Re-ranking', alpha=0.8)
        
        ax.set_xlabel('Métrique de Latence')
        ax.set_ylabel('Latence (ms)')
        ax.set_title('Comparaison des Latences')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evaluation_latency.png', dpi=300, bbox_inches='tight')
        print("✅ Graphique sauvegardé: evaluation_latency.png")

def main():
    """Pipeline d'évaluation complet"""
    print("=" * 60)
    print("🧪 ÉVALUATION DU MOTEUR DE RECHERCHE")
    print("=" * 60)
    
    # Initialiser l'évaluateur
    evaluator = SearchEvaluator()
    
    # Créer les requêtes de test
    test_queries = evaluator.create_test_queries(n_queries=100)
    
    # Évaluation BASELINE (sans re-ranking)
    print("\n" + "="*60)
    print("📊 ÉVALUATION BASELINE (sans re-ranking)")
    print("="*60)
    
    results_baseline, summary_baseline = evaluator.evaluate(
        test_queries,
        k_values=[1, 5, 10, 20],
        use_reranking=False
    )
    
    evaluator.print_summary(summary_baseline)
    
    # Sauvegarder les résultats baseline
    results_baseline.to_csv('evaluation_baseline.csv', index=False)
    with open('evaluation_baseline_summary.json', 'w') as f:
        json.dump(summary_baseline, f, indent=2)
    
    # Évaluation avec RE-RANKING
    print("\n" + "="*60)
    print("📊 ÉVALUATION AVEC RE-RANKING")
    print("="*60)
    
    results_reranking, summary_reranking = evaluator.evaluate(
        test_queries,
        k_values=[1, 5, 10, 20],
        use_reranking=True
    )
    
    evaluator.print_summary(summary_reranking)
    
    # Sauvegarder les résultats re-ranking
    results_reranking.to_csv('evaluation_reranking.csv', index=False)
    with open('evaluation_reranking_summary.json', 'w') as f:
        json.dump(summary_reranking, f, indent=2)
    
    # Comparaison
    print("\n" + "="*60)
    print("🔄 COMPARAISON BASELINE vs RE-RANKING")
    print("="*60)
    
    print(f"\n📈 Amélioration de la qualité:")
    for k in [1, 5, 10, 20]:
        recall_improv = (summary_reranking[f'recall@{k}'] - summary_baseline[f'recall@{k}']) * 100
        mrr_improv = (summary_reranking[f'mrr@{k}'] - summary_baseline[f'mrr@{k}']) * 100
        print(f"   @{k:2d}: Recall +{recall_improv:+.2f}%, MRR +{mrr_improv:+.2f}%")
    
    print(f"\n⏱️  Impact sur la latence:")
    latency_increase = (summary_reranking['avg_latency'] - summary_baseline['avg_latency']) * 1000
    print(f"   Augmentation: +{latency_increase:.2f} ms ({(latency_increase/summary_baseline['avg_latency']/10):.1f}%)")
    
    # Créer les graphiques
    evaluator.plot_results(summary_baseline, summary_reranking)
    
    print("\n" + "="*60)
    print("✅ ÉVALUATION TERMINÉE")
    print("="*60)
    
    print("\n📁 Fichiers générés:")
    print("   • evaluation_baseline.csv")
    print("   • evaluation_baseline_summary.json")
    print("   • evaluation_reranking.csv")
    print("   • evaluation_reranking_summary.json")
    print("   • evaluation_metrics.png")
    print("   • evaluation_latency.png")

if __name__ == "__main__":
    main()
