from sklearn.model_selection import train_test_split
import pandas as pd
import optuna
import warnings
from params import (DATASET_PATH, DATASET_NAME, TARGET_COL, TEST_SIZE, VAL_SIZE)
from visualizations import visualize_knowledge_graph
import networkx as nx
from grace import run_grace

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Main Application Logic ---

def load_and_split_data():
    df = pd.read_csv(DATASET_PATH, encoding='utf-8')
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    n_classes = y.nunique()
    print(f"Dataset: {len(X)} samples, {len(X.columns)} features, {n_classes} classes")
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=VAL_SIZE, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    results, kg = run_grace(X_train, X_val, X_test, y_train, y_val, y_test)
    print(f"Final Optimized Model Results:")
    print(f"Test Score: {results['test_score']:.4f}")
    print(f"Number of Features: {results['n_features']}")
    print(f"Number of Edges: {results['n_edges']}")
    print("\nVisualizing Final Knowledge Graph...")
    visualize_knowledge_graph(kg, DATASET_NAME)
    nx.write_graphml(kg, f"kg/{DATASET_NAME}_kg_final.graphml")
    print(f"Feature interaction graph saved to 'images/{DATASET_NAME}_feature_interaction_graph.png'")
    print(f"Final knowledge graph data saved to 'kg/{DATASET_NAME}_kg_final.graphml'")
    return results

if __name__ == "__main__":
    results = main() 