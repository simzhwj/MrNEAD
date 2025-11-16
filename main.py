"""
MrNEAD - Main Program
This is a simplified demonstration of the MrNEAD algorithm on the Dolphins dataset.
"""

import numpy as np
from mrNEAD_core import MrNEAD
from utils import load_network, evaluate_attack


def main():
    print("=" * 60)
    print("MrNEAD - Dolphins Network Attack")
    print("=" * 60)

    # Load Football network
    print("\n1. Loading network...")
    G, A = load_network("dolphins")
    n_nodes = A.shape[0]
    print(f"   Network loaded: {n_nodes} nodes, {G.number_of_edges()} edges")
    
    # Parameters for MrNEAD
    print("\n2. Setting parameters...")
    params = {
        'k': 4,          # embedding dimension
        'xi': 8,         # attack budget (number of edge modifications)
        'alpha': 0.01,    # reconstruction weight
        'beta': 0.01,     # modularity guidance weight
        'gamma': 0.01,    # adversarial weight
        'max_iter': 100   # maximum iterations
    }
    
    print(f"   Parameters: k={params['k']}, xi={params['xi']}")
    print(f"   alpha={params['alpha']}, beta={params['beta']}, gamma={params['gamma']}")
    
    # Run MrNEAD attack
    print("\n3. Running MrNEAD attack...")
    mrNEAD = MrNEAD(**params)
    results = mrNEAD.fit(A)

    # Extract results
    M = results['M']  # adversarial matrix
    A_attacked = results['A_attacked']  # attacked adjacency matrix
    obj_values = results['objective_values']
    
    print(f"   Final objective value: {obj_values[-1]:.6f}")
    print(f"   Number of modifications: {np.count_nonzero(M)//2}")  # divide by 2 for undirected
    
    # Evaluate attack effectiveness
    print("\n4. Evaluating attack effectiveness...")
    evaluation = evaluate_attack(A, A_attacked)

    print(f"   NMI (Louvain): {evaluation['nmi_louvain']:.4f}")
    print(f"   ARI (Louvain): {evaluation['ari_louvain']:.4f}")

    print("\n" + "=" * 60)
    print("Completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
    

