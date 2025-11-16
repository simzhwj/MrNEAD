"""
MrNEAD Core Algorithm

Based on the objective function:
min_{H,M,H_m} ||A-HH^T||_F^2 + α||A+M-H_m H_m^T||_F^2 + β Tr(HH^T H_m H_m^T) + γ||M-Q̂||_F^2

Subject to: H≥0, H_m≥0, A+M∈[0,1], ||M||_0 < ξ
"""

import numpy as np
import heapq

class MrNEAD:
    def __init__(self, k=5, xi=4, alpha=0.01, beta=0.01, gamma=10.0, max_iter=100, seed=2):
        """
        Initialize MrNEAD algorithm
        
        Parameters:
        -----------
        k : int, embedding dimension
        xi : int, attack budget (number of edge modifications)
        alpha : float, reconstruction weight
        beta : float, modularity guidance weight
        gamma : float, adversarial weight
        max_iter : int, maximum iterations
        seed : int, random seed
        """
        self.k = k
        self.xi = 2*xi
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.seed = seed
        
        np.random.seed(seed)
    
    def fit(self, A):
        self.A = A
        self.n = A.shape[0]
        
        # Initialize matrices
        self.H = np.maximum(0, np.random.rand(self.n, self.k))
        self.H_m = np.maximum(0, np.random.rand(self.n, self.k))
        self.M = np.zeros_like(A)
        
        # Compute inverted modularity matrix Q̂
        self.Q_hat = self._compute_inverted_modularity(A)
        
        # Store objective values
        objective_values = []
        
        print(f"   Starting optimization with {self.max_iter} iterations...")
        
        for iter_num in range(self.max_iter):
            # Update M (adversarial matrix)
            self.M = self._update_M()
            
            # Update H (original embedding)
            self.H = self._update_H()
            
            # Update H_m (attacked embedding)  
            self.H_m = self._update_H_m()
            
            # Compute objective value
            obj_val = self._compute_objective()
            objective_values.append(obj_val)
            
            if (iter_num + 1) % 10 == 0:
                print(f"     Iteration {iter_num + 1}/{self.max_iter}: Objective = {obj_val:.6f}")
        
        # Generate final attacked adjacency matrix
        A_attacked = A + np.sign(self.M)
        A_attacked = np.clip(A_attacked, 0, 1)  # Ensure valid adjacency matrix
        
        return {
            'M': np.sign(self.M),
            'A_attacked': A_attacked,
            'objective_values': objective_values,
            'H': self.H,
            'H_m': self.H_m
        }
    
    def _compute_inverted_modularity(self, A):
        """Compute inverted modularity matrix Q̂ = -Q"""
        degrees = np.sum(A, axis=1)
        m = np.sum(degrees) / 2  # Total number of edges
        
        # Q = A - (d_i * d_j) / (2m)
        # Q̂ = -Q = (d_i * d_j) / (2m) - A
        Q_hat = np.outer(degrees, degrees) / (2 * m) - A
        return Q_hat
    
    def _update_M(self):
        """Update adversarial matrix M using sparse selection"""
        # Compute candidate matrix
        # According to paper Eq. (516): Ṁ = (γQ̂ - α(A - B)) / (α + γ)
        B_m = self.H_m @ self.H_m.T
        M_candidate = (self.gamma * self.Q_hat - self.alpha * (self.A - B_m)) / (self.alpha + self.gamma)
        
        # Clip to valid range
        M_candidate = np.clip(M_candidate, -self.A, 1 - self.A)
        
        # Select top xi modifications using heap
        return self._sparse_selection(M_candidate, self.xi)
    
    def _sparse_selection(self, M_candidate, xi):
        """Select top xi modifications while avoiding isolated nodes"""
        n = M_candidate.shape[0]
        abs_M = np.abs(M_candidate)
        heap = []
        
        # Build heap with top 2*xi candidates (upper triangle only)
        for i in range(n):
            for j in range(i + 1, n):
                val = abs_M[i, j]
                if len(heap) < 2 * xi:
                    heapq.heappush(heap, (val, i, j))
                elif val > heap[0][0]:
                    heapq.heappushpop(heap, (val, i, j))
        
        # Initialize sparse matrix
        M = np.zeros_like(self.A)
        degree_tracker = np.sum(self.A, axis=1).copy()
        count = 0
        
        # Select modifications while avoiding isolated nodes
        while heap and count < xi:
            _, i, j = heapq.heappop(heap)
            
            m_val = M_candidate[i, j]
            new_degree_i = degree_tracker[i] + np.sign(m_val)
            new_degree_j = degree_tracker[j] + np.sign(m_val)
            
            # Skip if would create isolated nodes
            if new_degree_i <= 0 or new_degree_j <= 0:
                continue
            
            # Add modification
            M[i, j] = m_val
            M[j, i] = m_val
            degree_tracker[i] = new_degree_i
            degree_tracker[j] = new_degree_j
            count += 1
        
        return M
    
    def _update_H(self):
        """Update original embedding H using multiplicative updates"""
        # According to paper Eq. (623): H ← H ⊙ (AH) / (HH^T H + (β/2) H_m H_m^T H)
        numerator = self.A @ self.H
        denominator = self.H @ (self.H.T @ self.H) + (self.beta / 2) * (self.H_m @ self.H_m.T) @ self.H
        denominator = np.maximum(denominator, 1e-10)
        
        return self.H * (numerator / denominator)
    
    def _update_H_m(self):
        """Update attacked embedding H_m using multiplicative updates"""
        # According to paper Eq. (637): H_m ← H_m ⊙ (Â H_m) / (H_m H_m^T H_m + (β/(2α)) H H^T H_m)
        A_plus_M = self.A + self.M
        numerator = A_plus_M @ self.H_m
        denominator = self.H_m @ (self.H_m.T @ self.H_m) + (self.beta / (2 * self.alpha)) * (self.H @ self.H.T) @ self.H_m
        denominator = np.maximum(denominator, 1e-10)
        
        return self.H_m * (numerator / denominator)
    
    def _compute_objective(self):
        """Compute the overall objective function value"""
        # NEAD terms
        term1 = np.linalg.norm(self.A - self.H @ self.H.T, 'fro') ** 2
        term2 = self.alpha * np.linalg.norm(self.A + self.M - self.H_m @ self.H_m.T, 'fro') ** 2
        term3 = self.beta * np.trace((self.H @ self.H.T) @ (self.H_m @ self.H_m.T))
        
        # MIGS term
        term4 = self.gamma * np.linalg.norm(self.M - self.Q_hat, 'fro') ** 2
        
        return term1 + term2 + term3 + term4
