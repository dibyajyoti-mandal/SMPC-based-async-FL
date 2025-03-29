#include <iostream>
#include <vector>
#include <random>
#include <cmath>
using namespace std;

const int p = 1e9+7;  // modulus bound

// Helper function for modulo operation that works with negative numbers.
double mod(double a, int m) {
    double r = fmod(a, m);
    if (r < 0)
        r += m;
    return r;
}

int main(){
    int n = 3;      // number of clients
    int dim = 5;    // dimension of the weight vectors and common randomness vectors

    // Initialize the clients' weight vectors (each weight vector has 5 elements).
    vector<vector<double>> clientWeights(n, vector<double>(dim, 0.0));
    clientWeights[0] = {0.23, 0.45, 0.32, 0.12, 0.34};
    clientWeights[1] = {0.10, 0.20, 0.30, 0.40, 0.50};
    clientWeights[2] = {0.15, 0.25, 0.35, 0.45, 0.55};
    vector<double> clientWeightsAvg(dim, 0.0);
    vector<double> encryptedWeightsAvg(dim, 0.0);
    cout<<"Client weights averaged: "<<endl;
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < n; j++){
            clientWeightsAvg[i] += clientWeights[j][i];
        }
        clientWeightsAvg[i] /= n;
        cout<<clientWeightsAvg[i]<<" ";
    }cout<<endl;
    // Setup random generator to simulate pairwise common randomness.
    random_device rd;
    mt19937 gen(rd());
    // Generate randomness in the range [0, 10).
    uniform_real_distribution<> dis(0.0, 10.0);

    // For each pair (i, j) with i < j, generate a 5-dimensional randomness vector.
    vector<vector<double>> r(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++){
            for (int d = 0; d < n; d++){
                if(r[i][d] != 0.0) {
                    continue;
                }; // Skip if already generated
                r[i][d] = dis(gen);
                r[d][i] = r[i][d]; // Ensure symmetry
            }
    }
    
    cout<<"Mask weights generated: "<<endl;
    for (int i = 0; i < n; i++){
        for (int d = 0; d < n; d++){
            cout<<r[i][d]<<" ";
        }cout<<endl;
    }
    // Each client computes its masked weight vector:
    //   w̄_i = (w_i + sum_{j=i+1}^{n-1} r[i][j] - sum_{k=0}^{i-1} r[k][i]) mod p
    vector<vector<double>> maskedWeights(n, vector<double>(dim, 0.0));
    for (int i = 0; i < n; i++){
        // Start with the client's own weight vector.
        vector<double> m = clientWeights[i];
        
        // Add randomness from clients with higher indices.
        for (int j = i + 1; j < n; j++){
            for (int d = 0; d < dim; d++){
                m[d] += r[i][j];
            }
        }
        // subtract randomness from clients with lower indices.
        for (int k = 0; k < i; k++){
            for (int d = 0; d < dim; d++){
                m[d] -= r[k][i];
            }
        }
        maskedWeights[i] = m;
    }
    
    // Output the masked weights for each client.
    for (int i = 0; i < n; i++){
        cout << "Client " << i << " masked weights: ";
        for (int d = 0; d < dim; d++){
            cout << maskedWeights[i][d] << " ";
        }
        cout << endl;
    }

    cout<<"Encrypyed weights averaged: "<<endl;
    for(int i = 0; i < dim; i++){
        for(int j = 0; j < n; j++){
            encryptedWeightsAvg[i] += maskedWeights[j][i];
        }
        encryptedWeightsAvg[i] /= n;
        cout<<encryptedWeightsAvg[i]<<" ";
    }cout<<endl;
    
    return 0;
}
