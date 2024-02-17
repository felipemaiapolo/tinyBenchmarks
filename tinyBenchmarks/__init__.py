import requests
import pickle
import numpy as np
from scipy.optimize import minimize
import os

### Utility functions
def sigmoid(z):
    return 1/(1+np.exp(-z))

def item_curve(theta, a, b):
    z = np.clip(a*theta - b, -30, 30).sum(axis=1)
    return sigmoid(z)

def fit_theta(responses_test, seen_items, A, B, theta_init=None, eps=1e-10, optimizer="BFGS"):
    D = A.shape[1]
    # Define the negative log likelihood function
    def neg_log_like(x):
        P = item_curve(x.reshape(1, D, 1), A[:, :, seen_items], B[:, :, seen_items]).squeeze()
        log_likelihood = np.sum(responses_test[seen_items] * np.log(P + eps) + (1 - responses_test[seen_items]) * np.log(1 - P + eps))
        return -log_likelihood
    # Use the minimize function to find the ability parameters that minimize the negative log likelihood
    optimal_theta = minimize(neg_log_like, np.zeros(D), method = optimizer).x[None,:,None] 
    return optimal_theta

### Evaluation function
def evaluate(y_input, bench):

    assert len(y_input.shape)==1, "y_input must be a unidimensional numpy array."
    assert bench in ['lb', 'mmlu', 'helm_lite', 'alpaca']
    
    # Downloading files
    if not os.path.isfile("tinyBenchmarks.pkl"):
        url = "https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/main/tinyBenchmarks/tinyBenchmarks.pkl"
        response = requests.get(url)
        if response.status_code == 200:
            # Write the content to a file
            with open("tinyBenchmarks.pkl", "wb") as file:
                file.write(response.content)

    ### Loading and creating important objects
    number_of_examples = 100
    with open('tinyBenchmarks.pkl', 'rb') as handle:
        tinyBenchmarks = pickle.load(handle)

    seen_examples = tinyBenchmarks[bench]['seen_examples']
    examples_weights = tinyBenchmarks[bench]['examples_weights']
    irt_parameters = tinyBenchmarks[bench]['irt_parameters']
    A, B = irt_parameters['A'], irt_parameters['B']
    optimal_lambdas = tinyBenchmarks[bench]['optimal_lambdas']
    scenarios_position = tinyBenchmarks[bench]['scenarios_position']
    subscenarios_position = tinyBenchmarks[bench]['subscenarios_position']

    N = np.max([np.max(x) for x in scenarios_position.values()])+1
    balance_weights = np.ones(N)
    for scenario in scenarios_position.keys():
        N_sce = len(scenarios_position[scenario])
        n_sub = len(subscenarios_position[scenario])
        for sub in subscenarios_position[scenario].keys():
            n_i = len(subscenarios_position[scenario][sub])
        balance_weights[subscenarios_position[scenario][sub]] = N_sce/(n_sub*n_i) 

    ### Creating vector y and estimating theta
    y = np.zeros(N)
    for i, j in enumerate(seen_examples):
        y[j] = y_input[i]

    ### Getting estimates
    theta = fit_theta(y, seen_examples, A, B)
    estimates = {}
    unseen_examples = [i for i in range(N) if i not in seen_examples]

    for scenario in scenarios_position.keys():

        N_sce = len(scenarios_position[scenario])
        seen_examples_sce = [s for s in seen_examples if s in scenarios_position[scenario]]
        unseen_examples_sce = [s for s in unseen_examples if s in scenarios_position[scenario]]

        data_part_IRTp = ((balance_weights*y)[seen_examples_sce]).mean()
        irt_part = (balance_weights*item_curve(theta.reshape(1, A.shape[1], 1), A, B))[0, [unseen_examples_sce]].mean()
        IRTp_lambd = number_of_examples/N_sce
        IRT = (examples_weights[scenario]*y[seen_examples_sce]).sum()
        IRTp = IRTp_lambd * data_part_IRTp + (1 - IRTp_lambd) * irt_part
        IRTpp = optimal_lambdas[scenario]*IRT + (1-optimal_lambdas[scenario])*IRTp

        estimates[scenario] = {}
        estimates[scenario]['irt'] = IRT
        estimates[scenario]['pirt'] = IRTp
        estimates[scenario]['gpirt'] = IRTpp
        
    return estimates