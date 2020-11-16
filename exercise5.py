import numpy as np
import matplotlib.pyplot as plt

from robo.fmin import bayesian_optimization
from ho_exercise import runtime, objective_function


def random_search(num_iterations=50):
    for i in range(10):
        rng = np.random.RandomState()

        incumbents = np.zeros((10, 50))
        incumbents_values = np.zeros((10, 50))
        run_time = []
        incumbents_runtime = np.zeros((10, 50))

        X = []
        y = []

        for it in range(num_iterations):

            lower = np.array([-6, 32, 4, 4, 4])
            upper = np.array([0, 512, 10, 10, 10])
        
            # Choose next point to evaluate
            new_x = rng.uniform(lower, upper)

            # Evaluate
            new_y = objective_function(new_x)
            
            # Check the runtime
            new_runtime = runtime(new_x)

            # Update the data
            X.append(new_x.tolist())
            y.append(new_y)
            run_time.append(new_runtime)

            # The incumbent is just the best observation we have seen so far
            best_idx = np.argmin(y)

            incumbent_value = y[best_idx]
            
            best_idx_runtime = np.argmin(run_time)


            incumbents_values[i][it] = incumbent_value
            incumbents_runtime[i][it] = run_time[best_idx_runtime]


    inc = np.mean(incumbents_values, axis=0)
    inc_runtime = np.mean(incumbents_runtime, axis=0)
    
    epochs = np.array([i for i in range(50)])
    plt.plot(epochs, inc)
    plt.xlabel('epochs')
    plt.ylabel('incumbents')
    plt.show()
    return inc, inc_runtime

def bayesian_opt():

    lower = np.array([-6, 32, 4, 4, 4])
    upper = np.array([0, 512, 10, 10, 10])
    
    f = objective_function
    incumbents = np.zeros((10, 50))
    incumbents_runtime = np.zeros((10, 50))

    
    for i in range(10):
        result = bayesian_optimization(f, lower, upper, num_iterations=50)
    incumbents_runtime[i] = result["runtime"]
    incumbents[i] = result["incumbent_values"]
    inc = np.mean(incumbents, axis=0)
    runtime = np.mean(incumbents_runtime, axis=0)

    return inc, runtime

def plot_(l1, l2, title):
    epochs = np.array([i for i in range(50)])

    plt.plot(epochs, l1, label='bayesian')
    plt.plot(epochs, l2, label='random')
    plt.xlabel('epochs')
    plt.ylabel(title)
    plt.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':

    inc_bay, bayesian_runtime = bayesian_opt()
    inc_random, random_runtime = random_search()
    
    # plotting the mean performance of incumbents
    plot_(inc_bay, inc_random,'incumbents') 
    
    # plotting the the cumulative runtime after each iteration
    random_run = np.cumsum(random_runtime)
    bayesian_run = np.cumsum(bayesian_runtime)
    plot_(bayesian_run, random_run,'cumulative runtime')

