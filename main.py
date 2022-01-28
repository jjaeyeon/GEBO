import sys
import argparse
import os
import testFunctions.syntheticFunctions
from methods.GEBO import GEBO

def MixedSpace_Exps(args):
    
    saving_path = f'./results/{args.func}/'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
        
    # define the objective function
    obj_func = args.func
    if obj_func == 'Func2C':
        f = testFunctions.syntheticFunctions.func2C
        categories = [3, 5]

        bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2)},
            {'name': 'h2', 'type': 'categorical', 'domain': (0, 1, 2, 3, 4)},
            {'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)},
            {'name': 'x2', 'type': 'continuous', 'domain': (-1, 1)}]
        
    elif obj_func == 'DTWine':
        f = testFunctions.syntheticFunctions.dt_wine
        categories = [2, 2]

        bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1)},
                  {'name': 'h2', 'type': 'categorical', 'domain': (0, 1)},
                  {'name': 'x1', 'type': 'continuous', 'domain': (0, 1)},
                  {'name': 'x2', 'type': 'continuous', 'domain': (0, 1)}]
        
    elif obj_func == 'SVMBoston':
        f = testFunctions.syntheticFunctions.svm_boston
        categories = [4, 2]

        bounds = [{'name': 'h1', 'type': 'categorical', 'domain': (0, 1, 2, 3)},
                  {'name': 'h2', 'type': 'categorical', 'domain': (0, 1)},
                  {'name': 'x1', 'type': 'continuous', 'domain': (-4, 1)},
                  {'name': 'x2', 'type': 'continuous', 'domain': (-6, 0)}]

        
    elif obj_func == 'PressureVessel':
        f = testFunctions.syntheticFunctions.pressure_vessel
        categories = [100, 100]

        bounds = [{'name': 'h1', 'type': 'categorical', 'domain': tuple(range(100))},
                  {'name': 'h2', 'type': 'categorical', 'domain': tuple(range(100))},
                  {'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)},
                  {'name': 'x2', 'type': 'continuous', 'domain': (-1, 1)}]
        
    elif obj_func == 'CalibEnv':
        f = testFunctions.syntheticFunctions.calibrate_env
        categories = [285]

        bounds = [{'name': 'h1', 'type': 'categorical', 'domain': tuple(range(285))},
                  {'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)},
                  {'name': 'x2', 'type': 'continuous', 'domain': (-1, 1)},
                  {'name': 'x3', 'type': 'continuous', 'domain': (-1, 1)}]

    elif obj_func == 'NNML':
        f = testFunctions.syntheticFunctions.nn_ml
        categories = [14, 4, 9, 3]

        bounds = [{'name': 'h1', 'type': 'categorical', 'domain': tuple(range(14))},
                  {'name': 'h2', 'type': 'categorical', 'domain': tuple(range(4))},
                  {'name': 'h3', 'type': 'categorical', 'domain': tuple(range(9))},
                  {'name': 'h4', 'type': 'categorical', 'domain': tuple(range(3))},
                  {'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)},
                  {'name': 'x2', 'type': 'continuous', 'domain': (-1, 1)},
                  {'name': 'x3', 'type': 'continuous', 'domain': (-1, 1)}]
        
    elif obj_func == 'RobotPush':
        f = testFunctions.syntheticFunctions.robot_pushing
        categories = [11,11,11,11,21,21,21,21,29,29]

        bounds = [
                 {'name': f'h{1+i}', 'type': 'categorical', 'domain': tuple(range(11))}
                  for i in range(4)] + [
                  {'name': f'h{5+i}', 'type': 'categorical', 'domain': tuple(range(21))}
                  for i in range(4)] + [
                  {'name': f'h{9+i}', 'type': 'categorical', 'domain': tuple(range(29))}
                  for i in range(2)] + [
                  {'name': 'x1', 'type': 'continuous', 'domain': (-1, 1)},
                  {'name': 'x2', 'type': 'continuous', 'domain': (-1, 1)},
                  {'name': 'x3', 'type': 'continuous', 'domain': (-1, 1)},
                  {'name': 'x4', 'type': 'continuous', 'domain': (-1, 1)}]
        
    elif obj_func == 'Ackley53C':
        f = testFunctions.syntheticFunctions.ackleycC
        categories = [2]*50 
        
        bounds = []
        for i in range(50):
            bounds.append({'name': f'h{i+1}', 'type': 'categorical', 'domain': (0, 1)})
        for i in range(3):
            bounds.append({'name': f'x{i+1}', 'type': 'continuous', 'domain': (-1, 1)})
  
    else:
        raise NotImplementedError
    
    args.f, args.categories, args.bounds = f, categories, bounds
    
    model = GEBO(args)
                       
    model.runTrials(trials=args.trials, budget=args.max_itr, seed=args.seed if args.seed else 0, saving_path=saving_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run Mixed Space Optimization Experiments")
    parser.add_argument('-f', '--func', help='Objective function', default='Func2C',
                        type=str)
    parser.add_argument('-n', '--max_itr', help='Optimization iterations. Default = 100',
                        default=100, type=int)
    parser.add_argument('-tl', '--trials', help='Number of random trials. Default = 10',
                        default=10, type=int)
    parser.add_argument('-k', '--K', help='Weight hp for retraining',
                        default=1e-2, type=float)
    parser.add_argument('-l', '--lr', help='Learning rate',
                        default=0.05, type=float)
    parser.add_argument('-init', '--init_N', help='Initial number of points',
                        default=40, type=int)
    parser.add_argument('-head_n', '--head_num', help='Number of graphs to generate',
                        default=5, type=int)
    parser.add_argument('-hub_n', '--hub_num', help='Number of nodes to sample (max)',
                        default=3, type=int)
    parser.add_argument('-ls_dim', '--ls_dimension', help='Latent space dimension',
                        default=4, type=int)
    parser.add_argument('-lh', '--loss_hp', help='hp for function loss term',
                       default="0.1 0.1", type=str)
    parser.add_argument('-exp', '--exp_name', help='for saving', type=str)
    parser.add_argument('-s', '--seed', help='seed for initial points', default=None, type=int)
    
    args = parser.parse_args()
    print(f"Got arguments: \n{args}")

    MixedSpace_Exps(args)
