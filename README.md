## Code Impementation for "Mold into a Graph: Efficient Bayesian Optimization over Mixed Spaces"

This repo contains the implementation of GEBO algorithm.

## Dependencies
```
Anaconda Python 3.7
torch==1.7.1
torch-geometric==1.7.2
networkx==2.6.3
GPy==1.10.0
scikit-learn==1.0.2
scipy==1.7.1
numpy==1.20.3
tqdm==4.32.1
pandas==1.3.4
```


## To run the experiments,
```bash
  python main.py --func CalibEnv 
  python main.py --func RobotPush 
```
where ```--func``` specifics the task problem. You can either use the bash file ```run.sh``` to run the experiments.

## Acknowledgements
The code is built upon the source code of [CoCaBO](https://github.com/rubinxin/CoCaBO_code). We thank the authors for their provision of the code. 
