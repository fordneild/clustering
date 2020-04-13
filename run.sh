#train
python3 main.py --mode train --algorithm lambda_means --model-file speech.lambda_means.model --train-data ./datasets/speech.train
#test
python3 main.py --mode test --model-file speech.lambda_means.model --test-data ./datasets/speech.train --predictions-file speech.train.predictions
#compute clusters
python3 cluster_accuracy.py ./datasets/speech.test speech.test.predictions


python3 main.py --mode train --algorithm stochastic_k_means --model-file speech.stochastic_k_means.model --train-data ./datasets/speech.train --number-of-clusters 1