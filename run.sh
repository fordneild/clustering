#train
# python3 main.py --mode train --algorithm lambda_means --model-file speech.lambda_means.model --train-data ./datasets/speech.dev
#test
# python3 main.py --mode test --model-file speech.lambda_means.model --test-data ./datasets/speech.train --predictions-file speech.train.predictions
# #compute clusters
# python3 cluster_accuracy.py ./datasets/speech.test speech.test.predictions


# python3 main.py --mode train --algorithm stochastic_k_means --model-file speech.stochastic_k_means.model --train-data ./datasets/speech.train --number-of-clusters 1
#test
# python3 main.py --mode test --model-file speech.stochastic_k_means.model --test-data ./datasets/speech.train --predictions-file speech.train.predictions



# rm *.model 2> /dev/null
# rm *.predictions 2> /dev/null
# echo Training lambda means on speech
# python3 main.py --mode train --algorithm lambda_means --model-file speech.lambda.model --train-data datasets/speech.train
# python3 main.py --mode test --model-file speech.lambda.model --test-data datasets/speech.dev --predictions-file speech.dev.predictions
# python3 cluster_accuracy.py datasets/speech.dev speech.dev.predictions


rm *.model 2> /dev/null
rm *.predictions 2> /dev/null
echo Training stochastic_k means on speech
python3 main.py --mode train --algorithm stochastic_k_means --model-file speech.stochastic_k.model --train-data datasets/speech.train
python3 main.py --mode test --model-file speech.stochastic_k.model --test-data datasets/speech.dev --predictions-file speech.dev.predictions
python3 cluster_accuracy.py datasets/speech.dev speech.dev.predictions


