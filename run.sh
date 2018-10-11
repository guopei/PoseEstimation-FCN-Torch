#fcn for key point detection
CUDA_VISIBLE_DEVICES=6 th main.lua  -batchSize 16 -fcn ./models/resnet-50.t7 -data /mv_users/peiguo/dataset/cub-fewshot/full/ -nEpochs 50 -display false -outputRes 256 -gaussianSize 4 #-retrain checkpoints/model_18_10_08_16_07_34.t7 -testOnly false
