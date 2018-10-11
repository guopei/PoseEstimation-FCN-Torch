--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'nngraph'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)

cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)
print(model)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)


if opt.testOnly then
    local testAcc = trainer:test(0, valLoader)
    print(string.format(' * Results acc: %6.3f\n', testAcc))
    return
end

if opt.genOnly then
    print('=======================')
    print('starting generate train folder')
    trainer:generate(trainLoader, 1)
    print('train finished')
    print('=======================')
    print('starting generate val folder')
    trainer:generate(valLoader, 0)
    print('val finished')
    return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
for epoch = startEpoch, opt.nEpochs do
    -- Train for a single epoch
    local trainLoss, trainAcc = trainer:train(epoch, trainLoader)
    print(string.format(' | train finished with average loss %.3f and accuracy: %.3f', 
            trainLoss, trainAcc*100))
    local testLoss, testAcc = trainer:test(epoch, valLoader)
    print(string.format(' | test finished with average loss %.3f and accuracy: %.3f', 
            testLoss, testAcc*100))
    collectgarbage()
end

checkpoints.save(model, optimState, opt)
print('model weights has been saved!')
