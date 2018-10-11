--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--

require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'models/hg'

local M = {}

function M.setup(opt, checkpoint)
    local model
    if checkpoint then
        local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
        assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
        print('=> Resuming model from ' .. modelPath)
        model = torch.load(modelPath):cuda()
    elseif opt.retrain ~= 'none' then
        assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
        print('Loading model from file: ' .. opt.retrain)
        model = torch.load(opt.retrain):cuda()
        model.__memoryOptimized = nil
    elseif opt.fcn ~= 'none' then
        assert(paths.filep(opt.fcn), 'File not found: ' .. opt.fcn)
        print('Loading model from file: ' .. opt.fcn)
        resnet = torch.load(opt.fcn):cuda()

        local ratio = 2
        local num = 3

        function upblock(inCh) 
            local block = nn.Sequential()
            block:add(nn.SpatialUpSamplingBilinear(2))
            block:add(nn.SpatialConvolution(inCh, inCh/ratio, 1, 1, 1, 1))
            block:add(nn.SpatialBatchNormalization(inCh/2))
            block:add(nn.ReLU())

            return block
        end

        resnet:remove(#resnet.modules)
        resnet:remove(#resnet.modules)
        resnet:remove(#resnet.modules)

        for i = 1, num do
            local inch = 2048 / torch.pow(ratio,i-1)
            resnet:add(upblock(inch))
        end
        
        resnet:add(nn.SpatialConvolution(2048/torch.pow(ratio, num), opt.keypointNum, 1, 1, 1, 1))
        resnet:add(nn.SpatialUpSamplingBilinear(4))
        --resnet:add(nn.Narrow(2,1,opt.keypointnum))

        model = resnet:cuda()
        
        model.__memoryOptimized = nil
    end

    -- Set the CUDNN flags
    if opt.cudnn == 'fastest' then
        cudnn.fastest = true
        cudnn.benchmark = true
    elseif opt.cudnn == 'deterministic' then
        -- Use a deterministic convolution implementation
        model:apply(function(m)
            if m.setMode then m:setMode(1, 1, 1) end
        end)
    end

    -- Wrap the model with DataParallelTable, if using more than one GPU
    if opt.nGPU > 1 then
        local gpus = torch.range(1, opt.nGPU):totable()
        local fastest, benchmark = cudnn.fastest, cudnn.benchmark

        local dpt = nn.DataParallelTable(1, true, true)
        :add(model, gpus)
        :threads(function()
            local cudnn = require 'cudnn'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
        end)
        dpt.gradInput = nil

        model = dpt:cuda()
    end

    local criterion = nn[opt.crit .. 'Criterion']()
    return model, criterion:cuda()
end

return M
