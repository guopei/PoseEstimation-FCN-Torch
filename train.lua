--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'
require 'hdf5'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)

function Trainer:__init(model, criterion, opt, optimState)
    self.model = model
    self.criterion = criterion
    self.optimState = optimState or {
        learningRate = opt.LR,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        nesterov = true,
        dampening = 0.0,
        weightDecay = opt.weightDecay,
    }
    self.opt = opt
    self.params, self.gradParams = model:getParameters()
    self.trainLogger = optim.Logger(paths.concat(opt.logs, 'optim_log_' .. opt.timeString))
    self.mstd = {
        mean = { 0.485, 0.456, 0.406 },
        std = { 0.229, 0.224, 0.225 },
    }

end

function Trainer:train(epoch, dataloader)

    -- Trains the model for a single epoch
    self.optimState.learningRate = self:learningRate(epoch)

    local timer = torch.Timer()
    local dataTimer = torch.Timer()

    local function feval()
        return self.criterion.output, self.gradParams
    end

    local trainSize = dataloader:size()
    local accSum, lossSum = 0.0, 0.0
    local N = 0

    print('=> Training epoch # ' .. epoch)
    -- set the batch norm to training mode
    self.model:training()
    for n, sample in dataloader:run() do
        local dataTime = dataTimer:time().real

        -- Copy input and target to the GPU
        self:copyInputs(sample)

        local output = self.model:forward(self.input)
        local batchSize = output:size(1)

        local loss = self.criterion:forward(output, self.target)
        local acc = self:computeScore(output, self.target)

        self.model:zeroGradParameters()
        self.criterion:backward(output, self.target)
        self.model:backward(self.input, self.criterion.gradInput)

        optim.rmsprop(feval, self.params, self.optimState)

        lossSum = lossSum + loss*batchSize
        accSum = accSum + acc * batchSize
        N = N + batchSize

        if n % 100 == 0 then
            print((' | Epoch: [%d][%4d/%4d]  Time %.3f  Data %.3f  Err %1.3e(%1.3e)  PCK %6.3f(%6.3f)'):format(
            epoch, n, trainSize, timer:time().real, dataTime, loss, lossSum/N, acc*100, accSum/N*100))

            if self.opt.display then
                win1 = image.display{image=output[1], win=win1, zoom = 1}
                win2 = image.display{image=self.target[1], win=win2, zoom = 1}
            end

        end

        -- check that the storage didn't get changed do to an unfortunate getParameters call
        assert(self.params:storage() == self.model:parameters()[1]:storage())

        timer:reset()
        dataTimer:reset()
    end

    return  lossSum/N, accSum/N 
end

function Trainer:test(epoch, dataloader)
    -- Computes the top-1 and top-5 err on the validation set

    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local size = dataloader:size()

    local nCrops = self.opt.tenCrop and 10 or 1
    local accSum, lossSum = 0.0, 0.0
    local accAllSum = torch.zeros(17)
    local N = 0

    self.model:evaluate()
    for n, sample in dataloader:run() do

        local dataTime = dataTimer:time().real

        -- Copy input and target to the GPU
        self:copyInputs(sample)

        local output = self.model:forward(self.input)
        local batchSize = output:size(1) / nCrops
        local loss = self.criterion:forward(output, self.target)
        local acc = self:computeScore(output, self.target)

        lossSum = lossSum + loss*batchSize
        accSum = accSum + acc * batchSize
        N = N + batchSize

        if n % 100 == 0 then
            print((' | Test: [%d][%4d/%4d]   Time %.3f  Data %.3f  Err %1.3e(%1.3e)  PCK %7.3f(%7.3f)'):format(
            epoch, n, size, timer:time().real, dataTime, loss, lossSum/N, acc*100, accSum/N*100))
            if self.opt.display then
                win1 = image.display{image=output[1], win=win1, zoom = 1}
                win2 = image.display{image=self.target[1], win=win2, zoom = 1}
            end
        end

        timer:reset()
        dataTimer:reset()
    end 
    self.model:training()
    return lossSum/N, accSum/N
end

-- added trainValFlag for training field.
function Trainer:generate(dataloader, trainValFlag)
    local folder = self.opt.destFolder 

    if not paths.dirp(folder) then
        paths.mkdir(folder)
    end

    local size = dataloader:size()
    local epsilon = 1e-7
    local iouSum = 0
    local N = 0

    self.model:evaluate()
    for n, sample in dataloader:run() do
        self:copyInputs(sample)
        self.model:forward(self.input)
        local output = self.upg:forward(self.model.output)
        local iou = self:computeScore(output, self.target)
        local batchSize = output[self.opt.nStack]:size(1)
        iouSum = iouSum + iou * batchSize
        N = N + batchSize

        if n % 100 == 0 then
            print((' | Test: [%d/%d],: %6.3f(%6.3f)'):format(n, size, iou, iouSum / N))
            if self.opt.display then
                win1 = image.display{image=output[self.opt.nStack][1], win=win1, zoom = 1}
                win2 = image.display{image=output[self.opt.nStack][1]:gt(0.5), win=win2, zoom = 1}
            end
        end

        for i = 1, batchSize do
            local base = paths.dirname(sample.name[i]) .. '_' .. paths.basename(sample.name[i]):match('[^%.]+')
            local name = folder .. base .. '.hdf5'

            local prediction = output[self.opt.nStack][i]:float()

            -- load original image and scale
            local org = self.input[i]:float() 
            org = self:InvColorNormalize(org)
            local binary = prediction:gt(0.5)
            --win1 = image.display{image=org, win=win1}
            --io.read()

            local h5file = hdf5.open(name, 'w')
            h5file:write('image', org)
            h5file:write('training', torch.ByteTensor({trainValFlag}))
            h5file:write('mask', binary)
            h5file:write('output', prediction)
            h5file:write('groundtruth', self.target[self.opt.nStack][i]:float())
            h5file:close()
        end

    end
    self.model:training()
end

function Trainer:copyInputs(sample)
    -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
    -- if using DataParallelTable. The target is always copied to a CUDA tensor
    self.input = self.input or (self.opt.nGPU == 1
    and torch.CudaTensor()
    or cutorch.createCudaHostTensor())
    self.input:resize(sample.input:size()):copy(sample.input)
    self.target = torch.CudaTensor()
    self.target:resize(sample.target:size()):copy(sample.target) 
end

function Trainer:learningRate(epoch)
    -- Training schedule
    local decay = 0
    if self.opt.dataset == 'imagenet' then
        decay = math.floor((epoch - 1) / 30)
    elseif self.opt.dataset == 'cifar10' then
        decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
    elseif self.opt.dataset == 'cifar100' then
        decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
    end
    return self.opt.LR * math.pow(0.1, decay)
end

function Trainer:computeScore(output, target)
    local pck = self:pckScore(output,target)
    return pck, visi
end

function Trainer:threshDists(dists, thr)
    -- Return percentage below threshold while ignoring values with a -1
    if not thr then thr = 1 end
    if torch.ne(dists,-1):sum() > 0 then
        return dists:le(thr):eq(dists:ne(-1)):sum() / (dists:ne(-1):sum() + 1E-7)
    else
        return -1
    end
end

function Trainer:calcDists(preds, label, normalize)
    local dists = torch.Tensor(preds:size(2), preds:size(1))
    local diff = torch.Tensor(2)
    for i = 1,preds:size(1) do
        for j = 1,preds:size(2) do
            if label[i][j][1] > 1 and label[i][j][2] > 1 then
                dists[j][i] = torch.dist(label[i][j],preds[i][j])/(normalize[i] + 1E-7)
            else
                dists[j][i] = -1
            end
        end
    end
    return dists
end

function Trainer:getPreds(hm)
    assert(hm:size():size() == 4, 'Input must be 4-D tensor')
    local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
    local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    preds:select(3,1):apply(function(x) return (x - 1) % hm:size(4) + 1 end)
    preds:select(3,2):add(-1):div(hm:size(3)):floor():add(1)
    return preds
end

function Trainer:pckScore(output, labels)
    -- Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
    -- First value to be returned is average accuracy, followed by individual accuracies

    local pds = self:getPreds(output)
    local gts = self:getPreds(labels)

    -- 5% of the image size.
    local dists = self:calcDists(pds, gts, torch.ones(pds:size(1))*self.opt.outputRes/10)
    local acc = {}
    local avgAcc = 0.0
    local badIdxCount = 0

    for i = 1,dists:size(1) do
        acc[i+1] = self:threshDists(dists[i])

        if acc[i+1] >= 0 then 
            avgAcc = avgAcc + acc[i+1]
        else 
            badIdxCount = badIdxCount + 1 
        end
    end
    acc[1] = avgAcc / (dists:size(1) - badIdxCount + 1E-7)
    --return torch.Tensor(acc)
    return acc[1]
end


function Trainer:InvColorNormalize(img, meanstd)
    meanstd = meanstd or self.mstd
    for i=1,3 do
        img[i]:mul(meanstd.std[i])
        img[i]:add(meanstd.mean[i])
        img[i] = (img[i]:add(-img[i]:min()):div(img[i]:max()-img[i]:min()))
    end
    return img
end

return M.Trainer
