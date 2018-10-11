--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('facebook resent framework + stack hour glass -> key points regression')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-data',       '/home/peiguo/dataset/cub-200-train-val',         'Path to dataset')
    cmd:option('-manualSeed', 0,                                                'Manually set RNG seed')
    cmd:option('-GPU',        2,                                                'Default preferred GPU')
    cmd:option('-nGPU',       1,                                                'Number of GPUs to use by default')
    cmd:option('-backend',    'cudnn',                                          'Options: cudnn | cunn')
    cmd:option('-cudnn',      'fastest',                                        'Options: fastest | default | deterministic')
    cmd:option('-gen',        'gen',                                            'Path to save generated files')
    ------------- Data options ------------------------
    cmd:option('-nThreads',        1,       'number of data loading threads')
    ------------- Training options --------------------
    cmd:option('-nEpochs',         50,     'Number of total epochs to run')
    cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       6,       'mini-batch size (1 = pure stochastic)')
    cmd:option('-testOnly',        'false', 'Run on validation set only')
    cmd:option('-genOnly',         'false', 'Generate masks for all images')
    cmd:option('-saveOutput',      'false', 'Save output of test')
    cmd:option('-tenCrop',         'false', 'Ten-crop testing')
    ------------- Checkpointing options ---------------
    cmd:option('-dataset',         'imagenet',    'Directory in which to save dataset')
    cmd:option('-save',            'checkpoints', 'Directory in which to save checkpoints')
    cmd:option('-logs',            'logs',        'Directory in which to save logs')
    cmd:option('-resume',          'none',        'Resume from the latest checkpoint in this directory')
    cmd:option('-destFolder',      '',    'Path to generage seg masks')
    cmd:option('-fcn',             '',    'Path to fcn model')
    ---------- Optimization options ----------------------
    cmd:option('-LR',              1e-4,  'initial learning rate')
    cmd:option('-momentum',        0,       'momentum')
    cmd:option('-weightDecay',     0,       'weight decay')
    ---------- Model options ----------------------------------
    cmd:option('-retrain',      'none',   'Path to model to retrain with')
    cmd:option('-optimState',   'none',   'Path to an optimState to reload from')
    ---------- Model options ----------------------------------
    cmd:option('-gaussianSize',     1,      'Gaussian blob size')
    cmd:option('-outputRes',        64,     'Output heat map size')
    cmd:option('-nFeats',           256,    'output channel size for each conv block')
    cmd:option('-nModules',         1,      'repeating modules number')
    -- default value to predict cub key points
    cmd:option('-keypointNum',      15,     'Key point number') 
    cmd:option('-crit',            'MSE',   'Criterion type')
    cmd:option('-saveTest',        'false',   'save test')
    cmd:option('-display',         'false',   'display training process')
    cmd:option('-multiTask',       'false',   'add visibility prediction')
    cmd:option('-shg',             'false',   'create model instead of loading pretrained')
    cmd:text()

    local opt = cmd:parse(arg or {})

    if opt.dataset == 'tim' then
        opt.keypointNum = 1
    elseif opt.dataset == 'bf' or opt.dataset == 'bf-no-target' then
        opt.keypointNum = 9
    end
    
    -- create time stamp
    opt.timeString = os.date('%y_%m_%d_%H_%M_%S')
    -- create log file
    if not paths.dir(opt.logs) then paths.mkdir(opt.logs) end
    cmd:log(paths.concat(opt.logs, 'cmd_log_' .. opt.timeString), opt)

    opt.shg= opt.shg ~= 'false'
    opt.multiTask = opt.multiTask  ~= 'false'
    opt.genOnly= opt.genOnly ~= 'false'
    opt.display = opt.display ~= 'false'
    opt.testOnly = opt.testOnly ~= 'false'
    opt.saveOutput = opt.saveOutput ~= 'false'
    opt.saveTest = opt.saveTest ~= 'false'
    opt.tenCrop = opt.tenCrop ~= 'false'
    opt.shareGradInput = opt.shareGradInput ~= 'false'
    opt.optnet = opt.optnet ~= 'false'

    if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
        cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
    end

    return opt
end

return M
