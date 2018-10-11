local nnlib = cudnn
local conv = nnlib.SpatialConvolution
local batchnorm = nn.SpatialBatchNormalization
local relu = nnlib.ReLU

-- Main convolutional block
local function convBlock(numIn,numOut)
    return nn.Sequential()
        :add(batchnorm(numIn))
        :add(relu(true))
        :add(conv(numIn,numOut/2,1,1))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(conv(numOut/2,numOut/2,3,3,1,1,1,1))
        :add(batchnorm(numOut/2))
        :add(relu(true))
        :add(conv(numOut/2,numOut,1,1))
end

-- Skip layer
local function skipLayer(numIn,numOut)
    if numIn == numOut then
        return nn.Identity()
    else
        return nn.Sequential()
            :add(conv(numIn,numOut,1,1))
    end
end

local function basicBlock(numIn, numOut, kw, kh, dw, dh, padw, padh)
    kw = kw or 1
    kh = kh or 1
    dw = dw or 1
    dh = dh or 1
    padw = padw or 0
    padh = padh or 0
    local conv_block = nn.Sequential()
        :add(conv(numIn, numOut, kw, kh, dw, dh, padw, padh))
        :add(batchnorm(numOut))
        :add(relu(true))

    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(conv_block)
            :add(skipLayer(numIn, numOut)))
        :add(nn.CAddTable(true))
end

--[[
local function convBlock(numIn,numOut)
    return nn.Sequential()
        :add(basicBlock(numIn, numOut/2))
        :add(basicBlock(numOut/2, numOut/2),3,3,1,1,1,1) 
        :add(basicBlock(numOut/2, numOut)) 
end
]]

-- Residual block
function Residual(numIn,numOut)
    return nn.Sequential()
        --:add(convBlock(numIn,numOut))
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut))
            :add(skipLayer(numIn,numOut)))
        :add(nn.CAddTable(true))

end
