--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local ImagenetDataset = torch.class('resnet.ImagenetDataset', M)

function ImagenetDataset:__init(imageInfo, opt, split)
    self.imageInfo = imageInfo[split]
    self.opt = opt
    self.split = split
    self.dir = paths.concat(self.opt.data, split)
    local key_point_file = paths.cwd() .. '/datasets/cub-annot/' .. self.split .. ".dat"
    self.name_pt = torch.load(key_point_file)

    assert(paths.filep(key_point_file), 'file does not exist: ' .. key_point_file)
    assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function ImagenetDataset:get(i)
    local path = ffi.string(self.imageInfo.imagePath[i]:data())
    self.imageName = path
    local image = self:_loadImage(paths.concat(self.dir, path))
    local class = self.imageInfo.imageClass[i]

    local image_name = path:split('/')
    local key_points = self.name_pt[image_name[#image_name]]
    local out = torch.zeros(self.opt.keypointNum, self.opt.outputRes, self.opt.outputRes)
    local visi = torch.LongTensor(self.opt.keypointNum):fill(0)

    for key, value in pairs(key_points) do
        visi[tonumber(key)] = value[3]
        if value[3] > 0 then -- Checks that there is a ground truth annotation
            self:_drawGaussian(out[tonumber(key)], value, self.opt.gaussianSize)
        end
    end


    local accumulator = false
    for key, value in pairs(key_points) do 
        if value[3] > 0 then
            accumulator = true
        end 
    end

    assert(accumulator, 'no body is bigger than 0')
    if out:max() <= 0 then
        print('out is all zero')
        assert(false)
    end

    return {
        location = path,
        input  = image,
        target = out,
        visi = visi
    }
end

function ImagenetDataset:_loadImage(path)
    local ok, input = pcall(function()
        return image.load(path, 3, 'float')
    end)

    -- Sometimes image.load fails because the file extension does not match the
    -- image format. In that case, use image.decompress on a ByteTensor.
    if not ok then
        local f = io.open(path, 'r')
        assert(f, 'Error reading: ' .. tostring(path))
        local data = f:read('*a')
        f:close()

        local b = torch.ByteTensor(string.len(data))
        ffi.copy(b:data(), data, b:size(1))

        input = image.decompress(b, 3, 'float')
    end

    return input
end


function ImagenetDataset:_drawGaussian(img, pt, sigma)
    -- Draw a 2D gaussian
    -- Check that any part of the gaussian is in-bounds

    -- convert [0, 1] back to [0, 64]
    local pt_1 = pt[1] * self.opt.outputRes
    local pt_2 = pt[2] * self.opt.outputRes

    local tmpSize = math.ceil(3*sigma)
    local ul = {math.floor(pt_1 - tmpSize), math.floor(pt_2 - tmpSize)}
    local br = {math.floor(pt_1 + tmpSize), math.floor(pt_2 + tmpSize)}
    -- If not, return the image as is
    if (ul[1] > img:size(2) or ul[2] > img:size(1) or br[1] < 1 or br[2] < 1) then 
        --print(pt[1], pt[2], tmpSize, ul[1], ul[2], img:size(2), img:size(1), br[1], br[2])
        --print(self.imageName)
        --assert(false)
        return img
    end
    -- Generate gaussian
    local size = 2*tmpSize + 1
    local g = image.gaussian(size)
    -- Usable gaussian range
    local g_x = {math.max(1, -ul[1]), math.min(br[1], img:size(2)) - math.max(1, ul[1]) + math.max(1, -ul[1])}
    local g_y = {math.max(1, -ul[2]), math.min(br[2], img:size(1)) - math.max(1, ul[2]) + math.max(1, -ul[2])}
    -- Image range
    local img_x = {math.max(1, ul[1]), math.min(br[1], img:size(2))}
    local img_y = {math.max(1, ul[2]), math.min(br[2], img:size(1))}
    assert(g_x[1] > 0 and g_y[1] > 0)
    img:sub(img_y[1], img_y[2], img_x[1], img_x[2]):cmax(g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
    return img
end

function ImagenetDataset:size()
    return self.imageInfo.imageClass:size(1)
end

-- Computed from random subset of ImageNet training images
local meanstd = {
    mean = { 0.485, 0.456, 0.406 },
    std = { 0.229, 0.224, 0.225 },
}
local pca = {
    eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
    eigvec = torch.Tensor{
        { -0.5675,  0.7192,  0.4009 },
        { -0.5808, -0.0045, -0.8140 },
        { -0.5836, -0.6948,  0.4203 },
    },
}

function ImagenetDataset:preprocess()
    if self.split == 'train' then
        return t.Compose{
            
            ---t.ColorJitter({
            ---    brightness = 0.4,
            ---    contrast = 0.4,
            ---    saturation = 0.4,
            ---}),
            ---t.Lighting(0.1, pca.eigval, pca.eigvec),
            
            t.UniformScale(256),
            t.ColorNormalize(meanstd),
        }
    elseif self.split == 'val' then
        local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
        return t.Compose{
            t.UniformScale(256),
            t.ColorNormalize(meanstd),
        }
    else
        error('invalid split: ' .. self.split)
    end
end

return M.ImagenetDataset
