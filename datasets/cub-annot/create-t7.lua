require 'image'


function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end

function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do
    lines[#lines + 1] = line
  end
  return lines
end

image_name_file = 'image-name.txt'
image_name_lines = lines_from(image_name_file)
keypoints_file = 'keypoints.txt'
keypoints_lines = lines_from(keypoints_file)
image_read_file = 'train-list.txt'
image_read_lines = lines_from(image_read_file)

name_num_dict  = {}
name_size_dict = {}
for key, line in pairs(image_name_lines) do
    local num_and_name = line:split(' ')
    local num = num_and_name[1]
    local name = num_and_name[2]
    local full_name = '/scratch/peiguo/caffe/data/cub-200/CUB_200_2011/images/' .. name
    local local_train_name = '/mv_users/peiguo/dataset/cub-fewshot/full/train/' .. name
    local local_valid_name = '/mv_users/peiguo/dataset/cub-fewshot/full/val/' .. name
    name_num_dict[full_name] = num
    
    if paths.filep(local_train_name) then
        local img = image.load(local_train_name)
        local size  = img:size()
        name_size_dict[full_name] = size
    else
        local img = image.load(local_valid_name)
        local size  = img:size()
        name_size_dict[full_name] = size
    end
    
    -- print(name, name_size_dict[full_name])

end

num_keypoint_dict = {}
for key, line in pairs(keypoints_lines) do
    local point_list = line:split(' ')
    local num = point_list[1]
    local point_num = point_list[2]
    if not num_keypoint_dict[num] then num_keypoint_dict[num] = {} end
    num_keypoint_dict[num][point_num] = {tonumber(point_list[3]), tonumber(point_list[4]), 
        tonumber(point_list[5])}
end

image_dict = {}
for key, line in pairs(image_read_lines) do
    local image_label = line:split(' ')
    local image_name = image_label[1]
    image_dict[#image_dict + 1] = image_name
end

name_keypoint_dict = {}
for key, name in pairs(image_dict) do
    local keypoint =  num_keypoint_dict[name_num_dict[name]]
    local size = name_size_dict[name]
    
    name = name:split('/')
    name = name[#name]
    
    for num, cood in pairs(keypoint) do
        cood[1] = cood[1] / size[3]
        cood[2] = cood[2] / size[2]
        
    end
    
    print(name, coord)

    name_keypoint_dict[name] = keypoint

end

--torch.save('train.dat', name_keypoint_dict)
