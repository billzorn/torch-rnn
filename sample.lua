require 'torch'
require 'nn'

require 'LanguageModel'


local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'cv/checkpoint_1000.t7')
cmd:option('-seed', 0)
cmd:option('-length', 1000)
cmd:option('-batch_size', 200)
cmd:option('-start_text', '')
cmd:option('-temperature', 1)
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')
cmd:option('-verbose', 0)
local opt = cmd:parse(arg)


local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

local msg
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  model:cuda()
  msg = string.format('Running with CUDA on GPU %d', opt.gpu)
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  require 'cltorch'
  require 'clnn'
  model:cl()
  msg = string.format('Running with OpenCL on GPU %d', opt.gpu)
else
  msg = 'Running in CPU mode'
end
if opt.verbose == 1 then print(msg) end

local seed = 0
local seed_descr = ''
if opt.seed == 0 then
  seed = torch.seed()
  seed_descr = 'random'
else
  seed = opt.seed
  seed_descr = 'manual'
end

if opt.verbose == 1 then
  io.write('------------------------------------------------------------\n')
  print('Sampling ' .. opt.length .. ' characters from ' .. opt.checkpoint)
  print(seed_descr .. ' seed:', seed, 'temperature:', opt.temperature)
  if opt.start_text ~= '' then
    print('Starting with text: ', opt.start_text)
  end
  io.write('------------------------------------------------------------\n')
  io.flush()
end

model:sampling(seed, opt.temperature)
if opt.start_text ~= '' then
  model:observe(opt.start_text)
  io.write(opt.start_text)
  io.flush()
end

local remaining = opt.length
while remaining > 0 do
  local output = ''
  if remaining > opt.batch_size then
    output = model:generate(opt.batch_size)
    remaining = remaining - opt.batch_size
  else
    output = model:generate(remaining)
    remaining = remaining - remaining
  end
  io.write(output)
  io.flush()
end

io.write('\n')
io.flush()
