require 'torch'
require 'nn'

require 'LanguageModel'
local utf8 = require 'lua-utf8'

local cmd = torch.CmdLine()
cmd:option('-checkpoint', '')
cmd:option('-seed', 0)
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
if opt.verbose == 1 then io.stderr:write(msg, '\n') end

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
  io.stderr:write('------------------------------------------------------------\n')
  io.stderr:write('Interactive language model reading commands from stdin\n')
  io.stderr:write(seed_descr .. ' seed: ', seed, ' temperature: ', opt.temperature, '\n')
  io.stderr:write('------------------------------------------------------------\n')
  io.stderr:flush()
end

model:sampling(seed, opt.temperature)

local running = true
local input_string = io.input():read('*a')
local command = '?'

local i = 0
while running and input_string ~= '' do
  for idx, cp in utf8.codes(input_string) do
    local token = utf8.char(cp)
    if command == '?' then
      i = i + 1
      if token == 'x' then
	running = false
	break
      elseif token == 'o' or token == 'q' or token == 'p' then
	command = token
      elseif token == 'g' then
	local s = model:generate(1)
	io.write(s, '\n')
      else
	io.stderr:write('unknown command: ', token, ' ', cp)
	io.write('\n')
      end
    elseif command == 'o' then
      model:observe(token)
      io.write('\n')
      command = '?'
    elseif command == 'p' then
      local prob = model:prob(token)
      io.write(string.format('%.16e', prob), '\n')
      command = '?'
    elseif command == 'q' then
      local logprob = model:logprob(token)
      io.write(string.format('%.16e', logprob), '\n')
      command = '?'
    else
      io.stderr:write('bad command state: ', command)
      io.write('\n')
      command = '?'
    end
    
  end	 
  input_string = io.input():read('*a')
end

if opt.verbose == 1 then io.stderr:write('done, processed ', i, ' commands\n') end
