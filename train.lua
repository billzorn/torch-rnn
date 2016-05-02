require 'torch'
require 'nn'
require 'optim'

require 'LanguageModel'
require 'util.DataLoader'
require 'util.StreamLoader'

local utils = require 'util.utils'
local unpack = unpack or table.unpack
local utf8 = require 'lua-utf8'

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-vocab', '')
cmd:option('-input_h5', '')
cmd:option('-stream_cmd', '')
cmd:option('-unk', '\x1a')
cmd:option('-batch_size', 50)
cmd:option('-seq_length', 50)

-- Model options
cmd:option('-init_from', '')
cmd:option('-reset_training_history', 1)
cmd:option('-reset_training_position', 1)
cmd:option('-model_type', 'lstm')
cmd:option('-wordvec_size', 64)
cmd:option('-rnn_size', 128)
cmd:option('-num_layers', 2)
cmd:option('-dropout', 0)
cmd:option('-batchnorm', 0)

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-max_batches', 10000)
cmd:option('-learning_rate', 2e-3)
cmd:option('-grad_clip', 5)
cmd:option('-lr_decay_epochs', 5)
cmd:option('-lr_decay_batches', 1000)
cmd:option('-lr_decay_factor', 0.5)

-- Output options
cmd:option('-print_every', 1)
cmd:option('-eval_val_every', 1000)
cmd:option('-checkpoint_every', 1000)
cmd:option('-checkpoint_name', 'cv/checkpoint')
cmd:option('-verbose', 1)

-- Benchmark options
cmd:option('-speed_benchmark', 0)
cmd:option('-memory_benchmark', 0)

-- Backend options
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')

local opt = cmd:parse(arg)


-- Set up GPU stuff
local dtype = 'torch.FloatTensor'
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  dtype = 'torch.CudaTensor'
  if opt.verbose >= 1 then
    print(string.format('Running with CUDA on GPU %d', opt.gpu))
  end
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  -- Memory benchmarking is only supported in CUDA mode
  -- TODO: Time benchmarking is probably wrong in OpenCL mode.
  require 'cltorch'
  require 'clnn'
  cltorch.setDevice(opt.gpu + 1)
  dtype = torch.Tensor():cl():type()
  if opt.verbose >= 1 then
    print(string.format('Running with OpenCL on GPU %d', opt.gpu))
  end
else
  -- Memory benchmarking is only supported in CUDA mode
  opt.memory_benchmark = 0
  if opt.verbose >= 1 then
    print 'Running in CPU mode'
  end
end


-- The vocab has idx_to_token and token_to_idx; I'm honestly not sure why we clone them
local vocab = utils.read_json(opt.vocab)

-- we can take this opportunity to make sure the unknown character is in our vocabulary
local idx_to_token = {}
local found_unk = false
local vocab_size = 0
for k, v in pairs(vocab.idx_to_token) do
  idx_to_token[tonumber(k)] = v
  if v == opt.unk then found_unk = true end
  vocab_size = vocab_size + 1
end
if not found_unk then
   local unk_idx = #idx_to_token
   local inserted_unk = false
   while not inserted_unk do
      if idx_to_token[unk_idx] == nil then
	 idx_to_token[unk_idx] = opt.unk
	 inserted_unk = true
      else
	 unk_idx = unk_idx + 1
      end
   end
end
-- This final idx_to_token is now the official interface for our language model;
-- the language model will derive its own token_to_idx from it.

-- copy of opt, but with idx_to_token in it
local opt_vocab = torch.deserialize(torch.serialize(opt))
opt_vocab.idx_to_token = idx_to_token

-- create a fixed data loader, if we've specified an input h5 file
local loader = nil
if opt.input_h5 ~= '' then loader = DataLoader(opt) end
-- create a stream loader if we've specified a command
local streamer = nil
if opt.stream_cmd ~= '' then streamer = StreamLoader(opt_vocab) end

if loader == nil and streamer == nil then
  print('No data specified, aborting')
  os.exit(1)
end

if opt.verbose >= 2 then
  print('read vocabulary from:', opt.vocab)
  print('  vocab size:', vocab_size)
  print('  unk:', opt.unk, utf8.byte(opt.unk), 'in vocab?', found_unk)
  if streamer ~= nil then
    print('streaming training data using:', opt.stream_cmd)
  elseif loader ~= nil then
    print('read training data from:', opt.input_h5)
  end
  if loader ~= nil then
    print('read validation data from:', opt.input_h5)
  end
end

-- Set up some variables we will use below
local N, T = opt.batch_size, opt.seq_length
local train_loss_history = {}
local val_loss_history = {}
local val_loss_history_it = {}
local forward_backward_times = {}
local init_memory_usage, memory_usage = nil, {}

-- Initialize the model and criterion
local model = nil
local last_resumed_batch = 0
if opt.init_from ~= '' then
  if opt.verbose >= 2 then
    print('Initializing from ', opt.init_from)
  end
  local checkpoint = torch.load(opt.init_from)
  model = checkpoint.model:type(dtype)
  last_resumed_batch = checkpoint.i
  -- Recover model-related parameters from the loaded checkpoint
  opt.model_type = checkpoint.opt.model_type
  opt.wordvec_dim = checkpoint.opt.wordvec_dim
  opt.rnn_size = checkpoint.opt.rnn_size
  opt.num_layers = checkpoint.opt.num_layers
  opt.dropout = checkpoint.opt.dropout
  opt.batchnorm = checkpoint.opt.batchnorm
  -- Optionally recover training history
  if opt.reset_training_history == 0 then
    train_loss_history = checkpoint.train_loss_history
    val_loss_history = checkpoint.val_loss_history
    val_loss_history_it = checkpoint.val_loss_history_it
    forward_backward_times = checkpoint.forward_backward_times
    memory_usage = checkpoint.memory_usage
  end
else
  model = nn.LanguageModel(opt_vocab):type(dtype)
end
local params, grad_params = model:getParameters()
local crit = nn.CrossEntropyCriterion():type(dtype)

if opt.memory_benchmark == 1 then
  -- This should only be enabled in GPU mode
  assert(cutorch)
  cutorch.synchronize()
  local free, total = cutorch.getMemoryUsage(cutorch.getDevice())
  init_memory_usage = total - free
end

-- Loss function that we pass to an optim method
local function f(w)
  assert(w == params)
  grad_params:zero()

  -- Get a minibatch and run the model forward, maybe timing it
  local timer
  local x, y
  if streamer ~= nil then
    x, y = streamer:next_batch()
  else
    x, y = loader:nextBatch('train')
  end
  x, y = x:type(dtype), y:type(dtype)
  if opt.speed_benchmark == 1 then
    if cutorch then cutorch.synchronize() end
    timer = torch.Timer()
  end
  local scores = model:forward(x)

  -- Use the Criterion to compute loss; we need to reshape the scores to be
  -- two-dimensional before doing so. Annoying.
  local scores_view = scores:view(N * T, -1)
  local y_view = y:view(N * T)
  local loss = crit:forward(scores_view, y_view)

  -- Run the Criterion and model backward to compute gradients, maybe timing it
  local grad_scores = crit:backward(scores_view, y_view):view(N, T, -1)
  model:backward(x, grad_scores)
  if timer then
    if cutorch then cutorch.synchronize() end
    local time = timer:time().real
    if opt.verbose >= 1 then
      print('Forward / Backward pass took ', time)
    end
    table.insert(forward_backward_times, time)
  end

  -- Maybe record memory usage
  if opt.memory_benchmark == 1 then
    assert(cutorch)
    if cutorch then cutorch.synchronize() end
    local free, total = cutorch.getMemoryUsage(cutorch.getDevice())
    local memory_used = total - free - init_memory_usage
    local memory_used_mb = memory_used / 1024 / 1024
    if opt.verbose >= 1 then
      print(string.format('Using %dMB of memory', memory_used_mb))
    end
    table.insert(memory_usage, memory_used)
  end

  if opt.grad_clip > 0 then
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end

  return loss, grad_params
end

-- Skip batches that have already been trained on by a resumed checkpoint
local first_batch = 1
if opt.reset_training_position == 0 then
  if opt.verbose >= 2 then
    print('Skipping batches:', last_resumed_batch)
  end
  while first_batch <= last_resumed_batch do
    if streamer ~= nil then streamer:next_batch() end
    if loader ~= nil then loader:nextBatch('train') end
    first_batch = first_batch + 1
  end
end

-- Train the model!
local optim_config = {learningRate = opt.learning_rate}
local num_train = 0
if loader ~= nil then num_train = loader.split_sizes['train'] end
local num_iterations = 0
if streamer ~= nil then
  num_iterations = opt.max_batches
elseif loader ~= nil then
  num_iterations = opt.max_epochs * num_train
end

if opt.verbose >= 2 then
  print('Begin training: batch', first_batch, 'to', num_iterations)
end

model:training()
for i = first_batch, num_iterations do

  -- Epochs don't make sense if we have provided a stream for training
  if streamer ~= nil then
    -- Maybe decay learning rate
    if opt.lr_decay_batches > 0 and i % opt.lr_decay_batches == 0 then
      local old_lr = optim_config.learningRate
      optim_config = {learningRate = old_lr * opt.lr_decay_factor}
      if opt.verbose >= 1 then
	print('decayed learning rate to', old_lr * opt.lr_decay_factor)
      end
    end
  else
    local epoch = math.floor(i / num_train) + 1
    -- Check if we are at the end of an epoch
    if i % num_train == 0 then
      model:resetStates() -- Reset hidden states

      -- Maybe decay learning rate
      if opt.lr_decay_epochs > 0 and epoch % opt.lr_decay_epochs == 0 then
	local old_lr = optim_config.learningRate
	optim_config = {learningRate = old_lr * opt.lr_decay_factor}
	if opt.verbose >= 1 then
	  print('decayed learning rate to', old_lr * opt.lr_decay_factor)
	end
      end
    end
  end

  -- Take a gradient step and maybe print
  -- Note that adam returns a singleton array of losses
  local _, loss = optim.adam(f, params, optim_config)
  table.insert(train_loss_history, loss[1])
  if opt.verbose >= 1 and opt.print_every > 0 and i % opt.print_every == 0 then
    if streamer ~= nil then
      local msg = 'Streaming, i = %d / %d, loss = %f'
      local args = {msg, i, num_iterations, loss[1]}
      print(string.format(unpack(args)))
    else
      local float_epoch = i / num_train
      local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f'
      local args = {msg, float_epoch, opt.max_epochs, i, num_iterations, loss[1]}
      print(string.format(unpack(args)))
    end
  end

  -- Maybe evaluate val loss
  if opt.eval_val_every > 0 and (i % opt.eval_val_every == 0 or i == num_iterations) then
    -- Evaluate loss on the validation set. Note that we reset the state of
    -- the model; this might happen in the middle of an epoch, but that
    -- shouldn't cause too much trouble. Seems like the alternative is cloning the entire model?
    model:evaluate()
    model:resetStates()
    local num_val = loader.split_sizes['val']
    local val_loss = 0
    for j = 1, num_val do
      local xv, yv = loader:nextBatch('val')
      xv = xv:type(dtype)
      yv = yv:type(dtype):view(N * T)
      local scores = model:forward(xv):view(N * T, -1)
      val_loss = val_loss + crit:forward(scores, yv)
    end
    val_loss = val_loss / num_val
    if opt.verbose >= 1 then
      print('val_loss =', val_loss)
    end
    table.insert(val_loss_history, val_loss)
    table.insert(val_loss_history_it, i)
    model:resetStates()
    model:training()
    -- not clear whether we need to make this call; in the original code it always happened
    -- after evaluating val loss when we saved the checkpoint
    model:clearState()
    -- if we do make it, I think we'll have to do this as well
    params, grad_params = model:getParameters()
  end

  -- Maybe save a checkpoint
  if (opt.checkpoint_every > 0 and i % opt.checkpoint_every == 0) or i == num_iterations then
    -- First save a JSON checkpoint, excluding the model
    local checkpoint = {
      opt = opt,
      train_loss_history = train_loss_history,
      val_loss_history = val_loss_history,
      val_loss_history_it = val_loss_history_it,
      forward_backward_times = forward_backward_times,
      memory_usage = memory_usage,
      i = i
    }
    local filename = string.format('%s_%d.json', opt.checkpoint_name, i)
    if opt.verbose >= 1 then
      print('saving json checkpoint:', filename)
    end
    -- Make sure the output directory exists before we try to write it
    paths.mkdir(paths.dirname(filename))
    utils.write_json(filename, checkpoint)

    -- Now save a torch checkpoint with the model.
    -- You can either make both of these calls, or neither. I'm not sure what the ramifications
    -- of omitting them are for the saved checkpoints; it seems better not to interfere with
    -- the state during training any more than we have to, though it will apparently
    -- massively increase the size of checkpoints on disk since we're saving all kinds of 
    -- intermediate data.
    model:resetStates()
    model:clearState()
    -- Cast the model to float before saving so it can be used on CPU
    model:float()
    checkpoint.model = model
    local filename = string.format('%s_%d.t7', opt.checkpoint_name, i)
    if opt.verbose >= 1 then
      print('saving torch checkpoint:', filename)
    end
    paths.mkdir(paths.dirname(filename))
    torch.save(filename, checkpoint)
    model:type(dtype)
    params, grad_params = model:getParameters()
    collectgarbage()
  end

end

if streamer ~= nil then streamer:close_stream() end

if opt.verbose >= 2 then
  print('Training finished.')
end
