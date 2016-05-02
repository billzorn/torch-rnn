require 'torch'
local unistd = require 'posix.unistd'
local signal = require 'posix.signal'
local wait = require 'posix.sys.wait'
local utf8 = require 'lua-utf8'

local utils = require 'util.utils'

local StreamLoader = torch.class('StreamLoader')

-- Opens pipes and executes a command that produces multiple (infinite) streams of data
-- that can be packaged into batches.
-- Note that the StreamLoader object forks another process when created: use 
-- StreamLoader:close_stream() to terminate it.
function StreamLoader:__init(kwargs)
  self.idx_to_token = utils.get_kwarg(kwargs, 'idx_to_token')
  self.token_to_idx = {}
  self.vocab_size = 0
  self.max_idx = 0
  for idx, token in pairs(self.idx_to_token) do
    self.token_to_idx[token] = idx
    self.vocab_size = self.vocab_size + 1
    if idx > self.max_idx then self.max_idx = idx end
  end
  self.unk = utils.get_kwarg(kwargs, 'unk')
  self.unk_idx = self.token_to_idx[self.unk]
  assert(self.unk_idx ~= nil)

  self.batch_size = utils.get_kwarg(kwargs, 'batch_size')
  self.seq_length = utils.get_kwarg(kwargs, 'seq_length')
  self.stream_cmd = utils.get_kwarg(kwargs, 'stream_cmd')

  -- decide type for output tensors
  if self.max_idx >= torch.pow(2, 32) then
    self.x_seqs = torch.LongTensor(self.batch_size, self.seq_length)
    self.y_seqs = torch.LongTensor(self.batch_size, self.seq_length)
  elseif self.max_idx >= torch.pow(2, 16) then
    self.x_seqs = torch.IntTensor(self.batch_size, self.seq_length)
    self.y_seqs = torch.IntTensor(self.batch_size, self.seq_length)
  elseif self.max_idx >= torch.pow(2, 8) then
    self.x_seqs = torch.ShortTensor(self.batch_size, self.seq_length)
    self.y_seqs = torch.ShortTensor(self.batch_size, self.seq_length)
  else
    self.x_seqs = torch.ByteTensor(self.batch_size, self.seq_length)
    self.y_seqs = torch.ByteTensor(self.batch_size, self.seq_length)
  end

  -- stream state
  self.fds = {}
  self.buffers = {}

  -- need to buffer at least this many bytes to ensure we don't
  -- run out of characters and incorrectly interpret unicode
  self.buflen = (self.seq_length + 1) * 4

  -- unpack stream command
  self.stream_args = {}
  local i = 0
  for w in self.stream_cmd:gmatch('%S+') do
    table.insert(self.stream_args, i, w)
    i = i + 1
  end

  -- open pipes for streams
  for i = 1, self.batch_size do
    local rfd, wfd = unistd.pipe()
    table.insert(self.fds, rfd)
    table.insert(self.buffers, '')
    table.insert(self.stream_args, wfd)
  end

  -- fork and execute stream command
  local pid, errmsg = unistd.fork()
  if pid == nil then
    error(errmsg)
  elseif pid == 0 then
    -- child executes stream command
    assert(unistd.exec(self.stream_args[0], self.stream_args))
    -- control flow should never reach here
    error('stream failed?')
    return
  else
    -- parent needs to remember the pid of the child so it can kill it
    self.stream_pid = pid
    -- give the stream a moment to initialize and fail early if something
    -- seems to have gone wrong
    unistd.sleep(1)
    local child_pid, msg, status = wait.wait(self.stream_pid, wait.WNOHANG)
    if child_pid ~= 0 then error('stream terminated abnormally') end
    return
  end
end

function StreamLoader:close_stream()
  signal.kill(self.stream_pid, signal.SIGQUIT)
  local child_pid, msg, status = wait.wait(self.stream_pid)
  return child_pid, msg, status
end

-- all this iteration is probably really slow; it's not clear if it can be
-- replaced with torch tensor trickery, or if it even matters compared to
-- training speed
function StreamLoader:next_batch()
  for i, fd in ipairs(self.fds) do
    local buf = self.buffers[i]
    local buflen = buf:len()

    -- make sure buffer has enough characters
    while buflen < self.buflen do
      buf = buf .. unistd.read(fd, self.buflen - buflen)
      buflen = buf:len()
    end

    -- transcibe codepoints into output tensors
    local j = 1
    for idx, cp in utf8.codes(buf) do
      local symbol = self.token_to_idx[utf8.char(cp)]
      if symbol == nil then symbol = self.unk_idx end

      if j <= self.seq_length then self.x_seqs[i][j] = symbol end
      if j > 1 then self.y_seqs[i][j-1] = symbol end

      if j <= self.seq_length then
	j = j + 1
      else
	buf = buf:sub(idx, buflen)
	break
      end
    end

    -- store new buffer
    self.buffers[i] = buf
  end

  return self.x_seqs, self.y_seqs
end
