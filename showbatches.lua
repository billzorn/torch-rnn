require 'torch'

require 'util.DataLoader'
require 'util.StreamLoader'

local utils = require 'util.utils'
local unpack = unpack or table.unpack
local utf8 = require 'lua-utf8'

local cmd = torch.CmdLine()

cmd:option('-vocab', '')
cmd:option('-input_h5', '')
cmd:option('-stream_cmd', '')
cmd:option('-unk', '\x1a')
cmd:option('-batch_size', 50)
cmd:option('-seq_length', 50)
cmd:option('-length', 3)
cmd:option('-verbose', 1)

local opt = cmd:parse(arg)

local function print_seq(idx_to_token, seqs, i, n)
  for j = 1, n do
    local token = idx_to_token[seqs[i][j]]
    if token == '\n' then 
      io.write('\\n')
    else
      io.write(token)
    end
  end
  io.write('\n')
end

local function show_batch(idx_to_token, x_seqs, y_seqs)
  for i = 1, opt.batch_size do
    print('seq', i, 'x')
    print_seq(idx_to_token, x_seqs, i, opt.seq_length)
    if opt.verbose >= 3 then print(x_seqs[i]) end
    if opt.verbose >= 2 then
      print('seq', i, 'y')
      if idx_to_token[x_seqs[i][1]] == '\n' then io.write('  ') else io.write(' ') end
      print_seq(idx_to_token, y_seqs, i, opt.seq_length)
      if opt.verbose >= 3 then print(y_seqs[i]) end
    end
    io.write('\n')
  end
end

local vocab = utils.read_json(opt.vocab)

-- currently copied from train.lua; could be standardized in utils or something
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
-- we have no reason to keep a pristine set of options, yes?
opt.idx_to_token = idx_to_token

if opt.verbose >= 1 then
  print('read vocabulary from:', opt.vocab)
  print('  vocab size:', vocab_size)
  print('  unk:', opt.unk, utf8.byte(opt.unk), 'in vocab?', found_unk)
  if opt.verbose >= 2 then print(idx_to_token) end
end

-- implement later maybe
--local loader = nil
--if opt.input_h5 ~= '' then loader = DataLoader(opt) end

local streamer = nil
if opt.stream_cmd ~= '' then streamer = StreamLoader(opt) end

local function show_stream(streamer)
  for i = 1, opt.length do
    x, y = streamer:next_batch()
    -- really should show simultaneously so it's easier to see continuity
    print('==== batch ', i, '====')
    show_batch(idx_to_token, x, y)
  end
end

if streamer ~= nil then
  show_stream(streamer)
  streamer:close_stream()
end
