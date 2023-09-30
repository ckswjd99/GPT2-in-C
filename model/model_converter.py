import torch
import os

INPUT_PATH = './pytorch_gpt2.bin'
OUTPUT_PATH = './GPT2-124M.mymodel'

if not os.path.exists(INPUT_PATH):
    print('Input file doesn\'t exist. Start Downloading...')
    os.system(f"wget https://huggingface.co/gpt2/resolve/main/pytorch_model.bin -O {INPUT_PATH}")
if os.path.exists(OUTPUT_PATH):
    print('Output file already exists. Deleting it...')
    os.system ("rm " + OUTPUT_PATH)

def dump_tensor (path, tensor, tensor_name):
    tensor_info_string = f"TENSOR:{tensor_name}\n"
    with open(path, "a") as f:
        f.write(tensor_info_string)
    np_arr = tensor.detach().numpy()
    np_arr.astype('float32').tofile(path + ".tmp")
    data_size = os.path.getsize(path + ".tmp")
    with open(path, "a") as f:
        f.write("DATA_SIZE:" + str(data_size) + "\n")
        print("DATA_SIZE: " + str(data_size))
        f.write("DATA_START:\n")
    os.system ("cat " + path + ".tmp >> " + path)
    with open(path, "a") as f:
        f.write("DATA_END\n")
    os.system ("rm " + path + ".tmp")
    print(f"Dumped {tensor_name} {np_arr.shape} to {path}")
    with open(path, "a") as f:
        f.write("TENSOR_END\n")

state_dict = torch.load(INPUT_PATH)

wte = state_dict['wte.weight']
wpe = state_dict['wpe.weight']

decoder_blocks = []
for i in range(12):
    block = {}

    block['ln1_w'] = state_dict[f'h.{i}.ln_1.weight']
    block['ln1_b'] = state_dict[f'h.{i}.ln_1.bias']

    block['attn_wq'], block['attn_wk'], block['attn_wv'] = torch.split(
        state_dict[f'h.{i}.attn.c_attn.weight'], 768, dim=1
    )
    block['attn_bq'], block['attn_bk'], block['attn_bv'] = torch.split(
        state_dict[f'h.{i}.attn.c_attn.bias'], 768, dim=0
    )
    block['attn_wo'] = state_dict[f'h.{i}.attn.c_proj.weight']
    block['attn_bo'] = state_dict[f'h.{i}.attn.c_proj.bias']
    
    block['attn_wq'] = block['attn_wq'].permute(1, 0)
    block['attn_wk'] = block['attn_wk'].permute(1, 0)
    block['attn_wv'] = block['attn_wv'].permute(1, 0)
    block['attn_wo'] = block['attn_wo'].permute(1, 0)

    block['ln2_w'] = state_dict[f'h.{i}.ln_2.weight']
    block['ln2_b'] = state_dict[f'h.{i}.ln_2.bias']

    block['ffn1_w'] = state_dict[f'h.{i}.mlp.c_fc.weight'].permute(1, 0)
    block['ffn1_b'] = state_dict[f'h.{i}.mlp.c_fc.bias']
    block['ffn2_w'] = state_dict[f'h.{i}.mlp.c_proj.weight'].permute(1, 0)
    block['ffn2_b'] = state_dict[f'h.{i}.mlp.c_proj.bias']

    decoder_blocks.append(block)

ln_final_w = state_dict['ln_f.weight']
ln_final_b = state_dict['ln_f.bias']

with open(OUTPUT_PATH, "w") as f:
    f.write("NUM_TENSOR:" + str(2+12*16+2) + "\n")

dump_tensor(OUTPUT_PATH, wte, 'wte')
dump_tensor(OUTPUT_PATH, wpe, 'wpe')

for i, block in enumerate(decoder_blocks):
    dump_tensor(OUTPUT_PATH, block['ln1_w'], f'dblock_{i}.ln1_w')
    dump_tensor(OUTPUT_PATH, block['ln1_b'], f'dblock_{i}.ln1_b')
    dump_tensor(OUTPUT_PATH, block['attn_wq'], f'dblock_{i}.attn_wq')
    dump_tensor(OUTPUT_PATH, block['attn_wk'], f'dblock_{i}.attn_wk')
    dump_tensor(OUTPUT_PATH, block['attn_wv'], f'dblock_{i}.attn_wv')
    dump_tensor(OUTPUT_PATH, block['attn_wo'], f'dblock_{i}.attn_wo')
    dump_tensor(OUTPUT_PATH, block['attn_bq'], f'dblock_{i}.attn_bq')
    dump_tensor(OUTPUT_PATH, block['attn_bk'], f'dblock_{i}.attn_bk')
    dump_tensor(OUTPUT_PATH, block['attn_bv'], f'dblock_{i}.attn_bv')
    dump_tensor(OUTPUT_PATH, block['attn_bo'], f'dblock_{i}.attn_bo')
    dump_tensor(OUTPUT_PATH, block['ln2_w'], f'dblock_{i}.ln2_w')
    dump_tensor(OUTPUT_PATH, block['ln2_b'], f'dblock_{i}.ln2_b')
    dump_tensor(OUTPUT_PATH, block['ffn1_w'], f'dblock_{i}.ffn1_w')
    dump_tensor(OUTPUT_PATH, block['ffn1_b'], f'dblock_{i}.ffn1_b')
    dump_tensor(OUTPUT_PATH, block['ffn2_w'], f'dblock_{i}.ffn2_w')
    dump_tensor(OUTPUT_PATH, block['ffn2_b'], f'dblock_{i}.ffn2_b')

dump_tensor(OUTPUT_PATH, ln_final_w, 'ln_f_w')
dump_tensor(OUTPUT_PATH, ln_final_b, 'ln_f_b')