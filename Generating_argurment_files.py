import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--files", type=int, help="NODEs")
parser.add_argument("--GPU_per_node", type=int, help="NODEs")
args = parser.parse_args()

number_of_node = args.files
number_of_GPU_per_node = args.GPU_per_node

alpha_1 = [2,4,8]

LIST = []
for a1 in alpha_1:
    line = ' --N ' + str(a1)
    LIST.append(line)

LIST2 = []
for file_i in range(number_of_node):
    for gpu_ind in range(number_of_GPU_per_node):
        if file_i * number_of_GPU_per_node + gpu_ind < len(LIST):
            line = 'python ./Generate_data.py --node ' + str(file_i+1) + ' --GPU_index ' + str(gpu_ind) +  LIST[file_i * number_of_GPU_per_node + gpu_ind]
        if file_i * number_of_GPU_per_node + gpu_ind >= len(LIST):
            line = ' '
            
        LIST2.append(line)

for file_i in range(number_of_node):
    names = LIST2[file_i*number_of_GPU_per_node:(file_i+1)*number_of_GPU_per_node]
    
    with open(r'arguement_files'+str(file_i+1), 'w') as fp:
        for item in names:
            fp.write("%s\n" % item)
    fp.close()
    
    
