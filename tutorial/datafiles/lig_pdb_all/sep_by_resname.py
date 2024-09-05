import os
import sys

ion_list = ["SO4", "PO4", "ACT", "MGF", "BR3", "NO3", "NA", "MG", "CL", "CA", "MN", "BR"]
smol_list = ["EDO", "EPE", "GOL", "PO4", "PG4", "DMS", "PGE", "MSE", "CSD", "CSO", "TPO", \
               "ACE", "KCX", "ALY", "PTR", "OCS", "DTT", "FMT", "SGM"]


def split_by_res(fn_path):
    fn = fn_path.split('/')[-1] 
    path = '/'.join(fn_path.split('/')[:-1])
    if len(path) > 0:
        path += '/'
    pdb = fn.split('_')[0]
    chain = fn.split('.')[0].split('_')[-1]
    res_dict = {}
    with open(fn_path, 'r') as f1:
        lines = f1.readlines()
        for line in lines:
            num = line[22:26].strip()
            res = line[17:20].strip()
            if res_dict.get(num) is None and res not in ion_list + smol_list:
                res_dict[num] = res
            
    
    for num,res in res_dict.items(): 
        with open(path+"%s_%s_%s_%s.pdb"%(pdb,chain,res,num), "w") as f1:
            print(path+"%s_%s_%s_%s.pdb"%(pdb,chain,res,num))
            for line in lines:
                if line[17:20].strip() == res and line[22:26].strip() == num:
#                    print(line)
                    f1.write(line)

if __name__ == '__main__':
    fn_path = sys.argv[1]
    split_by_res(fn_path)
    


