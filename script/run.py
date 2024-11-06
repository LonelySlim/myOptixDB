import time
import os
import sys
import numpy as np

def raydb_ssbflat_singlequery(query_name, dataset_size, interval_x, interval_y, group_num, predicate_num, predicate_file, input_file, attr_tag):
    output_file = f"log/{query_name}-raydb.log"
    args = f'-n {dataset_size} -x {interval_x} -y {interval_y} -g {group_num} -p {predicate_num} -s {predicate_file} -i {input_file}'
    if attr_tag == 1:
        args += ' -a'
    cmd = f"./build/bin/raydb {args} >> {output_file}"
    print(cmd)
    os.system(cmd)

def raydb_ssbflat():
    # output_file = f"log/raydb-{queryname}.log"
    os.system('make clean')
    os.system('make raydb')



raydb_ssbflat()
    