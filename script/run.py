import time
import os
import sys
import numpy as np

def raydb_ssbflat_singlequery(query_name, dataset_size, interval_x, interval_y, group_num, predicate_num, predicate_file, input_file, attr_tag=0):
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
    raydb_ssbflat_singlequery('q11', 119994608, 6, 1, 1, 3, '/data/sxr/ssb_data/q1dot1/predicate.txt', '/data/sxr/ssb_data/q1dot1/data.txt', 1)
    raydb_ssbflat_singlequery('q12', 119994608, 100, 1, 1, 3, '/data/sxr/ssb_data/q1dot2/predicate.txt', '/data/sxr/ssb_data/q1dot2/data.txt', 1)
    raydb_ssbflat_singlequery('q13', 119994608, 200, 1, 1, 4, '/data/sxr/ssb_data/q1dot3/predicate.txt', '/data/sxr/ssb_data/q1dot3/data.txt', 1)
    raydb_ssbflat_singlequery('q21', 119994608, 2000, 14, 2, 2, '/data/sxr/ssb_data/q2dot1/predicate.txt', '/data/sxr/ssb_data/q2dot1/data.txt')
    raydb_ssbflat_singlequery('q22', 119994608, 2000, 350, 2, 2, '/data/sxr/ssb_data/q2dot2/predicate.txt', '/data/sxr/ssb_data/q2dot2/data.txt')
    raydb_ssbflat_singlequery('q23', 119994608, 2000, 140, 2, 2, '/data/sxr/ssb_data/q2dot3/predicate.txt', '/data/sxr/ssb_data/q2dot2/data.txt')
    raydb_ssbflat_singlequery('q31', 119994608, 50, 875, 3, 3, '/data/sxr/ssb_data/q3dot1/predicate.txt', '/data/sxr/ssb_data/q3dot1/data.txt')
    raydb_ssbflat_singlequery('q32', 119994608, 500, 43750, 3, 3, '/data/sxr/ssb_data/q3dot2/predicate.txt', '/data/sxr/ssb_data/q3dot2/data.txt')
    raydb_ssbflat_singlequery('q33', 119994608, 20000, 4375, 3, 3, '/data/sxr/ssb_data/q3dot3/predicate.txt', '/data/sxr/ssb_data/q3dot3/data.txt')
    raydb_ssbflat_singlequery('q34', 119994608, 20000, 8750, 3, 3, '/data/sxr/ssb_data/q3dot4/predicate.txt', '/data/sxr/ssb_data/q3dot4/data.txt')
    raydb_ssbflat_singlequery('q41', 119994608, 100, 18, 2, 3, '/data/sxr/ssb_data/q4dot1/predicate.txt', '/data/sxr/ssb_data/q4dot1/data.txt')
    raydb_ssbflat_singlequery('q42', 119994608, 500, 88, 3, 4, '/data/sxr/ssb_data/q4dot2/predicate.txt', '/data/sxr/ssb_data/q4dot2/data.txt')
    raydb_ssbflat_singlequery('q43', 119994608, 200, 350000, 3, 3, '/data/sxr/ssb_data/q4dot3/predicate.txt', '/data/sxr/ssb_data/q4dot3/data.txt')



raydb_ssbflat()
    