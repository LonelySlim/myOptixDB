#ifndef GROUP_H
#define GROUP_H

#include<vector>

void groupMerge(const std::vector<int>& groups,const std::vector<int>& dimScale,int dimCount,int& group){
    group = 0;
    int rate = 1;
    for(int i = dimCount - 1; i >= 0;--i){
        group += groups[i] * rate;
        rate *= dimScale[i];
    }
}

void groupMergeInverse(std::vector<int>& groups,const std::vector<int>& dimScale,int dimCount,int group){
    for(int i = dimCount - 1; i >= 0;--i){
        groups[i] = group % dimScale[i];
        group /= dimScale[i];
    }
}

#endif