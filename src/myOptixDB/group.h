#ifndef GROUP_H
#define GROUP_H

#include<vector>
#include<map>
#include<assert.h>

#define MAX_AVG_NUM 1
#define MAX_GROUP_NUM 10
#define MAX_SCAN_NUM 10

struct Groups{
    int groupvector[MAX_GROUP_NUM];
    int groupnum;

    Groups(){
        for(int i = 0; i < MAX_GROUP_NUM; i++) {
            groupvector[i] = 0;
        }
        groupnum = 0;
    }
};

bool operator<(const Groups &a, const Groups &b) {
    assert(a.groupnum == b.groupnum);
    for(int i = 0; i < a.groupnum; i++) {
        if(a.groupvector[i] < b.groupvector[i]) {
            return true;
        } else if(a.groupvector[i] > b.groupvector[i]) {
            return false;
        }
    }
    return false;
}

std::map<int,int> predicatemap[MAX_SCAN_NUM];
std::map<int,int> inversepredicatemap[MAX_SCAN_NUM];
int predicateCounts[MAX_SCAN_NUM];

std::map<Groups, int> groupsmap;
std::map<int, Groups> groupsmapinverse;

int mapGroups(int **groupbuffer,int dimCount, int data_num) {
    for(int i = 0; i < data_num; i++) {
        Groups groups;
        groups.groupnum = dimCount;
        for(int j = 0; j < dimCount; j++) {
            groups.groupvector[j] = groupbuffer[j][i];
        }
        groupsmap[groups] = 1;
    }
    int tmpcount = 0;
    for(std::map<Groups,int>::iterator it = groupsmap.begin(); it != groupsmap.end(); it++) {
        groupsmapinverse[tmpcount] = (*it).first;
        groupsmap[(*it).first] = tmpcount;
        tmpcount++;
    }
    return tmpcount;
}

int getGroupFromGroupsMap(Groups groups) {
    return groupsmap[groups];
}

Groups getGroupsFromGroupsMapInverse(int n) {
    return groupsmapinverse[n];
}

int mapPredicate(int *rowPredicate, int predicateIdx, int data_num){
    for(int i = 0; i < data_num; i++) {
        if(predicatemap[predicateIdx].find(rowPredicate[i]) == predicatemap[predicateIdx].end()) {
            predicatemap[predicateIdx][rowPredicate[i]] = 1;
        }
    }
    int count = 0;
    for(std::map<int,int>::iterator it = predicatemap[predicateIdx].begin(); it != predicatemap[predicateIdx].end(); it++) {
        predicatemap[predicateIdx][(*it).first] = count;
        inversepredicatemap[predicateIdx][count] = (*it).first;
        count++;
    }
    predicateCounts[predicateIdx] = count;
    return count;
}

void predicateMerge(int *predicates,int dimCount,int& predicate){
    predicate = 0;
    int rate = 1;
    for(int i = dimCount - 1; i >= 0;--i){
        predicate += predicates[i] * rate;
        rate *= predicateCounts[i];
    }
}

void predicateMergeInverse(int *predicates,int dimCount,int predicate){
    for(int i = dimCount - 1; i >= 0;--i){
        predicates[i] = predicate % predicateCounts[i];
        predicate /= predicateCounts[i];
    }
}

#endif