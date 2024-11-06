#include<stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cstdint>

#define MAX_LINE_SIZE 1024
#define AVG_FUNC_NUM 1
#define SCAN_NUM 3
#define GROUP_NUM 2
#define N  119994608

void processCSVLine(char *line, uint32_t **outputdata_avg, uint32_t **outputdata_rtscan, uint32_t **outputdata_group, uint32_t &count) {
    char *delimiter = ",";
    char **endptr;

    for(uint32_t i = 0; i < AVG_FUNC_NUM; i++) {
        char *token = strtok(line, delimiter);
        uint32_t ttoken = atoi(token);
        outputdata_avg[i][count] = ttoken;
    }

    for(uint32_t i = 0; i < GROUP_NUM; i++) {
        char *token = strtok(nullptr, delimiter);
        if(i == 0) {
            uint32_t ttoken = atoi(token);
            outputdata_group[i][count] = ttoken;
        }
        else {
            for(uint32_t k = 0; token[k] != '\0'; k++) {
                if(token[k] == '\"') {
                    token[k] = '\0';
                }
            }
            uint32_t ttoken = atoi(token + 6);
            outputdata_group[i][count] = ttoken;
        }
        
    }

    for(uint32_t i = 0; i < SCAN_NUM; i++) {
        if(i == 0) {
            char *token = strtok(nullptr, delimiter);
            for(uint32_t k = 0; token[k] != '\0'; k++) {
                if(token[k] == '\"') {
                    token[k] = '\0';
                }
            }
            uint32_t ttoken = atoi(token + 6);
            outputdata_rtscan[i][count] = ttoken;
        }
        else if(i == 1) {
            char *token = strtok(nullptr, delimiter);
            uint32_t tag = -1;
            if(!strcmp(token, "\"AFRICA\"\n")) {
                tag = 0;
            } 
            else if(!strcmp(token, "\"AMERICA\"\n")) {
                tag = 1;
            } 
            else if(!strcmp(token, "\"ASIA\"\n")) {
                tag = 2;
            } 
            else if(!strcmp(token, "\"EUROPE\"\n")) {
                tag = 3;
            }
            else if(!strcmp(token, "\"MIDDLE EAST\"\n")) {
                tag = 4;
            }
            outputdata_rtscan[i][count] = tag;
        } else {
            outputdata_rtscan[i][count] = 0;
        }
    }
    count++;
}

int main(){
    FILE *inputfile,*outputfile, *outputfile1;
    char line[MAX_LINE_SIZE];
    uint32_t *outputdata_rtscan[3];
    uint32_t *outputdata_avg[1];
    uint32_t *outputdata_group[2];
    uint32_t count = 0;
    uint32_t *pad = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; ++i) {
        pad[i] = 0;
    } 

    outputdata_rtscan[0] = (uint32_t *)malloc(N * sizeof(uint32_t));
    outputdata_rtscan[1] = (uint32_t *)malloc(N * sizeof(uint32_t));
    outputdata_rtscan[2] = (uint32_t *)malloc(N * sizeof(uint32_t));

    outputdata_avg[0] = (uint32_t *)malloc(N * sizeof(uint32_t));

    outputdata_group[0] = (uint32_t *)malloc(N * sizeof(uint32_t));
    outputdata_group[1] = (uint32_t *)malloc(N * sizeof(uint32_t));

    //open the input file and output file
    inputfile = fopen("../q2dot1/file_q2dot1.csv", "r");
    if(inputfile == nullptr) {
        perror("Error open input file\n");
        return 1;
    }
    outputfile = fopen("data.txt", "wb");
    if(outputfile == nullptr) {
        perror("Error open output_rtdb file\n");
        return 1;
    }
    outputfile1 = fopen("outputfile_rtscan_q2dot2.txt", "wb");
    if(outputfile1 == nullptr) {
        perror("Error open output file\n");
        return 1;
    }

    //read input csv file line by line and handle
    while(fgets(line, sizeof(line), inputfile) != nullptr) {
        processCSVLine(line, outputdata_avg, outputdata_rtscan, outputdata_group, count);
    }

    // printf("%lf %lf %d %d %d %d", outputdata_avg[0][0], outputdata_avg[0][6001214], outputdata_group[0][0], outputdata_group[0][6001214], outputdata_rtscan[0][0], outputdata_rtscan[0][6001214]);
    
    fwrite(outputdata_avg[0], sizeof(uint32_t), N, outputfile);

    fwrite(outputdata_group[0], sizeof(uint32_t), N, outputfile);
    fwrite(outputdata_group[1], sizeof(uint32_t), N, outputfile);

    fwrite(outputdata_rtscan[1], sizeof(uint32_t), N, outputfile);
    fwrite(outputdata_group[1], sizeof(uint32_t), N, outputfile);

    fclose(inputfile);
    fclose(outputfile);

    fwrite(outputdata_rtscan[1], sizeof(uint32_t), N, outputfile1);
    fwrite(outputdata_group[1], sizeof(uint32_t), N, outputfile1);
    fwrite(pad, sizeof(uint32_t), N, outputfile1);
    fclose(outputfile1);

    return 0;
}