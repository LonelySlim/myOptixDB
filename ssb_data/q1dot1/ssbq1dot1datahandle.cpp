#include<stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cstdint>

#define MAX_LINE_SIZE 1024
#define AVG_FUNC_NUM 1
#define SCAN_NUM 3
#define GROUP_NUM 0
#define N  119994608

void processCSVLine(char *line, uint32_t **outputdata_avg, uint32_t **outputdata_scan, uint32_t **outputdata_group, uint32_t &count) {
    char *delimiter = ",";
    char **endptr;

    for(uint32_t i = 0; i < AVG_FUNC_NUM; i++) {
        char *token = strtok(line, delimiter);
        uint32_t ttoken = atoi(token);
        outputdata_avg[i][count] = ttoken;
    }

    if(GROUP_NUM == 0) {
        outputdata_group[0][count] = 0; //No grouping means all in the same group
    }
    
    for(uint32_t i = 0; i < SCAN_NUM; i++) {
        char *token = strtok(nullptr, delimiter);
        if(i == 0) {
            char cleanedStr[9];  // 8个数字字符 + 1个终止符 \0
            uint32_t j = 0;
            for(uint32_t k = 0; token[k] != '\0'; k++) {
                if(token[k] != '-') {
                    cleanedStr[j++] = token[k];
                }
            }
            cleanedStr[j] = '\0';
            uint32_t ttoken = atoi(cleanedStr);
            outputdata_scan[i][count] = ttoken / 10000;
        }
        else {
            uint32_t ttoken = atoi(token);
            outputdata_scan[i][count] = ttoken;
        }
    }

    char *token = strtok(nullptr, delimiter);
    uint32_t ttoken = atoi(token);
    outputdata_avg[1][count] = ttoken;

    count++;
}

int main(){
    FILE *inputfile, *outputfile, *outputfile1;
    char line[MAX_LINE_SIZE];
    uint32_t *outputdata_scan[3];
    uint32_t *outputdata_avg[2];
    uint32_t *outputdata_group[1];
    uint32_t count = 0;

    outputdata_scan[0] = (uint32_t *)malloc(N * sizeof(uint32_t));
    outputdata_scan[1] = (uint32_t *)malloc(N * sizeof(uint32_t));
    outputdata_scan[2] = (uint32_t *)malloc(N * sizeof(uint32_t));

    outputdata_avg[0] = (uint32_t *)malloc(N * sizeof(uint32_t));
    outputdata_avg[1] = (uint32_t *)malloc(N * sizeof(uint32_t));

    outputdata_group[0] = (uint32_t *)malloc(N * sizeof(uint32_t));

    //open the input file and output file
    inputfile = fopen("file_q1dot1.csv", "r");
    if(inputfile == nullptr) {
        perror("Error open input file\n");
        return 1;
    }
    outputfile = fopen("data.txt", "wb");
    if(outputfile == nullptr) {
        perror("Error open output file\n");
        return 1;
    }
    outputfile1 = fopen("outputfile_rtscan_q1dot1.txt", "wb");
    if(outputfile1 == nullptr) {
        perror("Error open output file\n");
        return 1;
    }

    //read input csv file line by line and handle
    while(fgets(line, sizeof(line), inputfile) != nullptr) {
        processCSVLine(line, outputdata_avg, outputdata_scan, outputdata_group, count);
    }

    // printf("%lf %lf %d %d %d %d", outputdata_avg[0][0], outputdata_avg[0][6001214], outputdata_group[0][0], outputdata_group[0][6001214], outputdata_scan[0][0], outputdata_scan[0][6001214]);

    fwrite(outputdata_avg[0], sizeof(uint32_t), N, outputfile);

    fwrite(outputdata_group[0], sizeof(uint32_t), N, outputfile);

    fwrite(outputdata_scan[0], sizeof(uint32_t), N, outputfile);
    fwrite(outputdata_scan[1], sizeof(uint32_t), N, outputfile);
    fwrite(outputdata_scan[2], sizeof(uint32_t), N, outputfile);

    fwrite(outputdata_avg[1], sizeof(uint32_t), N, outputfile);

    fclose(inputfile);
    fclose(outputfile);

    fwrite(outputdata_scan[0], sizeof(uint32_t), N, outputfile1);
    fwrite(outputdata_scan[1], sizeof(uint32_t), N, outputfile1);
    fwrite(outputdata_scan[2], sizeof(uint32_t), N, outputfile1);
    fclose(outputfile1);

    return 0;
}