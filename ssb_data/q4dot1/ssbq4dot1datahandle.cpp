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
            uint32_t tag = -1;
            if(!strcmp(token, "\"ALGERIA\"")) {
                tag = 0;
            } 
            else if(!strcmp(token, "\"ARGENTINA\"")) {
                tag = 1;
            } 
            else if(!strcmp(token, "\"BRAZIL\"")) {
                tag = 2;
            } 
            else if(!strcmp(token, "\"CANADA\"")) {
                tag = 3;
            }
            else if(!strcmp(token, "\"EGYPT\"")) {
                tag = 4;
            }
            else if(!strcmp(token, "\"ETHIOPIA\"")) {
                tag = 5;
            }
            else if(!strcmp(token, "\"FRANCE\"")) {
                tag = 6;
            }
            else if(!strcmp(token, "\"GERMANY\"")) {
                tag = 7;
            }
            else if(!strcmp(token, "\"INDIA\"")) {
                tag = 8;
            }
            else if(!strcmp(token, "\"INDONESIA\"")) {
                tag = 9;
            }
            else if(!strcmp(token, "\"IRAN\"")) {
                tag = 10;
            }
            else if(!strcmp(token, "\"IRAQ\"")) {
                tag = 11;
            }
            else if(!strcmp(token, "\"JAPAN\"")) {
                tag = 12;
            }
            else if(!strcmp(token, "\"JORDAN\"")) {
                tag = 13;
            }
            else if(!strcmp(token, "\"KENYA\"")) {
                tag = 14;
            }
            else if(!strcmp(token, "\"MOROCCO\"")) {
                tag = 15;
            }
            else if(!strcmp(token, "\"MOZAMBIQUE\"")) {
                tag = 16;
            }
            else if(!strcmp(token, "\"PERU\"")) {
                tag = 17;
            }
            else if(!strcmp(token, "\"CHINA\"")) {
                tag = 18;
            }
            else if(!strcmp(token, "\"ROMANIA\"")) {
                tag = 19;
            }
            else if(!strcmp(token, "\"SAUDI ARABIA\"")) {
                tag = 20;
            }
            else if(!strcmp(token, "\"VIETNAM\"")) {
                tag = 21;
            }
            else if(!strcmp(token, "\"RUSSIA\"")) {
                tag = 22;
            }
            else if(!strcmp(token, "\"UNITED KINGDOM\"")) {
                tag = 23;
            }
            else if(!strcmp(token, "\"UNITED STATES\"")) {
                tag = 24;
            }
            outputdata_group[i][count] = tag;
        }
    }

    for(uint32_t i = 0; i < SCAN_NUM; i++) {
        char *token = strtok(nullptr, delimiter);
        if(i == 2) {
            char* token1 = (char*)malloc(2);
            strncpy(token1, token + 6, 1);
            token1[1] = '\0';
            outputdata_rtscan[i][count] = atoi(token1);
        }
        else {
            uint32_t tag = -1;
            if(!strcmp(token, "\"AFRICA\"")) {
                tag = 0;
            } 
            else if(!strcmp(token, "\"AMERICA\"")) {
                tag = 1;
            } 
            else if(!strcmp(token, "\"ASIA\"")) {
                tag = 2;
            } 
            else if(!strcmp(token, "\"EUROPE\"")) {
                tag = 3;
            }
            else if(!strcmp(token, "\"MIDDLE EAST\"")) {
                tag = 4;
            }
            outputdata_rtscan[i][count] = tag;
        }
    }
    count++;
}

int main(){
    FILE *inputfile,*outputfile,*outputfile1;
    char line[MAX_LINE_SIZE];
    uint32_t *outputdata_rtscan[3];
    uint32_t *outputdata_avg[1];
    uint32_t *outputdata_group[2];
    uint32_t count = 0;

    outputdata_rtscan[0] = (uint32_t *)malloc(N * sizeof(uint32_t));
    outputdata_rtscan[1] = (uint32_t *)malloc(N * sizeof(uint32_t));
    outputdata_rtscan[2] = (uint32_t *)malloc(N * sizeof(uint32_t));

    outputdata_avg[0] = (uint32_t *)malloc(N * sizeof(uint32_t));

    outputdata_group[0] = (uint32_t *)malloc(N * sizeof(uint32_t));
    outputdata_group[1] = (uint32_t *)malloc(N * sizeof(uint32_t));

    //open the input file and output file
    inputfile = fopen("file_q4dot1.csv", "r");
    if(inputfile == nullptr) {
        perror("Error open input file\n");
        return 1;
    }
    outputfile = fopen("data.txt", "wb");
    if(outputfile == nullptr) {
        perror("Error open output file\n");
        return 1;
    }
    outputfile1 = fopen("outputfile_rtscan_q4dot1.txt", "wb");
    if(outputfile1 == nullptr) {
        perror("Error open output file\n");
        return 1;
    }

    //read input csv file line by line and handle
    while(fgets(line, sizeof(line), inputfile) != nullptr) {
        processCSVLine(line, outputdata_avg, outputdata_rtscan, outputdata_group, count);
    }

    // printf("%lf %lf %d %d %d %d", outputdata_avg[0][0], outputdata_avg[0][6001214], outputdata_group[1][0], outputdata_group[1][6001214], outputdata_rtscan[2][0], outputdata_rtscan[2][6001214]);
    
    fwrite(outputdata_avg[0], sizeof(uint32_t), N, outputfile);

    fwrite(outputdata_group[0], sizeof(uint32_t), N, outputfile);
    fwrite(outputdata_group[1], sizeof(uint32_t), N, outputfile);

    fwrite(outputdata_rtscan[0], sizeof(uint32_t), N, outputfile);
    fwrite(outputdata_rtscan[1], sizeof(uint32_t), N, outputfile);
    fwrite(outputdata_rtscan[2], sizeof(uint32_t), N, outputfile);

    fclose(inputfile);
    fclose(outputfile);

    fwrite(outputdata_rtscan[0], sizeof(uint32_t), N, outputfile1);
    fwrite(outputdata_rtscan[1], sizeof(uint32_t), N, outputfile1);
    fwrite(outputdata_rtscan[2], sizeof(uint32_t), N, outputfile1);
    fclose(outputfile1);

    return 0;
}