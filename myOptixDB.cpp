//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "myOptixDB.h"
#include "timer.h"
#include "group.h"

#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <set>
#include <float.h>
#include <unistd.h>


#include <sutil/Camera.h>
#include <sutil/Trackball.h>

#define THREAD_NUM 20

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RangeRecord
{
    int minAvgValue = INT_MAX;
    int maxAvgValue = INT_MIN;
    int minGroupValue = INT_MAX;
    int maxGroupValue = INT_MIN;
    int minScanValue  = INT_MAX;
    int maxScanValue  = INT_MIN;

    void modifyAvg(int avgvalue) {
        if(avgvalue < minAvgValue) {
            minAvgValue = avgvalue;
        }
        if(avgvalue > maxAvgValue) {
            maxAvgValue = avgvalue;
        }
    }

    void modifyGroup(int groupvalue) {
        if(groupvalue < minGroupValue) {
            minGroupValue = groupvalue;
        }
        if(groupvalue > maxGroupValue) {
            maxGroupValue = groupvalue;
        }
    }

    void modifyScan(int scanvalue) {
        if(scanvalue < minScanValue) {
            minScanValue = scanvalue;
        }
        if(scanvalue > maxScanValue) {
            maxScanValue = scanvalue;
        }
    }
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

//
//variable
//
Timer                   timer_;
int **d_groupBuffer = nullptr;
int **tmpGroupBuffer = nullptr;

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

RangeRecord inputDataHandle(std::vector<float3>& vertices, FILE *inputData, int* dimCounts, int data_num, int interval_x, int interval_y) {
    int *avgbuffer[MAX_AVG_NUM];
    int *groupbuffer[MAX_GROUP_NUM];
    int *scanbuffer[MAX_SCAN_NUM];
    RangeRecord rr;
    // float half_interval = (float)interval_x / 2;

    for(int i = 0;i < dimCounts[0]; i++){
        avgbuffer[i] = (int *)malloc(sizeof(int) * data_num);
        fread(avgbuffer[i], sizeof(int), data_num, inputData);
    }
    for(int i = 0;i < dimCounts[1]; i++){
        groupbuffer[i] = (int *)malloc(sizeof(int) * data_num);
        fread(groupbuffer[i], sizeof(int), data_num, inputData);
    }
    for(int i = 0;i < dimCounts[2]; i++){
        scanbuffer[i] = (int *)malloc(sizeof(int) * data_num);
        fread(scanbuffer[i], sizeof(int), data_num, inputData);
    }
    mapGroups(groupbuffer, dimCounts[1], data_num);
    for(int i = 0;i < dimCounts[2]; i++) {
        mapPredicate(scanbuffer[i], i, data_num);
    }
    for(int i = 0; i < data_num; i++) {
        int p1 = avgbuffer[0][i];
        rr.modifyAvg(p1);
        Groups groups;
        groups.groupnum = dimCounts[1];
        for(int j = 0; j < dimCounts[1]; j++) {
            groups.groupvector[j] = groupbuffer[j][i];
        }
        // int p2 = getGroupFromGroupsMap(groups);
        int p2 = groupbuffer[0][i];
        rr.modifyGroup(p2);
        int p3;
        int predicates[MAX_SCAN_NUM];
        for(int j = 0; j < dimCounts[2]; j++) {
            predicates[j] = predicatemap[j][scanbuffer[j][i]];
        }
        predicateMerge(predicates, dimCounts[2], p3);
        rr.modifyScan(p3);
        // vertices.push_back({(float)p1 + half_interval, (float)p2, (float)p3});
        // vertices.push_back({(float)p1 - half_interval, (float)p2 - 0.5f, (float)p3 - 0.5f});
        // vertices.push_back({(float)p1 - half_interval, (float)p2 + 0.5f, (float)p3 + 0.5f});
        vertices.push_back({(float)p1, (float)p2, (float)p3});
        vertices.push_back({(float)p1 + 2 * interval_x, (float)p2, (float)p3});
        vertices.push_back({(float)p1, (float)p2 + 2 * interval_y, (float)p3});
    }
    for(int i = 0; i < dimCounts[1] - 1; i++) {
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &tmpGroupBuffer[i] ), sizeof( int ) * data_num ) );
            CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( tmpGroupBuffer[i] ),
                    groupbuffer[i + 1], sizeof( int ) * data_num,
                    cudaMemcpyHostToDevice
                    ) );
    }
    return rr;
}

void inputPredicateHandle(std::ifstream &inputPredicate, int predicateCount, int** &scanRange, int* &scanType) {
    scanRange = new int*[predicateCount];
    for(int i = 0; i < predicateCount; i++) {
        scanRange[i] = new int[10];
        std::string str;
        std::getline(inputPredicate, str);
        std::stringstream ss(str);
        std::string token;
        int tmpCount = 0;
        while (std::getline(ss, token, ',')) {
            scanRange[i][tmpCount++] = std::stoi(token);
        }
    }
    scanType = new int[predicateCount];
    std::string str;
    std::getline(inputPredicate, str);
    std::stringstream ss(str);
    std::string token;
    int tmpCount = 0;
    while (std::getline(ss, token, ',')) {
        scanType[tmpCount++] = std::stoi(token);
    }
}

int main( int argc, char* argv[] )
{
    int         width;
    int         height;
    int         depth = 1;
    int         dimCounts[3] = {1,1,1};
    int         data_num = 0;
    int         **scanRange;
    int         *scanType;
    char        inputDataPath[256] = "\0";
    char        inputPredicatePath[256] = "\0";
    // int         interval;
    int         interval_x;
    int         interval_y;
    int         resultbufferLength;
    bool        complexAvg = false;

    char opt;
    while ((opt = getopt(argc, argv, "n:g:p:s:i:x:y:a")) != -1) {
        switch(opt){
            case 'n':
                data_num = atoi(optarg);
                break;
            case 'g':
                dimCounts[1] = atoi(optarg);
                break;
            case 'p':
                dimCounts[2] = atoi(optarg);
                break;
            case 'i':
                strcpy(inputDataPath, optarg);
                break;
            case 's':
                strcpy(inputPredicatePath, optarg);
                break;
            case 'x':
                interval_x = stoi(optarg);
                break;
            case 'y':
                interval_y = stoi(optarg);
                break;
            case 'a':
                complexAvg = true;
                break;
            default:
                exit(-1);
        }
    }

    tmpGroupBuffer = new int*[dimCounts[1] - 1];

    try
    {
        char log[2048]; // For error reporting from OptiX creation functions

        std::vector<float3> vertices;
        FILE *inputData = fopen(inputDataPath, "rb");
        RangeRecord rr = inputDataHandle(vertices, inputData, dimCounts, data_num, interval_x, interval_y);

        int *extraAvgBuffer = nullptr;
        if(complexAvg) {
            extraAvgBuffer = (int *)malloc(sizeof(int) * data_num);
            fread(extraAvgBuffer, sizeof(int), data_num, inputData);
        }

        std::ifstream inputPredicate(inputPredicatePath);
        inputPredicateHandle(inputPredicate, dimCounts[2], scanRange, scanType);

        timer_.commonGetStartTime(0);

        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            fprintf(stdout,"[execute] Initialize CUDA and create OptiX context begin...\n");
            // Initialize CUDA
            CUDA_CHECK( cudaFree( 0 ) );

            // Initialize the OptiX API, loading all API entry points
            OPTIX_CHECK( optixInit() );

            // Specify context options
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction       = &context_log_cb;
            options.logCallbackLevel          = 4;

            // Associate a CUDA context (and therefore a specific GPU) with this
            // device context
            CUcontext cuCtx = 0;  // zero means take the current context
            OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &context ) );
            fprintf(stdout,"[execute] Initialize CUDA and create OptiX context done\n");
        }

        //
        // accel handling
        //
        OptixTraversableHandle gas_handle;
        CUdeviceptr            d_gas_output_buffer;
        int primCount = 0;
        {
            fprintf(stdout,"[execute] Accel handling begin...\n");
            // Use default options for simplicity.  In a real use case we would want to
            // enable compaction, etc
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
            accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

            const size_t vertices_size = sizeof( float3 )*vertices.size();
            primCount = vertices.size() / 3;
            // std::cout << "primCount:" << primCount << std::endl;
            CUdeviceptr d_vertices=0;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_vertices ), vertices_size ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( d_vertices ),
                        vertices.data(),
                        vertices_size,
                        cudaMemcpyHostToDevice
                        ) );

            // Our build input is a simple list of non-indexed triangle vertices
            const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL }; // !!! ensure the correctness of accumulation !!!
            OptixBuildInput triangle_input = {};
            triangle_input.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangle_input.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.numVertices   = static_cast<uint32_t>( vertices.size() );
            triangle_input.triangleArray.vertexBuffers = &d_vertices;
            triangle_input.triangleArray.flags         = triangle_input_flags;
            triangle_input.triangleArray.numSbtRecords = 1;

            OptixAccelBufferSizes gas_buffer_sizes;
            OPTIX_CHECK( optixAccelComputeMemoryUsage(
                        context,
                        &accel_options,
                        &triangle_input,
                        1, // Number of build inputs
                        &gas_buffer_sizes
                        ) );
            CUdeviceptr d_temp_buffer_gas;
            CUDA_CHECK( cudaMalloc(
                        reinterpret_cast<void**>( &d_temp_buffer_gas ),
                        gas_buffer_sizes.tempSizeInBytes
                        ) );
            CUDA_CHECK( cudaMalloc(
                        reinterpret_cast<void**>( &d_gas_output_buffer ),
                        gas_buffer_sizes.outputSizeInBytes
                        ) );

            OPTIX_CHECK( optixAccelBuild(
                        context,
                        0,                  // CUDA stream
                        &accel_options,
                        &triangle_input,
                        1,                  // num build inputs
                        d_temp_buffer_gas,
                        gas_buffer_sizes.tempSizeInBytes,
                        d_gas_output_buffer,
                        gas_buffer_sizes.outputSizeInBytes,
                        &gas_handle,
                        nullptr,            // emitted property list
                        0                   // num emitted properties
                        ) );

            // We can now free the scratch space buffer used during build and the vertex
            // inputs, since they are not needed by our trivial shading method
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer_gas ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_vertices        ) ) );
            fprintf(stdout,"[execute] Accel handling done\n");
        }

        //
        // Create module
        //
        OptixModule module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            fprintf(stdout,"[execute] Create module begin...\n");
            OptixModuleCompileOptions module_compile_options = {};
            module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

            pipeline_compile_options.usesMotionBlur        = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipeline_compile_options.numPayloadValues      = 3;
            pipeline_compile_options.numAttributeValues    = 2;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
            pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

            const std::string ptx = sutil::getPtxString( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "myOptixDB.cu" );
            size_t sizeof_log = sizeof( log );

            OPTIX_CHECK_LOG( optixModuleCreateFromPTX(
                        context,
                        &module_compile_options,
                        &pipeline_compile_options,
                        ptx.c_str(),
                        ptx.size(),
                        log,
                        &sizeof_log,
                        &module
                        ) );
            fprintf(stdout,"[execute] Create module done\n");
        }

        //
        // Create program groups
        //
        OptixProgramGroup raygen_prog_group   = nullptr;
        OptixProgramGroup miss_prog_group     = nullptr;
        OptixProgramGroup hitgroup_prog_group = nullptr;
        {
            fprintf(stdout,"[execute] Create program groups begin...\n");
            OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc    = {}; //
            raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module            = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
            size_t sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &raygen_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &raygen_prog_group
                        ) );

            OptixProgramGroupDesc miss_prog_group_desc  = {};
            miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            miss_prog_group_desc.miss.module            = module;
            miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
            sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &miss_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &miss_prog_group
                        ) );

            OptixProgramGroupDesc hitgroup_prog_group_desc = {};
            hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            hitgroup_prog_group_desc.hitgroup.moduleAH            = module;
            hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
            sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixProgramGroupCreate(
                        context,
                        &hitgroup_prog_group_desc,
                        1,   // num program groups
                        &program_group_options,
                        log,
                        &sizeof_log,
                        &hitgroup_prog_group
                        ) );
            fprintf(stdout,"[execute] Create program groups done\n");
        }

        //
        // Link pipeline
        //
        OptixPipeline pipeline = nullptr;
        {
            fprintf(stdout,"[execute] Link pipeline begin...\n");
            const uint32_t    max_trace_depth  = 1;
            OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth          = max_trace_depth;
            pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            size_t sizeof_log = sizeof( log );
            OPTIX_CHECK_LOG( optixPipelineCreate(
                        context,
                        &pipeline_compile_options,
                        &pipeline_link_options,
                        program_groups,
                        sizeof( program_groups ) / sizeof( program_groups[0] ),
                        log,
                        &sizeof_log,
                        &pipeline
                        ) );

            OptixStackSizes stack_sizes = {};
            for( auto& prog_group : program_groups )
            {
                OPTIX_CHECK( optixUtilAccumulateStackSizes( prog_group, &stack_sizes ) );
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                                     0,  // maxCCDepth
                                                     0,  // maxDCDEpth
                                                     &direct_callable_stack_size_from_traversal,
                                                     &direct_callable_stack_size_from_state, &continuation_stack_size ) );
            OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                                    direct_callable_stack_size_from_state, continuation_stack_size,
                                                    1  // maxTraversableDepth
                                                    ) );
            fprintf(stdout,"[execute] Link pipeline done\n");
        }

        //
        // Set up shader binding table
        //
        OptixShaderBindingTable sbt = {};
        {
            fprintf(stdout,"[execute] Set up shader binding table begin...\n");
            CUdeviceptr  raygen_record;
            const size_t raygen_record_size = sizeof( RayGenSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
            RayGenSbtRecord rg_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( raygen_record ),
                        &rg_sbt,
                        raygen_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof( MissSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &miss_record ), miss_record_size ) );
            MissSbtRecord ms_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( miss_record ),
                        &ms_sbt,
                        miss_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            CUdeviceptr hitgroup_record;
            size_t      hitgroup_record_size = sizeof( HitGroupSbtRecord );
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &hitgroup_record ), hitgroup_record_size ) );
            HitGroupSbtRecord hg_sbt;
            OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( hitgroup_record ),
                        &hg_sbt,
                        hitgroup_record_size,
                        cudaMemcpyHostToDevice
                        ) );

            sbt.raygenRecord                = raygen_record;
            sbt.missRecordBase              = miss_record;
            sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
            sbt.missRecordCount             = 1;
            sbt.hitgroupRecordBase          = hitgroup_record;
            sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
            sbt.hitgroupRecordCount         = 1;
            fprintf(stdout,"[execute] Set up shader binding table done\n");
        }

        unsigned int *d_primFlag;
        int primFlagLen = (primCount + 31) / 32;
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_primFlag ), sizeof(unsigned int) * primFlagLen) );

        int *d_extraAvgBuffer = nullptr;
        if(complexAvg) {
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_extraAvgBuffer ), sizeof( int ) * data_num) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( d_extraAvgBuffer ),
                        extraAvgBuffer, sizeof( int ) * data_num,
                        cudaMemcpyHostToDevice
                        ) );
        }

        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_groupBuffer ), sizeof( int* ) * (dimCounts[1] - 1) ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( d_groupBuffer ),
                    tmpGroupBuffer, sizeof( int* ) * (dimCounts[1] - 1),
                    cudaMemcpyHostToDevice
                    ) );

        timer_.commonGetEndTime(0);
        timer_.showTime(0, "Initialize");

        timer_.commonGetStartTime(1);

        for(int i = 0; i < dimCounts[2]; i++) {
                if(scanType[i] == 0) {
                    scanRange[i][0] = predicatemap[i][scanRange[i][0]];
                    scanRange[i][1] = predicatemap[i][scanRange[i][1]];
                }else {
                    for(int j = 0; j < scanType[i]; j++) {
                        scanRange[i][j] = predicatemap[i][scanRange[i][j]];
                    }
                }
            }

        width = (rr.maxAvgValue - rr.minAvgValue + interval_x) / interval_x + 1;
        height = (rr.maxGroupValue - rr.minGroupValue + interval_y) / interval_y + 1;
        for(int i = 0; i < dimCounts[2]; i++) {
            if(scanType[i] == 0 && i != dimCounts[2] - 1) {
                depth *= scanRange[i][1] - scanRange[i][0] + 1;
            }else if(scanType[i] != 0) {
                depth *= scanType[i];
            }
        }

        //TODO: modify here
        // resultbufferLength = ((1998-1992+1) * (5 * 5 * 40));
        // resultbufferLength = ((1998-1992+1) * 25 * 25);
        // resultbufferLength = ((1998-1992+1) * 250 * 250);
        // resultbufferLength = ((1998-1992+1) * 25);
        // resultbufferLength = ((1998-1992+1) * 25 * 25);
        resultbufferLength = ((1998-1992+1) * 250 * 1000);
        sutil::CUDAOutputBuffer<unsigned long long> output_buffer( sutil::CUDAOutputBufferType::CUDA_DEVICE, resultbufferLength , 1 );

        CUDA_CHECK(cudaMemset(output_buffer.map(), 0 , resultbufferLength * sizeof(unsigned long long)));

        //
        // launch
        //
        Params params;
        {
            fprintf(stdout,"[execute] Launch begin...\n");
            CUstream stream;
            CUDA_CHECK( cudaStreamCreate( &stream ) );
            int lastPredicateIdx = dimCounts[2] - 1;

            params.handle = gas_handle;
            params.bias = 0.5;
            params.minAvgValue = (int)rr.minAvgValue;
            params.resultValue = output_buffer.map();
            params.interval_x = interval_x;
            params.interval_y = interval_y;
            params.primFlag = d_primFlag;
            params.complexAvg = complexAvg;
            params.extraAvgBuffer = d_extraAvgBuffer;
            params.groupBuffer = d_groupBuffer;
            params.groupNum = dimCounts[1] - 1;
            // params.rayLength = scanRange[lastPredicateIdx * 2 + 1] - scanRange[lastPredicateIdx * 2];
            if(scanType[lastPredicateIdx] == 0) {
                params.rayLength = scanRange[lastPredicateIdx][1] - scanRange[lastPredicateIdx][0];
            } else {
                params.rayLength = 0;
            }

            // std::cout << "params.rayLength:" << params.rayLength << std::endl;
            
            int *rayOrigin_z = new int[depth];
            for(int i = 0; i < depth; i++) {
                int idx = i;
                int predicates[MAX_SCAN_NUM];
                
                for(int j = 0; j < MAX_SCAN_NUM; j++) {
                    predicates[j] = 0;
                }
                for(int j = dimCounts[2] - 1; j >= 0; --j){
                    if(scanType[j] == 0 && j != dimCounts[2] - 1) {
                        predicates[j] = idx % (scanRange[j][1] - scanRange[j][0] + 1);
                        idx /= (scanRange[j][1] - scanRange[j][0] + 1);
                    } else if(scanType[j] != 0){
                        predicates[j] = idx % scanType[j];
                        idx /= scanType[j];
                    }
                }
                for(int j = dimCounts[2] - 1; j >= 0;--j) {
                    // predicates[j] += scanRange[j * 2];
                    if(scanType[j] == 0) {
                        predicates[j] += scanRange[j][0];
                    } else {
                        predicates[j] = scanRange[j][predicates[j]];
                    }
                }
                predicateMerge(predicates, dimCounts[2], rayOrigin_z[i]);

                // std::cout << rayOrigin_z[i] << ' ';
            }

            // std::cout << std::endl;

            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &params.rayOrigin_z ), sizeof( int ) * depth ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( params.rayOrigin_z ),
                        rayOrigin_z, sizeof( int ) * depth,
                        cudaMemcpyHostToDevice
                        ) );
            
            CUDA_CHECK(cudaMemset(params.primFlag, 0 , sizeof(unsigned int) * primFlagLen));
            
            CUdeviceptr d_param;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( d_param ),
                        &params, sizeof( params ),
                        cudaMemcpyHostToDevice
                        ) );

            timer_.commonGetStartTime(2);

            OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, width, height, depth ) );
            CUDA_SYNC_CHECK();

            timer_.commonGetEndTime(2);
            timer_.commonGetEndTime(1);

            output_buffer.unmap();

            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_param ) ) );

            std::cout << width << ' ' << height << ' ' << depth << std::endl;
            fprintf(stdout,"[execute] Launch done\n");
        }

        //
        // Display results
        //
        {
            fprintf(stdout,"[execute] Display results begin...\n");
            unsigned long long* resultValue = output_buffer.getHostPointer();

            timer_.showTime(1, "Launch(Prepare included)");
            timer_.showTime(2, "Launch");
            timer_.showTime(3, "Handle Groups(what groups should be launched)");
            timer_.clear();

            std::cout << "---------------------------------------------------" << std::endl;
            fprintf(stdout,"Result below:\n");
            std::vector<int> newGroups = {0,0};
            int tmpcount = 0;
            // int sum = 0;
            for(int i = 0;i < resultbufferLength;++i)
            {
                Groups groups;  
                groups = getGroupsFromGroupsMapInverse(i);
                
                if(resultValue[i] != 0){
                    for(int j = 0; j < groups.groupnum; j ++) {
                        std::cout << groups.groupvector[j] << " ";
                    }
                    std::cout  << resultValue[i] << ' ' << std::endl;
                    // sum += resultValue[i];
                    tmpcount++;
                }   
            }
            // std::cout << "Sum : " << sum << std::endl;
            std::cout << "Line Num : " << tmpcount << std::endl;
            std::cout << "---------------------------------------------------" << std::endl;
            fprintf(stdout,"[execute] Display results done\n");
        }
        
        //
        // Cleanup
        //
        {
            fprintf(stdout,"[execute] Cleanup begin...\n");
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.raygenRecord       ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.missRecordBase     ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( sbt.hitgroupRecordBase ) ) );
            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_gas_output_buffer    ) ) );

            OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
            OPTIX_CHECK( optixProgramGroupDestroy( hitgroup_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( miss_prog_group ) );
            OPTIX_CHECK( optixProgramGroupDestroy( raygen_prog_group ) );
            OPTIX_CHECK( optixModuleDestroy( module ) );

            OPTIX_CHECK( optixDeviceContextDestroy( context ) );
            fprintf(stdout,"[execute] Cleanup done\n");
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
