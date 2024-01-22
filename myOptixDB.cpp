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
#include <float.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>



template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RangeRecord
{
    float minAvgValue = FLT_MAX;
    float maxAvgValue = FLT_MIN;
    int minGroupValue = INT_MAX;
    int maxGroupValue = INT_MIN;
    int minScanValue  = INT_MAX;
    int maxScanValue  = INT_MIN;

    void modifyAvg(float avgvalue) {
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

static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

// static void createVerticesArray(std::vector<float3>& vertices, std::ifstream& in,int* dimCounts, const std::vector<int>& groupDimScale)
// {
//     fprintf(stdout,"[execute] Create vertices array begin...\n");
//     int p1,p2,p3;
//     std::string line;
//     while(std::getline(in,line))
//     {
//         std::istringstream iss(line);
//         std::string element;
//         std::getline(iss, element, ' ');
//         p1 = std::stoi(element);
//         vector<int> groups;
//         for(int j = 0;j < dimCounts[1];++j){
//             std::getline(iss,element,' ');
//             groups.push_back(std::stoi(element));
//         }
//         groupMerge(groups,groupDimScale,dimCounts[1],p2);
//         std::getline(iss, element, ' ');
//         p3 = std::stoi(element);
//         // vertices.push_back({p1 + 0.5f, p2, p3});
//         // vertices.push_back({p1, p2 - 0.5f, p3});
//         // vertices.push_back({p1, p2 + 0.5f, p3});
//         vertices.push_back({p1 + 0.5f, (float)p2, (float)p3});
//         vertices.push_back({p1 - 0.5f, p2 - 0.5f, p3 - 0.5f});
//         vertices.push_back({p1 - 0.5f, p2 + 0.5f, p3 + 0.5f});
//     }
//     fprintf(stdout,"[execute] Create vertices array done\n");
// }

RangeRecord inputfileHandle(std::vector<float3>& vertices, FILE *inputfile, int* dimCounts, std::vector<int>& groupDimScale, int data_num) {
    double *avgbuffer[MAX_AVG_NUM];
    int *groupbuffer[MAX_GROUP_NUM];
    int *scanbuffer[MAX_SCAN_NUM];
    RangeRecord rr;

    for(int i = 0;i < dimCounts[0]; i++){
        avgbuffer[i] = (double *)malloc(sizeof(double) * data_num);
        fread(avgbuffer[i], sizeof(double), data_num, inputfile);
    }
    for(int i = 0;i < dimCounts[1]; i++){
        groupbuffer[i] = (int *)malloc(sizeof(int) * data_num);
        fread(groupbuffer[i], sizeof(int), data_num, inputfile);
        //groupDimScale[i] = mapGroup(groupbuffer[i], i, data_num);
        //printf("\nDIM:%d\n", groupDimScale[i]);
    }
    for(int i = 0;i < dimCounts[2]; i++){
        scanbuffer[i] = (int *)malloc(sizeof(int) * data_num);
        fread(scanbuffer[i], sizeof(int), data_num, inputfile);
    }
    mapGroups(groupbuffer, dimCounts[1], data_num);
    for(int i = 0; i < data_num; i++) {
        double p1 = avgbuffer[0][i];
        rr.modifyAvg((float)p1);
        // int p2;
        // vector<int> groups;
        // for(int j = 0;j < dimCounts[1];++j){
        //     groups.push_back(groupbuffer[j][i]);
        //     //groups.push_back(inversegroupmap[j][groupbuffer[j][i]]);
        //     //printf("\n%d\n", groups[j]);
        // }
        // groupMerge(groups,groupDimScale,dimCounts[1],p2);
        Groups groups;
        groups.groupnum = dimCounts[1];
        for(int j = 0; j < dimCounts[1]; j++) {
            groups.groupvector[j] = groupbuffer[j][i];
        }
        int p2 = getGroupFromGroupsMap(groups);
        rr.modifyGroup(p2);
        int p3 = scanbuffer[0][i];
        rr.modifyScan(p3);
        vertices.push_back({(float)p1 + 500.0f, (float)p2, (float)p3});
        vertices.push_back({(float)p1 - 500.0f, (float)p2 - 0.5f, (float)p3 - 0.5f});
        vertices.push_back({(float)p1 - 500.0f, (float)p2 + 0.5f, (float)p3 + 0.5f});
    }
    return rr;
}

int main( int argc, char* argv[] )
{
    int         width;
    int         height;
    int         depth = 1;
    int         dimCounts[3] = {1,2,1};
    vector<int> groupDimScale = {150000,25};
    bool        useBitmap = true;
    FILE *bitmapfile = fopen("/home/sxr/resultbitmapq10.txt", "rb");
    int         data_num = 6001215;
    unsigned int *bitmap = (unsigned int *)malloc(((data_num + 31) >> 5) * sizeof(unsigned int));
    fread(bitmap, sizeof(unsigned int), (data_num + 31) >> 5,bitmapfile);
    fclose(bitmapfile);

    try
    {
        char log[2048]; // For error reporting from OptiX creation functions


        std::vector<float3> vertices;
        //std::ifstream in("/home/sxr/rtdb/SDK/optixDB/tools/generateData/uniform_data_100000000.0_10.txt");
        // std::ifstream in("/home/sxr/rtdb/SDK/myOptixDB/tools/data/uniform_data_100000000.0_10_2.txt");
        // if(!in.is_open())
        // {
        //     std::cerr << "can not open file outputdata.txt !" << std::endl;
        //     return 1;
        // }
        // createVerticesArray(vertices, in, dimCounts, groupDimScale);
        // in.close();
        FILE *inputfile = fopen("/home/sxr/outputfile_rtdb_q10.txt", "rb");
        RangeRecord rr = inputfileHandle(vertices, inputfile, dimCounts, groupDimScale, data_num);


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
        {
            fprintf(stdout,"[execute] Accel handling begin...\n");
            // Use default options for simplicity.  In a real use case we would want to
            // enable compaction, etc
            OptixAccelBuildOptions accel_options = {};
            accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
            accel_options.operation  = OPTIX_BUILD_OPERATION_BUILD;

            // Triangle build input: simple list of three vertices
            // const std::vector<float3> vertices =
            // { 
            //       { 5.5f, 5.0f, 5.0f },
            //       { 5.0f, 5.5f, 5.0f },
            //       { 5.0f, 4.5f, 5.0f }
            // };

            const size_t vertices_size = sizeof( float3 )*vertices.size();
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

        timer_.commonGetEndTime(0);
        timer_.showTime(0, "Initialize");

        timer_.commonGetStartTime(1);

        width = ((int)rr.maxAvgValue + 1 - (int)rr.minAvgValue + 999) / 1000 + 1;
        // height = 1;
        // for(int i = 0; i < groupDimScale.size(); i++) {
        //     height *= groupDimScale[i];
        // }
        // height = 250000;
        height = groupsmap.size();

        sutil::CUDAOutputBuffer<float> output_buffer_0( sutil::CUDAOutputBufferType::CUDA_DEVICE, height , 1 );
        sutil::CUDAOutputBuffer<int> output_buffer_1( sutil::CUDAOutputBufferType::CUDA_DEVICE, height , 1 );
        // sutil::CUDAOutputBuffer<float> output_buffer_0( sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height);
        // sutil::CUDAOutputBuffer<int> output_buffer_1( sutil::CUDAOutputBufferType::CUDA_DEVICE, width, height);

        CUDA_CHECK(cudaMemset(output_buffer_0.map(), 0 , height * sizeof(float)));
        CUDA_CHECK(cudaMemset(output_buffer_1.map(), 0 , height * sizeof(int)));

        //
        // launch
        //
        Params params;
        {
            fprintf(stdout,"[execute] Launch begin...\n");
            CUstream stream;
            CUDA_CHECK( cudaStreamCreate( &stream ) );


            params.handle = gas_handle;
            params.bias = 1e-5;
            params.rayMode = std::stoi(argv[2]);
            params.maxSelectValue = (int)rr.maxAvgValue + 1;
            params.minSelectValue = (int)rr.minAvgValue;
            params.maxGroupbyValue = rr.maxGroupValue;
            params.minGroupbyValue = rr.minGroupValue;
            params.maxWhereValue = std::stoi(argv[1]);
            params.minWhereValue = rr.minScanValue;
            params.resultValue = output_buffer_0.map();
            params.resultCount = output_buffer_1.map();
            if(params.rayMode == 0){
                depth = (params.maxWhereValue - params.minWhereValue + 2) / 2;
                params.rayLength = 1.0f;
                if((params.maxWhereValue - params.minWhereValue + 2) % 2 == 0){
                    params.rayLastLength = 1e-5;
                }else{
                    params.rayLastLength = params.rayLength;
                }
            }else if(params.rayMode == 1){
                params.rayLength = params.maxWhereValue - params.minWhereValue;
                params.rayLastLength = params.maxWhereValue - params.minWhereValue;
            }
            params.enableBitmap = useBitmap;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &params.bitmap ), sizeof(unsigned int) * ((data_num + 31) >> 5) ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( params.bitmap ),
                        bitmap, sizeof(unsigned int) * ((data_num + 31) >> 5),
                        cudaMemcpyHostToDevice
                        ) );

            CUdeviceptr d_param;
            CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_param ), sizeof( Params ) ) );
            CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( d_param ),
                        &params, sizeof( params ),
                        cudaMemcpyHostToDevice
                        ) );

            timer_.commonGetStartTime(2);

            std::cout << width << ' ' << height << ' ' << depth << std::endl;

            OPTIX_CHECK( optixLaunch( pipeline, stream, d_param, sizeof( Params ), &sbt, width, height, depth ) );
            CUDA_SYNC_CHECK();

            timer_.commonGetEndTime(2);
            

            output_buffer_0.unmap();
            output_buffer_1.unmap();

            CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_param ) ) );
            fprintf(stdout,"[execute] Launch done\n");
        }

        //
        // Display results
        //
        {
            fprintf(stdout,"[execute] Display results begin...\n");
            float* resultValue = output_buffer_0.getHostPointer();
            int* resultCount = output_buffer_1.getHostPointer();

            timer_.commonGetEndTime(1);
            timer_.showTime(1, "Launch(Prepare included)");
            timer_.showTime(2, "Launch");
            timer_.clear();

            std::cout << "---------------------------------------------------" << std::endl;
            fprintf(stdout,"Result below:\n");
            std::vector<int> newGroups = {0,0};
            int tmpcount = 0;
            for(int i = 0;i < height;++i)
            {
                // groupMergeInverse(newGroups,groupDimScale,dimCounts[1],i);
                // if(resultCount[i] != 0){
                //     std::cout << newGroups[0] << ' ' << newGroups[1] << ' ' << resultValue[i] << ' ' << resultCount[i] << ' ' << (resultValue[i])/resultCount[i] << std::endl;
                //     //std::cout << inversegroupmap[0][newGroups[0]] << ' ' << inversegroupmap[1][newGroups[1]] << ' ' << resultValue[i] << ' ' << resultCount[i] << ' ' << (resultValue[i])/resultCount[i] << std::endl;
                //     tmpcount++;
                // }   
                Groups groups = getGroupsFromGroupsMapInverse(i);
                if(resultCount[i] != 0){
                    for(int j = 0; j < groups.groupnum; j ++) {
                        std::cout << groups.groupvector[j] << " ";
                    }
                    std::cout  << resultValue[i] << ' ' << resultCount[i] << ' ' << (resultValue[i])/resultCount[i] << std::endl;
                    //std::cout << inversegroupmap[0][newGroups[0]] << ' ' << inversegroupmap[1][newGroups[1]] << ' ' << resultValue[i] << ' ' << resultCount[i] << ' ' << (resultValue[i])/resultCount[i] << std::endl;
                    tmpcount++;
                }   
            }
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
