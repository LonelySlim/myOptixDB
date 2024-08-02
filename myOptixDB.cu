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

#include "myOptixDB.h"
#include <cuda/helpers.h>
#include <cuda_runtime.h>
#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    float3 ray_origin;
    // ray_origin = {idx.x * params.interval + params.minAvgValue, idx.y, params.rayOrigin_z[idx.z] - params.bias};
    ray_origin = {idx.x * params.interval_x + params.minAvgValue, idx.y * params.interval_y, params.rayOrigin_z[idx.z] - params.bias};
    
    float3 ray_direction = {0,0,1};

    float rayLength = params.rayLength + 2 * params.bias;

    // Trace the ray against our scene hierarchy
    unsigned int p = 0;

    optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            0.0f,                // Min intersection distance
            rayLength,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p);
    
    //atomicAdd(&params.resultValue[idx.y] , (unsigned long long)p);

}


extern "C" __global__ void __miss__ms()
{
    //do nothing
}


extern "C" __global__ void __closesthit__ch()
{
    //do nothing
}

extern "C" __global__ void __anyhit__ah()
{
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int primIdx = optixGetPrimitiveIndex();
    float3 vertices[3];
    optixGetTriangleVertexData(params.handle, primIdx, optixGetSbtGASIndex(), 0.0f, vertices);
    // float resultValue = vertices[0].x - params.interval / 2;
    unsigned long long resultValue = params.complexAvg ? (unsigned long long)vertices[0].x * params.extraAvgBuffer[primIdx] : (unsigned long long)vertices[0].x;
    int resultIndex = (int)vertices[0].y;

    //optixSetPayload_0(optixGetPayload_0() + (unsigned)resultValue);
    //optixSetPayload_0(optixGetPayload_0() + 1);

    int flagIndex = primIdx / 32;
    int flagOffset = 31 - primIdx % 32;
    unsigned int flagMask = 1 << flagOffset;
    unsigned int flag = atomicOr(&params.primFlag[flagIndex], flagMask);
    if(!(flagMask & flag)) {
        int scanFlag = 1;
        for(int i = 0; i < params.scanNum; i++) {
            if(params.scanType[i] == 0) {
                if(!(params.scanBuffer[i][primIdx] >= params.scanRange[i][0] && params.scanBuffer[i][primIdx] <= params.scanRange[i][1])) {
                    scanFlag = 0;
                }
            }else {
                int tmpFlag = 1;
                for(int j = 0; j < params.scanType[i]; j++) {
                    if(params.scanBuffer[i][primIdx] == params.scanRange[i][j]) {
                        tmpFlag = 0;
                    }
                }
                if(tmpFlag) {
                    scanFlag = 0;
                }
            }
        }
        if(scanFlag)
            atomicAdd(&params.resultValue[resultIndex] , (unsigned long long)resultValue);
    }
    optixIgnoreIntersection();
}
