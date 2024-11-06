raydb-build = ./build
optix-lib = $(raydb-build)/lib

raydb-src = ./src/raydb
raydb-srcs = $(raydb-src)/raydb.cu $(raydb-src)/raydb.cpp $(raydb-src)/raydb.h $(raydb-src)/timer.h $(raydb-src)/group.h 

ifndef BUILD_TYPE
	BUILD_TYPE = Release
endif

raydb: $(raydb-srcs)
	cd $(raydb-build) && \
	cmake ../src/ -D CMAKE_C_COMPILER=/usr/bin/gcc-8 -D CMAKE_BUILD_TYPE=$(BUILD_TYPE) && \
	make

clean:
	rm -rf ./bin/* $(raydb-build)/*