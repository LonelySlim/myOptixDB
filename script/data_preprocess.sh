cd ssb_data

cd q1dot1
gcc -o ssbq1dot1datahandle ssbq1dot1datahandle.cpp
./ssbq1dot1datahandle
cd ..

cd q1dot2
gcc -o ssbq1dot2datahandle ssbq1dot2datahandle.cpp
./ssbq1dot2datahandle
cd ..

cd q1dot3
gcc -o ssbq1dot3datahandle ssbq1dot3datahandle.cpp
./ssbq1dot3datahandle
cd ..

cd q2dot1
gcc -o ssbq2dot1datahandle ssbq2dot1datahandle.cpp
./ssbq2dot1datahandle
cd ..

cd q2dot2
gcc -o ssbq2dot2datahandle ssbq2dot2datahandle.cpp
./ssbq2dot2datahandle
cd ..

cp q2dot2/outputfile_rtscan_q2dot2.txt q2dot3/outputfile_rtscan_q2dot3.txt
cp q2dot2/data.txt q2dot3/data.txt

cd q3dot1
gcc -o ssbq3dot1datahandle ssbq3dot1datahandle.cpp
./ssbq3dot1datahandle
cd ..

cd q3dot2
gcc -o ssbq3dot2datahandle ssbq3dot2datahandle.cpp
./ssbq3dot2datahandle
cd ..

cd q3dot3
gcc -o ssbq3dot3datahandle ssbq3dot3datahandle.cpp
./ssbq3dot3datahandle
cd ..

cd q3dot4
gcc -o ssbq3dot4datahandle ssbq3dot4datahandle.cpp
./ssbq3dot4datahandle
cd ..

cd q4dot1
gcc -o ssbq4dot1datahandle ssbq4dot1datahandle.cpp
./ssbq4dot1datahandle
cd ..

cd q4dot2
gcc -o ssbq4dot2datahandle ssbq4dot2datahandle.cpp
./ssbq4dot2datahandle
cd ..

cd q4dot3
gcc -o ssbq4dot3datahandle ssbq4dot3datahandle.cpp
./ssbq4dot3datahandle
cd ..

cd ..
