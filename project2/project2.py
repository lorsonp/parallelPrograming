import os
for t in [ 1, 2, 3, 4, 5, 6, 7, 8 ]:
    # print ("NUMT = %d" % t)
    for s in [ 100, 500, 1000, 2000, 3000, 7000 ]:
        # print ("NUMS = %d" % s)
        cmd = "g++ -DNUMT=$i -DNUMNODES=$k -o proj2 project2.cpp -lm -fopenmp"
        os.system( cmd )
        cmd = "./prog2"
        os.system( cmd )
