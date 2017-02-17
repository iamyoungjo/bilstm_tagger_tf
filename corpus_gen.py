import sys
import numpy as np
import random

V = 50000
T = 10 
C = 100000

file = open("corpus.txt", 'w')
C = int(sys.argv[1])
	
print("generating tagged corpus w/ size = " , C, " ...")

for i in range(C):
	file.write( "word" + str(random.randint(1, V)) + "/" + "tag" + str(random.randint(1, T)) + " " )
	if(i%10 == 0):
                file.write("\n")

print("done.")	
