import sys
from nltk.corpus import brown

num = int(sys.argv[1])

f = open("tagged_corpus.txt", "w")
for i in range(num):
     f.write(brown.tagged_words()[i][0] + "/" + brown.tagged_words()[i][1]+ " ")
