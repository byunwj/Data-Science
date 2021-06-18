import sys

"second change"

if len(sys.argv) != 3:
    print("Need 3 arguments: python file, infilepath, outfilepath") 
    sys.exit()
else:
    infilepath = sys.argv[1]
    outfilepath = sys.argv[2]


