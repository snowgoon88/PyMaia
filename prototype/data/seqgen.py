from numpy import random
import sys, getopt, json

def main():
    try:
        opts, files = getopt.getopt(sys.argv[1:], "l:o:", ["help"])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(1)

    length = 10000
    suffix = "_data"
    for opt, arg in opts:
        if opt == '--help':
            usage()
            sys.exit(0)
        elif opt == '-l':
            length = int(arg)
        elif opt == '-o':
            suffix = arg
    
    for json_file in files:
        fd = open(json_file, "r")
        param = json.load(fd)
        fd.close()

        seq = generate(length, param)
        fd = open("%s%s"%(json_file, suffix), "w")
        for i in range(len(seq)):
            fd.write("%s\n" % seq[i])
        fd.close()

def generate(length, param):
    seq = []
    for i in range(max(param["delay"])):
        idx = int(random.random_sample()*len(param["alphabet"]))
        seq.append(param["alphabet"][idx])
    for i in range(max(param["delay"]), length):
        tmp = []
        for j in range(len(param["delay"])):
            tmp.append(seq[i-param["delay"][j]])
        seq.append(param["rules"][''.join(tmp)])
    return seq

def usage():
    print 'usage: python seqgen.py [--help] [-l SEQUENCE_LENGTH] [-o OUTPUT_SUFFIX] SEQUENCE_FILE*'

if __name__ == "__main__":
    main()
