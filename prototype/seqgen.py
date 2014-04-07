from numpy import random
import sys, getopt, json

def main():
    try:
        opts, files = getopt.getopt(sys.argv[1:], "l:", ["help"])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(1)

    length = 0
    for opt, arg in opts:
        if opt == '--help':
            usage()
            sys.exit(0)
        elif opt == '-l':
            length = int(arg)
    
    for json_file in files:
        fd = open(json_file, "r")
        param = json.load(fd)
        fd.close()

        data = generate(length, param)
        fd = open("%s_data"%json_file, "w")
        for i in range(len(data)):
            fd.write(data[i])
        fd.close()

def generate(length, param):
    data = []
    for i in range(max(param["delay"])):
        idx = int(random.random_sample()*len(param["alphabet"]))
        data.append(param["alphabet"][idx])
    for i in range(max(param["delay"]), length):
        tmp = param["rules"]
        for j in range(len(param["delay"])):
            tmp = tmp[data[i-param["delay"][j]]]
        data.append(tmp)
    return data

def usage():
    print 'python seqgen.py [--help] -l <sequence_length> <sequence_file>*'

if __name__ == "__main__":
    main()
