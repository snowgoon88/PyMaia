import ESN as esn
from Display import *
from numpy import *
from matplotlib.pyplot import show

import json, getopt, sys

def main():
    try:
        opts, files = getopt.getopt(sys.argv[1:], "", ["help"])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(1)

    for opt, arg in opts:
        if opt == '--help':
            usage()
            sys.exit(0)

    for json_file in files :
        print "=====", json_file, "====="

        fd=open(json_file, 'r')
        json_data = json.load(fd)
        fd.close()

        if json_data['data']['type'] == 'MackeyGlass' :
            tmp = loadtxt(json_data['data']['path'])
            data = zeros((1, len(tmp)))
            for i in range(len(tmp)):
                data[:, i] = tmp[i]
            process(json_file, json_data, data, displayMackeyGlass)
            
        elif json_data['data']['type'] == 'Sequence' :
            tmp = loadtxt(json_data['data']['path'], dtype=string0)
            data = zeros((json_data['esn']['K'], len(tmp)))
            for i in range(len(tmp)):
                data[:, i] = array(json_data['data']['encode'][tmp[i]])
            process(json_file, json_data, data, displaySequence)

        elif json_data['data']['type'] == 'Trajectory':
            tmpData = loadtxt(json_data['data']['path'])
            tmpTarget = loadtxt(json_data['data']['target'])
            process(json_file, json_data, (tmpData.T, tmpTarget.T), displayTrajectory)

        else :
            print 'Unsupported data type: ',json_data['data']['type']
            sys.exit(2)

    show()


def process(json_file, json_data, data, display):
    for test in json_data["test"]:
        print "*", test
        if test == 'generation':
            Ytarget, Y = esn.generation(json_data['esn']['K'], 
                                        json_data['esn']['N'], 
                                        json_data['esn']['L'], 
                                        json_data['esn']['seed'], 
                                        json_data['esn']['leaking_rate'], 
                                        json_data['esn']['rho_factor'], 
                                        json_data['esn']['regul_coef'],
                                        data, 
                                        json_data['data']['init_len'], 
                                        json_data['data']['train_len'], 
                                        json_data['data']['test_len'])
            display("%s: %s"%(test, json_file), Ytarget, Y)
        elif test == 'prediction':
            Ytarget, Y = esn.prediction(json_data['esn']['K'], 
                                        json_data['esn']['N'], 
                                        json_data['esn']['L'], 
                                        json_data['esn']['seed'], 
                                        json_data['esn']['leaking_rate'], 
                                        json_data['esn']['rho_factor'], 
                                        json_data['esn']['regul_coef'],
                                        data, 
                                        json_data['data']['init_len'], 
                                        json_data['data']['train_len'], 
                                        json_data['data']['test_len'])
            display("%s: %s"%(test, json_file), Ytarget, Y)
        elif test == 'rappelGeneration':
            Ytarget, Y = esn.rappelGeneration(json_data['esn']['K'], 
                                              json_data['esn']['N'], 
                                              json_data['esn']['L'],
                                              json_data['esn']['seed'], 
                                              json_data['esn']['leaking_rate'], 
                                              json_data['esn']['rho_factor'], 
                                              json_data['esn']['regul_coef'],
                                              data, 
                                              json_data['data']['init_len'], 
                                              json_data['data']['train_len'])
            display("%s: %s"%(test, json_file), Ytarget, Y)
        elif test == 'rappelPrediction':
            Ytarget, Y = esn.rappelPrediction(json_data['esn']['K'], 
                                              json_data['esn']['N'], 
                                              json_data['esn']['L'], 
                                              json_data['esn']['seed'], 
                                              json_data['esn']['leaking_rate'], 
                                              json_data['esn']['rho_factor'], 
                                              json_data['esn']['regul_coef'],
                                              data, 
                                              json_data['data']['init_len'], 
                                              json_data['data']['train_len'])
            display("%s: %s"%(test, json_file), Ytarget, Y)
        elif test == 'classification':
            Ytarget, Y = esn.classification(json_data['esn']['K'], 
                                            json_data['esn']['N'], 
                                            json_data['esn']['L'], 
                                            json_data['esn']['seed'], 
                                            json_data['esn']['leaking_rate'], 
                                            json_data['esn']['rho_factor'], 
                                            json_data['esn']['regul_coef'],
                                            data[0],
                                            data[1], 
                                            json_data['data']['init_len'], 
                                            json_data['data']['train_len'], 
                                            json_data['data']['test_len'])
            display("%s: %s"%(test, json_file), Ytarget, Y, X=data[0][:, json_data['data']['init_len']+json_data['data']['train_len']:json_data['data']['init_len']+json_data['data']['train_len']+json_data['data']['test_len']])

        elif test == 'classificationPrediction':
            Ytarget, Y = esn.classificationPrediction(json_data['esn']['K'], 
                                                      json_data['esn']['N'], 
                                                      json_data['esn']['L'], 
                                                      json_data['esn']['seed'], 
                                                      json_data['esn']['leaking_rate'], 
                                                      json_data['esn']['rho_factor'], 
                                                      json_data['esn']['regul_coef'],
                                                      data[0],
                                                      data[1], 
                                                      json_data['data']['init_len'], 
                                                      json_data['data']['train_len'], 
                                                      json_data['data']['test_len'])
            display("%s: %s"%(test, json_file), Ytarget, Y, X=data[0][:, json_data['data']['init_len']+json_data['data']['train_len']:json_data['data']['init_len']+json_data['data']['train_len']+json_data['data']['test_len']])

        elif test == 'rappelClassification':
            Ytarget, Y = esn.rappelClassification(json_data['esn']['K'], 
                                                  json_data['esn']['N'], 
                                                  json_data['esn']['L'], 
                                                  json_data['esn']['seed'], 
                                                  json_data['esn']['leaking_rate'], 
                                                  json_data['esn']['rho_factor'], 
                                                  json_data['esn']['regul_coef'],
                                                  data[0],
                                                  data[1], 
                                                  json_data['data']['init_len'], 
                                                  json_data['data']['train_len'])
            display("%s: %s"%(test, json_file), Ytarget, Y, X=data[0][:, json_data['data']['init_len']+json_data['data']['train_len']:json_data['data']['init_len']+json_data['data']['train_len']+json_data['data']['test_len']])


def usage():
    print 'usage: python RunTest.py [--help] TEST_FILE... )'

if __name__ == "__main__":
    main()