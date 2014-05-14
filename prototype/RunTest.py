from Task import *
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
            process(json_file, json_data, data)
            
        elif json_data['data']['type'] == 'Sequence' :
            tmp = loadtxt(json_data['data']['path'], dtype=string0)
            data = zeros((json_data['esn']['K'], len(tmp)))
            for i in range(len(tmp)):
                data[:, i] = array(json_data['data']['encode'][tmp[i]])
            process(json_file, json_data, data)

        elif json_data['data']['type'] == 'Trajectory':
            tmpData = loadtxt(json_data['data']['path'])
            tmpTarget = loadtxt(json_data['data']['target'])
            process(json_file, json_data, (tmpData.T, tmpTarget.T))

        else :
            print 'Unsupported data type: ',json_data['data']['type']
            sys.exit(2)

    show()


def process(json_file, json_data, data):
    ticks = []
    if json_data['data']['type'] == 'Sequence':
      ticks = [chr(x+65) for x in xrange(26)]
    elif json_data['data']['type'] == 'Trajectory':
      ticks = ['2-petals', '3-petals', '4-petals']

    for task in json_data["task"]:
        print "-----", task, "-----"

        if task == 'generation':
            Ytarget, Y = generation(json_data['esn']['K'], 
                                    json_data['esn']['N'], 
                                    json_data['esn']['L'], 
                                    json_data['esn']['Win'], 
                                    json_data['esn']['W'], 
                                    json_data['esn']['leaking_rate'], 
                                    json_data['esn']['rho_factor'], 
                                    json_data['esn']['regul_coef'],
                                    data, 
                                    json_data['data']['init_len'], 
                                    json_data['data']['train_len'], 
                                    json_data['data']['test_len'])
            for display in json_data["task"][task]:
              if display == "displayAcc":
                displayAcc("%s: %s"%(task, json_file), Ytarget, Y, ticks)
              if display == "displayRMSE":
                displayRMSE("%s: %s"%(task, json_file), Ytarget, Y)
              if display == "displayF":
                displayF("%s: %s"%(task, json_file), Ytarget, Y, ticks)
        
        elif task == 'prediction':
            Ytarget, Y = prediction(json_data['esn']['K'], 
                                        json_data['esn']['N'], 
                                        json_data['esn']['L'], 
                                        json_data['esn']['Win'], 
                                        json_data['esn']['W'], 
                                        json_data['esn']['leaking_rate'], 
                                        json_data['esn']['rho_factor'], 
                                        json_data['esn']['regul_coef'],
                                        data, 
                                        json_data['data']['init_len'], 
                                        json_data['data']['train_len'], 
                                        json_data['data']['test_len'])
            for display in json_data["task"][task]:
              if display == "displayAcc":
                displayAcc("%s: %s"%(task, json_file), Ytarget, Y, ticks)
              if display == "displayRMSE":
                displayRMSE("%s: %s"%(task, json_file), Ytarget, Y)
              if display == "displayF":
                displayF("%s: %s"%(task, json_file), Ytarget, Y, ticks)
        
        elif task == 'rappelGeneration':
            Ytarget, Y = rappelGeneration(json_data['esn']['K'], 
                                              json_data['esn']['N'], 
                                              json_data['esn']['L'],
                                              json_data['esn']['Win'], 
                                              json_data['esn']['W'], 
                                              json_data['esn']['leaking_rate'], 
                                              json_data['esn']['rho_factor'], 
                                              json_data['esn']['regul_coef'],
                                              data, 
                                              json_data['data']['init_len'], 
                                              json_data['data']['train_len'])
            for display in json_data["task"][task]:
              if display == "displayAcc":
                displayAcc("%s: %s"%(task, json_file), Ytarget, Y, ticks)
              if display == "displayRMSE":
                displayRMSE("%s: %s"%(task, json_file), Ytarget, Y)
              if display == "displayF":
                displayF("%s: %s"%(task, json_file), Ytarget, Y, ticks)
        
        elif task == 'rappelPrediction':
            Ytarget, Y = rappelPrediction(json_data['esn']['K'], 
                                              json_data['esn']['N'], 
                                              json_data['esn']['L'], 
                                              json_data['esn']['Win'], 
                                              json_data['esn']['W'], 
                                              json_data['esn']['leaking_rate'], 
                                              json_data['esn']['rho_factor'], 
                                              json_data['esn']['regul_coef'],
                                              data, 
                                              json_data['data']['init_len'], 
                                              json_data['data']['train_len'])
            for display in json_data["task"][task]:
              if display == "displayAcc":
                displayAcc("%s: %s"%(task, json_file), Ytarget, Y, ticks)
              if display == "displayRMSE":
                displayRMSE("%s: %s"%(task, json_file), Ytarget, Y)
              if display == "displayF":
                displayF("%s: %s"%(task, json_file), Ytarget, Y, ticks)
  
        elif task == 'classification':
            Ytarget, Y = classification(json_data['esn']['K'], 
                                            json_data['esn']['N'], 
                                            json_data['esn']['L'], 
                                            json_data['esn']['Win'], 
                                            json_data['esn']['W'], 
                                            json_data['esn']['leaking_rate'], 
                                            json_data['esn']['rho_factor'], 
                                            json_data['esn']['regul_coef'],
                                            data[0],
                                            data[1], 
                                            json_data['data']['init_len'], 
                                            json_data['data']['train_len'], 
                                            json_data['data']['test_len'])
            for display in json_data["task"][task]:
              if display == "displayAcc":
                displayAcc("%s: %s"%(task, json_file), Ytarget, Y, ticks)
              if display == "displayRMSE":
                displayRMSE("%s: %s"%(task, json_file), Ytarget, Y)
              if display == "displayTraj2D":
                displayTraj2D("%s: %s"%(task, json_file), Ytarget, Y, data[0][:, json_data['data']['init_len']+json_data['data']['train_len']:json_data['data']['init_len']+json_data['data']['train_len']+json_data['data']['test_len']])
              if display == "displayTraj3D":
                displayTraj3D("%s: %s"%(task, json_file), Ytarget, Y, data[0][:, json_data['data']['init_len']+json_data['data']['train_len']:json_data['data']['init_len']+json_data['data']['train_len']+json_data['data']['test_len']])
              if display == "displayF":
                displayF("%s: %s"%(task, json_file), Ytarget, Y, ticks)
 
        elif task == 'classificationPrediction':
            Ytarget, Y = classificationPrediction(json_data['esn']['K'], 
                                                      json_data['esn']['N'], 
                                                      json_data['esn']['L'], 
                                                      json_data['esn']['Win'], 
                                                      json_data['esn']['W'], 
                                                      json_data['esn']['leaking_rate'], 
                                                      json_data['esn']['rho_factor'], 
                                                      json_data['esn']['regul_coef'],
                                                      data[0],
                                                      data[1], 
                                                      json_data['data']['init_len'], 
                                                      json_data['data']['train_len'], 
                                                      json_data['data']['test_len'])
            for display in json_data["task"][task]:
              if display == "displayAcc":
                displayAcc("%s: %s"%(task, json_file), Ytarget, Y, ticks)
              if display == "displayRMSE":
                displayRMSE("%s: %s"%(task, json_file), Ytarget, Y)
              if display == "displayTraj2D":
                displayTraj2D("%s: %s"%(task, json_file), Ytarget, Y, data[0][:, json_data['data']['init_len']+json_data['data']['train_len']:json_data['data']['init_len']+json_data['data']['train_len']+json_data['data']['test_len']])
              if display == "displayTraj3D":
                displayTraj3D("%s: %s"%(task, json_file), Ytarget, Y, data[0][:, json_data['data']['init_len']+json_data['data']['train_len']:json_data['data']['init_len']+json_data['data']['train_len']+json_data['data']['test_len']])
              if display == "displayF":
                displayF("%s: %s"%(task, json_file), Ytarget, Y, ticks)

        elif task == 'rappelClassification':
            Ytarget, Y = rappelClassification(json_data['esn']['K'], 
                                                  json_data['esn']['N'], 
                                                  json_data['esn']['L'], 
                                                  json_data['esn']['Win'], 
                                                  json_data['esn']['W'], 
                                                  json_data['esn']['leaking_rate'], 
                                                  json_data['esn']['rho_factor'], 
                                                  json_data['esn']['regul_coef'],
                                                  data[0],
                                                  data[1], 
                                                  json_data['data']['init_len'], 
                                                  json_data['data']['train_len'])
            for display in json_data["task"][task]:
              if display == "displayAcc":
                displayAcc("%s: %s"%(task, json_file), Ytarget, Y, ticks)
              if display == "displayRMSE":
                displayRMSE("%s: %s"%(task, json_file), Ytarget, Y)
              if display == "displayTraj2D":
                displayTraj2D("%s: %s"%(task, json_file), Ytarget, Y, data[0][:, json_data['data']['init_len']+json_data['data']['train_len']:json_data['data']['init_len']+json_data['data']['train_len']+json_data['data']['test_len']])
              if display == "displayTraj3D":
                displayTraj3D("%s: %s"%(task, json_file), Ytarget, Y, data[0][:, json_data['data']['init_len']+json_data['data']['train_len']:json_data['data']['init_len']+json_data['data']['train_len']+json_data['data']['test_len']])
              if display == "displayF":
                displayF("%s: %s"%(task, json_file), Ytarget, Y, ticks)

def usage():
    print 'usage: python RunTest.py [--help] TEST_FILE... )'

if __name__ == "__main__":
    main()