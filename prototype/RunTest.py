import ESN as esn
from numpy import *
from matplotlib.pyplot import *
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
            display("%s: %s"%(test, json_file), Ytarget, Y)
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
            display("%s: %s"%(test, json_file), Ytarget, Y)


def displayTrajectory(windowsTitle, Ytarget, Y):
    print_Ytarget = []
    print_Y = []

    accuracy = []
    acc = 0

    for i in range(len(Y.T)):
        print_Ytarget.append(where(Ytarget[:, i]==0.7)[0][0])
        print_Y.append(where(Y[:, i]==max(Y[:, i]))[0][0])

        if print_Y[i] == print_Ytarget[i]:
            acc+=1
        accuracy.append(float(acc)/(i+1))

    print "Accuracy:", accuracy[-1]

    fig = figure()
    fig.clear()
    fig.canvas.set_window_title(windowsTitle)
    subplot(211)
    yticks(range(3), ["Class %s"%x for x in range(3)])
    plot(print_Ytarget, 'wo')
    plot(print_Y, 'b+')
    subplot(212)
    plot(accuracy, 'r')
    legend(['Accuracy'])
    yinf, ysup = fig.get_axes()[0].get_ylim()
    fig.get_axes()[0].set_ylim(yinf-0.5, ysup+0.5)

def displayMackeyGlass(windowsTitle, Ytarget, Y):
    err = []
    rmse = []
    #avg = []
    #var = []
    for i in range(len(Y.T)):
        err.append(Ytarget.T[i] - Y.T[i])
        rmse.append( np.sqrt(sum(np.power(err,2))/len(err)) )
        #avg.append(sum(np.absolute(err))/len(err))
        #var.append(sum(np.power(np.absolute(err) - avg[i], 2))/len(err))
        
    print "RMSE:", rmse[-1]

    fig = figure()
    fig.clear()
    fig.canvas.set_window_title(windowsTitle)
    subplot(211)
    plot(Ytarget.T, 'k')
    plot(Y.T, 'b')
    legend(['Targeted', 'Generated'])
    subplot(212)
    plot(err, 'k')
    plot(rmse, 'r')
    legend(['Error', 'RMSE'])

def displaySequence(windowsTitle, Ytarget, Y):
    print_Ytarget = []
    print_Y = []

    # bonneAttribution = {}
    # attribution = {}
    # appartenant = {}
    # precision = []
    # rappel = []
    # fmesure = []
    accuracy = []
    acc = 0

    for i in range(len(Y.T)):
        print_Ytarget.append(where(Ytarget[:, i]==1)[0][0])
        print_Y.append(where(Y[:, i]==max(Y[:, i]))[0][0])

        # if not attribution.has_key(print_Y[i]):
        #     attribution[print_Y[i]] = 0
        # if not bonneAttribution.has_key(print_Ytarget[i]):
        #     bonneAttribution[print_Ytarget[i]] = 0
        # if not appartenant.has_key(print_Ytarget[i]):
        #     appartenant[print_Ytarget[i]] = 0

        # appartenant[print_Ytarget[i]] +=1
        # attribution[print_Y[i]] += 1

        if print_Y[i] == print_Ytarget[i]:
            # bonneAttribution[print_Y[i]]+=1
            acc+=1
        accuracy.append(float(acc)/(i+1))

    #     tmpP = 0.0
    #     tmpR = 0.0
                
    #     for j in range(len(Y)):
    #         if bonneAttribution.has_key(j) and attribution.has_key(j) :
    #             tmpP += float(bonneAttribution[j]) / ( len(Y) * attribution[j] )
    #         if appartenant.has_key(j) and attribution.has_key(j):
    #             tmpR += float(bonneAttribution[j]) / ( len(Y) * appartenant[j] )
                
    #     precision.append(tmpP)
    #     rappel.append(tmpR)
    #     if tmpP == 0 and tmpR == 0 :
    #         fmesure.append(0)
    #     else :
    #         fmesure.append( 2*tmpP*tmpR / (tmpP + tmpR) )

    # for i in range(len(Y)):
    #     tmpP = float(bonneAttribution[i]) / attribution[i]
    #     tmpR = float(bonneAttribution[i]) / appartenant[i]
    #     print '-', chr(65 + i)
    #     print '\tPrecision:', tmpP
    #     print '\tRecall:', tmpR
    #     if tmpP == 0 and tmpR == 0 :
    #         print '\tF-Measure:', 0
    #     else :
    #         print '\tF-Measure:', 2 * tmpP * tmpR / (tmpP + tmpR)
    # print '=> Global precision:', precision[-1]
    # print '=> Global recall:', rappel[-1]
    # print '=> Global F-Measure:', fmesure[-1]
    print 'Accuracy:', accuracy[-1]

    fig = figure()
    fig.clear()
    fig.canvas.set_window_title(windowsTitle)
    subplot(211)
    yticks(range(26), [chr(65 + x) for x in range(26)])
    plot(print_Ytarget, 'wo')
    plot(print_Y, 'b+')
    legend(['Target', 'Prediction'])
    subplot(212)
    plot(accuracy, 'r')
    # plot(precision, 'g--')
    # plot(rappel, 'b--')
    # plot(fmesure, 'r--')
    legend(['Accuracy', 'Precision', 'Recall', 'F-Measure'])
    yinf, ysup = fig.get_axes()[0].get_ylim()
    fig.get_axes()[0].set_ylim(yinf-0.5, ysup+0.5)


def usage():
    print 'usage: python RunTest.py [--help] TEST_FILE... )'

if __name__ == "__main__":
    main()