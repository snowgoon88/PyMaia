import ESN as esn
from numpy import *
from matplotlib.pyplot import *
import json, getopt, sys

def main():
    try:
        opts, files = getopt.getopt(sys.argv[1:], "gp", ["help", "rappel"])
    except getopt.GetoptError as err:
        print str(err)
        usage()
        sys.exit(1)

    generationFlag = False
    predictionFlag = False
    rappelFlag = False
    for opt, arg in opts:
        if opt == '--help':
            usage()
            sys.exit(0)
        elif opt == '--rappel':
            rappelFlag = True
        elif opt == '-g':
            generationFlag = True
        elif opt == '-p':
            predictionFlag = True

    numFig = 0 ;
    for k in range(len(files)) :
        print "=====", files[k], "====="

        fd=open(files[k], 'r')
        json_data = json.load(fd)
        fd.close()

        if json_data['data']['type'] == 'MackeyGlass' :
            tmp = loadtxt(json_data['data']['path'])
            data = zeros((1, len(tmp)))
            for i in range(len(tmp)):
                data[:, i] = tmp[i]

            if not rappelFlag:
                if predictionFlag:
                    print "* Prediction"
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
                    displayMackeyGlass(numFig, "Prediction avec %s"%files[k], Ytarget, Y)
                    numFig += 1

                if generationFlag:
                    print "* Generation"
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
                    displayMackeyGlass(numFig, "Generation avec %s"%files[k], Ytarget, Y)     
                    numFig += 1           
            else:
                if predictionFlag:
                    print "* Rappel sur prediction"
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
                    displayMackeyGlass(numFig, "Rappel sur prediction avec  %s"%files[k], Ytarget, Y)
                    numFig += 1

                if generationFlag:
                    print "* Rappel sur generation"
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
                    displayMackeyGlass(numFig, "Rappel sur generation avec %s"%files[k], Ytarget, Y)     
                    numFig += 1  

        elif json_data['data']['type'] == 'Sequence' :
            tmp = loadtxt(json_data['data']['path'], dtype=string0)
            data = zeros((json_data['esn']['K'], len(tmp)))
            for i in range(len(tmp)):
                data[:, i] = array(json_data['data']['encode'][tmp[i]])

            if not rappelFlag:
                if predictionFlag:
                    print "* Prediction"
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
                    displaySequence(numFig, "Prediction avec %s"%files[k], Ytarget, Y)
                    numFig += 1

                if generationFlag:
                    print "* Generation"
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
                    displaySequence(numFig, "Generation avec %s"%files[k], Ytarget, Y)     
                    numFig += 1           
            else:
                if predictionFlag:
                    print "* Rappel sur prediction"
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
                    displaySequence(numFig, "Rappel sur prediction avec  %s"%files[k], Ytarget, Y)
                    numFig += 1

                if generationFlag:
                    print "* Rappel sur generation"
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
                    displaySequence(numFig, "Rappel sur generation avec %s"%files[k], Ytarget, Y)     
                    numFig += 1  

        else :
            print 'Unsupported data type: ',json_data['data']['type']
            sys.exit(2)

    show()

def displayMackeyGlass(k, windowsTitle, Ytarget, Y):
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

    fig = figure(k)
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

def displaySequence(k, windowsTitle, Ytarget, Y):
    print_Ytarget = []
    print_Y = []

    bonneAttribution = {}
    attribution = {}
    appartenant = {}
    precision = []
    rappel = []
    fmesure = []
    #accuracy = []
    #acc = 0

    for i in range(len(Y.T)):
        print_Ytarget.append(where(Ytarget[:, i]==1)[0][0])
        print_Y.append(where(Y[:, i]==max(Y[:, i]))[0][0])

        if not attribution.has_key(print_Y[i]):
            attribution[print_Y[i]] = 0
        if not bonneAttribution.has_key(print_Ytarget[i]):
            bonneAttribution[print_Ytarget[i]] = 0
        if not appartenant.has_key(print_Ytarget[i]):
            appartenant[print_Ytarget[i]] = 0

        appartenant[print_Ytarget[i]] +=1
        attribution[print_Y[i]] += 1

        if print_Y[i] == print_Ytarget[i]:
            bonneAttribution[print_Y[i]]+=1
            #acc+=1
        #accuracy.append(float(acc)/(i+1))

        tmpP = 0.0
        tmpR = 0.0
                
        for j in range(len(Y)):
            if bonneAttribution.has_key(j) and attribution.has_key(j) :
                tmpP += float(bonneAttribution[j]) / ( len(Y) * attribution[j] )
            if appartenant.has_key(j) and attribution.has_key(j):
                tmpR += float(bonneAttribution[j]) / ( len(Y) * appartenant[j] )
                
        precision.append(tmpP)
        rappel.append(tmpR)
        if tmpP == 0 and tmpR == 0 :
            fmesure.append(0)
        else :
            fmesure.append( 2*tmpP*tmpR / (tmpP + tmpR) )

    for i in range(len(Y)):
        tmpP = float(bonneAttribution[i]) / attribution[i]
        tmpR = float(bonneAttribution[i]) / appartenant[i]
        print '-', chr(65 + i)
        print '\tPrecision:', tmpP
        print '\tRecall:', tmpR
        if tmpP == 0 and tmpR == 0 :
            print '\tF-Measure:', 0
        else :
            print '\tF-Measure:', 2 * tmpP * tmpR / (tmpP + tmpR)
    print '=> Global precision:', precision[-1]
    print '=> Global recall:', rappel[-1]
    print '=> Global F-Measure:', fmesure[-1]

    fig = figure(k)
    fig.clear()
    fig.canvas.set_window_title(windowsTitle)
    subplot(211)
    yticks(range(26), [chr(65 + x) for x in range(26)])
    plot(print_Ytarget, 'wo')
    plot(print_Y, 'b+')
    legend(['Target', 'Prediction'])
    subplot(212)
    plot(precision, 'g--')
    plot(rappel, 'b--')
    plot(fmesure, 'r')
    legend(['Precision', 'Recall', 'F-Measure'])
    yinf, ysup = fig.get_axes()[0].get_ylim()
    fig.get_axes()[0].set_ylim(yinf-0.5, ysup+0.5)


def usage():
    print 'usage: python QuickTest.py [--help] TEST_FILE... )'

if __name__ == "__main__":
    main()