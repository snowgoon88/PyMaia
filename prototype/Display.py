from numpy import where
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D


def displayTraj2D(windowsTitle, Ytarget, Y, X):
    print_Ytarget = []
    print_Y = []
    print_traj = {
            'z':[],
            'color':[],
            'x1':[],
            'x2':[]
    }
    colors = ('r', 'g', 'b')
    labels = ('2-petals', '3-petals', '4-petals')

    tmpx1 = []
    tmpx2 = []
    z = 0
    c = ''
    for i in xrange(len(Y.T)):
        print_Ytarget.append(where(Ytarget[:, i]==max(Ytarget[:, i]))[0][0])
        print_Y.append(where(Y[:, i]==max(Y[:, i]))[0][0])

        tmpx1.append(X[0, i])
        tmpx2.append(X[1, i])
        newcol = colors[print_Y[-1]]
        if c == '':
            c = newcol
        if not c == newcol:
            print_traj['x1'].append(tmpx1)
            print_traj['x2'].append(tmpx2)
            print_traj['z'].append(z)
            print_traj['color'].append(c)
            tmpx1 = tmpx1[-1:]
            tmpx2 = tmpx2[-1:]
            c = newcol
        if i % 30 == 29:
            print_traj['x1'].append(tmpx1);
            print_traj['x2'].append(tmpx2);
            print_traj['z'].append(z);
            print_traj['color'].append(c)
            tmpx1 = tmpx1[-1:]
            tmpx2 = tmpx2[-1:]
            z+=1

    fig = figure()
    fig.clear()
    fig.canvas.set_window_title(windowsTitle)
    [plot(None,None,ls='-',c=c,label=l) for c,l in zip(colors,labels)]
    legend(labels)
    for i in xrange(len(print_traj['z'])):
        plot([x + 2*print_traj['z'][i] for x in print_traj['x1'][i]], print_traj['x2'][i], c=print_traj['color'][i])


def displayTraj3D(windowsTitle, Ytarget, Y, X):
    print_Ytarget = []
    print_Y = []
    print_traj = {
            'z':[],
            'color':[],
            'x1':[],
            'x2':[]
    }
    colors = ('r', 'g', 'b')
    labels = ('2-petals', '3-petals', '4-petals')

    tmpx1 = []
    tmpx2 = []
    z = 0
    c = ''
    for i in xrange(len(Y.T)):
        print_Ytarget.append(where(Ytarget[:, i]==max(Ytarget[:, i]))[0][0])
        print_Y.append(where(Y[:, i]==max(Y[:, i]))[0][0])

        tmpx1.append(X[0, i])
        tmpx2.append(X[1, i])
        newcol = colors[print_Y[-1]]
        if c == '':
            c = newcol
        if not c == newcol:
            print_traj['x1'].append(tmpx1)
            print_traj['x2'].append(tmpx2)
            print_traj['z'].append(z)
            print_traj['color'].append(c)
            tmpx1 = tmpx1[-1:]
            tmpx2 = tmpx2[-1:]
            c = newcol
        if i % 30 == 29:
            print_traj['x1'].append(tmpx1);
            print_traj['x2'].append(tmpx2);
            print_traj['z'].append(z);
            print_traj['color'].append(c)
            tmpx1 = tmpx1[-1:]
            tmpx2 = tmpx2[-1:]
            z+=1

    fig = figure()
    fig.clear()
    fig.canvas.set_window_title(windowsTitle)
    ax = fig.gca(projection='3d')
    [ax.plot([],[],ls='-',c=c,label=l) for c,l in zip(colors,labels)]
    ax.legend(labels)
    for i in xrange(len(print_traj['z'])):
        ax.plot(print_traj['x1'][i], print_traj['x2'][i], print_traj['z'][i], zdir='z', c=print_traj['color'][i])


def displayRMSE(windowsTitle, Ytarget, Y):
    err = []
    rmse = []
    for i in xrange(len(Y.T)):
        err.append(Ytarget.T[i] - Y.T[i])
        rmse.append( np.sqrt(sum(np.power(err,2))/len(err)) )
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


def displayAcc(windowsTitle, Ytarget, Y, ticks):
    print_Ytarget = []
    print_Y = []
    accuracy = []
    acc = 0

    for i in xrange(len(Y.T)):
        print_Ytarget.append(where(Ytarget[:, i]==max(Ytarget[:, i]))[0][0])
        print_Y.append(where(Y[:, i]==max(Y[:, i]))[0][0])
        if print_Y[i] == print_Ytarget[i]:
            acc+=1
        accuracy.append(float(acc)/(i+1))
    print 'Accuracy:', accuracy[-1]

    fig = figure()
    fig.clear()
    fig.canvas.set_window_title(windowsTitle)
    subplot(211)
    yticks(range(26), ticks)
    plot(print_Ytarget, 'wo')
    plot(print_Y, 'bx')
    legend(['Target', 'Prediction'])
    subplot(212)
    plot(accuracy, 'r')
    legend(['Accuracy'])
    yinf, ysup = fig.get_axes()[0].get_ylim()
    fig.get_axes()[0].set_ylim(yinf-0.5, ysup+0.5)


def displayF(windowsTitle, Ytarget, Y, ticks):
    print_Ytarget = []
    print_Y = []

    bonneAttribution = {}
    attribution = {}
    appartenant = {}

    precision = []
    rappel = []
    fmesure = []

    for i in xrange(len(Y)):
        attribution[i]=0;
        bonneAttribution[i]=0;
        appartenant[i]=0;

    for i in xrange(len(Y.T)):
        print_Ytarget.append(where(Ytarget[:, i]==max(Ytarget[:, i]))[0][0])
        print_Y.append(where(Y[:, i]==max(Y[:, i]))[0][0])

        # if not attribution.has_key(print_Y[i]):
        #     attribution[print_Y[i]] = 0
        # if not bonneAttribution.has_key(print_Ytarget[i]):
        #     bonneAttribution[print_Ytarget[i]] = 0
        # if not appartenant.has_key(print_Ytarget[i]):
        #     appartenant[print_Ytarget[i]] = 0

        appartenant[print_Ytarget[i]] +=1
        attribution[print_Y[i]] += 1

        if print_Y[i] == print_Ytarget[i]:
            bonneAttribution[print_Y[i]]+=1

        tmpP = 0.0
        tmpR = 0.0
                
        for j in range(len(Y)):
            if attribution[j] != 0 :
                tmpP += float(bonneAttribution[j]) / ( len(Y) * attribution[j] )
            if appartenant[j] != 0 :
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
        print '- class', ticks[i]
        print '\tPrecision:', tmpP
        print '\tRecall:', tmpR
        if tmpP == 0 and tmpR == 0 :
            print '\tF-Measure:', 0
        else :
            print '\tF-Measure:', 2 * tmpP * tmpR / (tmpP + tmpR)

    print 'Precision:', precision[-1]
    print 'Recall:', rappel[-1]
    print 'F-Measure:', fmesure[-1]

    fig = figure()
    fig.clear()
    fig.canvas.set_window_title(windowsTitle)
    subplot(211)
    yticks(range(26), ticks)
    plot(print_Ytarget, 'wo')
    plot(print_Y, 'bx')
    legend(['Target', 'Prediction'])
    subplot(212)
    plot(precision, 'g--')
    plot(rappel, 'b--')
    plot(fmesure, 'r')
    legend(['Precision', 'Recall', 'F-Measure'])
    yinf, ysup = fig.get_axes()[0].get_ylim()
    fig.get_axes()[0].set_ylim(yinf-0.5, ysup+0.5)