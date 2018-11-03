
import interface.model_interface as mi
import numpy as np

class EnsembleModel:

    def __init__(self):

        self.model_inception = mi.ModelInterface('Inception')
        self.model_mobilenet = mi.ModelInterface('Mobilenet')

    def predict(self, filename, sess1, sess2,threshold):

        label1, prob1 = self.model_inception.predict(filename, sess1)
        label2, prob2 = self.model_mobilenet.predict(filename, sess2)

#        print('label1,prob1:\n',label1)
#        print(prob1)
#        print('label2,prob2:\n',label2)
#        print(prob2)
        return self.ensemble_predict([label1, prob1], [label2, prob2],threshold)


    def ensemble_predict(self, pre1, pre2, threshold):
        
        A=0.65
        B=0.35
        prob_normal=(A*pre1[1][pre1[0].index('normal')]+B*pre2[1][pre2[0].index('normal')])
        prob_porosity=(A*pre1[1][pre1[0].index('porosity')]+B*pre2[1][pre2[0].index('porosity')])
        prob_crack=(A*pre1[1][pre1[0].index('crack')]+B*pre2[1][pre2[0].index('crack')])
        prob_burnthrough=(A*pre1[1][pre1[0].index('burnthrough')]+B*pre2[1][pre2[0].index('burnthrough')])
        
        integrate_arr=np.array([prob_normal,prob_porosity,prob_crack,prob_burnthrough])
        
        max_index=np.argmax(integrate_arr).astype(np.int32)
#        print('Max index:',max_index,',Prob:',integrate_arr[max_index],'threshold:',threshold)
        if max_index>0 and integrate_arr[max_index]>threshold:
#            print('Final defect type:',max_index,'\n')
            return (max_index).astype(np.int32)
        else:
#            print('Final defect type:','No defect','\n')
            return 0