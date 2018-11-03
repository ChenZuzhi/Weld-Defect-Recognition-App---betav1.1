# -*- coding: utf-8 -*-
"""

"""

from PyQt5 import QtGui,QtCore,QtWidgets
import sys,os,shutil,cv2
import numpy as np
import interface.ensemble_interface as ei
import tensorflow as tf

#TODO: Make the executive app
#TODO: The app comments,like copyright or something
#TODO: The readme
#TODO: Spyder not restarting the kernel the app will die


class DefectClassWindow(QtWidgets.QDialog):

    def __init__(self,parent= None):
        print('Configuring the main dialog window...\n')
        super(DefectClassWindow,  self).__init__(parent)
        
        # The dialog window
        self._minHeight=700
        self._minWidth=432
        self._maxHeight=1024
        self._maxWidth=633
        self.setWindowTitle("Weld Defect Recognition App (Beta v1.1)")
        self.setMinimumHeight(self._minHeight)
        self.setMinimumWidth(self._minWidth)
        self.setMaximumHeight(self._maxHeight)
        self.setMaximumWidth(self._maxWidth)
        
        # The image window
        self.imageView = QtWidgets.QLabel("Please open a weld image.")# Creating the widget for image presenting
        self.imageView.setAlignment(QtCore.Qt.AlignCenter)# Place that widget in the middle of the dialog window
        
        # The 'Open' button
        self.open_button = QtWidgets.QPushButton("Select an image") # Creating the 'open' button widget
        self.open_button.clicked.connect(self.on_Open_clicked) # Connect this button to a function
        
        # The 'Run' button
        self.run_button = QtWidgets.QPushButton("Run") # Creating the 'Run' button widget
        self.run_button.clicked.connect(self.on_Run_clicked) # Connect this button to a function
        
        # The 'Export' button
        self.export_button = QtWidgets.QPushButton("Export") # Creating the 'Export' button widget
        self.export_button.clicked.connect(self.on_Export_clicked) # Connect this button to a function
        
        # Add the widgets to the layout
        self.vlayout = QtWidgets.QVBoxLayout()
        self.vlayout.addWidget(self.imageView)
        self.vlayout.addWidget(self.open_button)
        self.vlayout.addWidget(self.run_button)
        self.vlayout.addWidget(self.export_button)
        self.setLayout(self.vlayout)
        
        # Some directory stuff for later use
        self.main_path=os.getcwd()
        self.tmp_path=self.main_path + os.path.sep + 'tmp'
        if os.path.exists(self.tmp_path):
            shutil.rmtree(self.tmp_path)
        os.makedirs(self.tmp_path)
        
        # Load model interface
        self.model = ei.EnsembleModel()
        self.graph1 = self.model.model_inception.modelGraph()
        self.graph2 = self.model.model_mobilenet.modelGraph()
        
        # Some parametres about the recognition
        self.threshold=0.5
        self.block_size=[32,32]
        self.overlap=[int(self.block_size[0]/2),int(self.block_size[1]/2)]
        
        # Some variables that will be valued later
        self.filename=None # The path of the opened image
        self.image=None # The original image as the type of np.array
        self.image_shape=None # The shape of the original image
        self.imageClass=None # The original image as a class(for presenting by PyQt5)
        self.processed_image=None # The image after pre-processing(That is the extracted weld image)
        self.p_image_shape=None # The shape of the processed image
        self.res_image=None # The predicted processed image
        self.defect_label=None # Recording the position of the defects
        self.rolling_window=None # The image in the rolling window
        self.x_b=None # The sliding Times w.r.t width direction
        self.y_b=None # The sliding Times w.r.t height direction
        self.x_ss=None
        self.y_ss=None # For the use of recording the block that the app is working at the moment
        self.deltaxx=None
        self.deltayy=None
        self.color_set=[(0,255,255),(0,255,255),(0,255,255)] # Here, all kinds of defects is marked by yellow(which RGB is 0,255,255)lines
        self.new_block_size=np.array([self.block_size[0]-self.overlap[0],self.block_size[1]-self.overlap[1]])# Whatever this name is,you know its meaning
        self.pt=None # Recording the position of each small block
        self.rolling_size=[16,16] # Means the rolling windows is 
                            #H:rolling_size[0]*block_size[0],W:rolling_size[1]*block_size[1]
        self.determine_start=[int(self.rolling_size[0]/2),int(self.rolling_size[1]/2)]
        print('Configuration is finished.\n')


    def on_Open_clicked(self, checked):
        # Get the filename(as well as path)
        self.filename = QtWidgets.QFileDialog.getOpenFileName(self, "OpenFile", ".", 
            "Image Files(*.jpg *.jpeg *.png *.bmp)")[0]
        if len(self.filename):
            print('Loading image from:',str(self.filename))
            # Open the iamge by openCV
            self.image=np.array(cv2.imread(str(self.filename)))
            self.res_image=np.copy(self.image)
            self.image_shape=np.shape(self.image) # (height,width)
            # if the image oversizes the dialog window, we resized the image and restored it.
            tmp_max_direction=np.max(self.image_shape)
            if tmp_max_direction>self._maxWidth:
                # Resize the image and saved it to the tmp directory
                tmp_ratio=self._maxWidth/tmp_max_direction
                size=(int(tmp_ratio*self.image_shape[1]),int(tmp_ratio*self.image_shape[0]))
                tmp_img=cv2.resize(self.image, size, interpolation=cv2.INTER_CUBIC) 
                if not os.path.exists(self.tmp_path):    
                    os.makedirs(self.tmp_path)
                cv2.imwrite(self.tmp_path+os.path.sep+'tmp.png',tmp_img)
                # Open the resized image as an image class
                self.imageClass = QtGui.QImage(self.tmp_path+os.path.sep+'tmp.png')
            else:
                # Directly open the image
                self.imageClass = QtGui.QImage(self.filename)
            # Show the image in the dialog window
            self.imageView.setPixmap(QtGui.QPixmap.fromImage(self.imageClass))
            self.resize(self.imageClass.width(), self.imageClass.height())
            print('Image loaded.\n')
        else:
            print('Image unload.\n')
            
            
    def on_Run_clicked(self):
        print('Image pre-processing...')
        ##########################################################################
        #########   Put the pre-processing code in the blank below ###############
        #                                                                        #
        #                                                                        #
        #                                                                        #
        #                                                                        #
        #   The variable self.image's value is the original image
        #   The pre-processing code can take the self.image as the input
        #   And output the pre-processed image to the variable self.processed_image
        #   Here, I don't have the pre-processed code,so I directly gives the 
        #       original image to the variable self.processed_image
        #   
        #                                                                        #
        #                                                                        #
        #                                                                        #
        #                                                                        #
        #                                                                        #
        #                                                                        #
        #                                                                        #
        #                                                                        #
        #                                                                        #
        #                                                                        #
        #                                                                        #
        #                                                                        #
        h, w, _ = self.image.shape
        #img_gray = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
        im=np.copy(self.image)
        im=np.copy(im[:,:,0])
        # turn the black pixel to white
        for i in range (0,2047):
            for j in range (0,2047):
                if (im[i,j] < 50):
                    im[i,j] = 255
                else: continue
        # cover the margin
        for i in range (0,450):
            for j in range (0,2047):
                im[i,j] = 255
        for i in range (1700,2047):
            for j in range (0,2047):
                im[i,j] = 255
        for i in range (0,200):
            for j in range (0,900):
                im[j,i] = 255
        for i in range (1800,2047):
            for j in range (0,900):
                im[j,i] = 255
        for i in range (1800,2047):
            for j in range (1500,2047):
                im[j,i] = 255
        for i in range (0,200):
            for j in range (1500,2047):
                im[j,i] = 255
        
        
        ret,thresh = cv2.threshold(im,140,255,cv2.THRESH_TOZERO_INV)
        ##cv2.imwrite("thresh.png", thresh)
        thresh_blur = cv2.blur(thresh,(10,10)); 
        #blur
        ##cv2.imwrite("thresh_blur.png", thresh_blur)
        _, contours, hierarchy = cv2.findContours( thresh_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        # Find Contour get the outline
        # define a list for cv2.drawContours()
        # c_max = []
        #Put the group of area those>1/100 in list c_max
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)  
            if(area < (h/10*w/10)):
                c_min = []
                c_min.append(cnt)
                #When thickness =!-1,draw outline
                cv2.drawContours(thresh_blur, c_min, -1, (0,0,0), thickness=-1)
                #Turn the small area into black
                continue
        #    c_max.append(cnt)
        ##cv2.imwrite("thresh1_flat.png", thresh_blur)
        #cv2.drawContours(thresh_blur, c_max, -1, (255, 255, 255), thickness=-1)
        
        #turn large area into white
        ##cv2.imwrite("thresh1_white.png", thresh_blur)
        #lock down all the space
        im1 = np.array(thresh_blur)
        im1[:,0] = 255
        im1[0,:] = 255
        im1[2047,:] = 255
        im1[:,2047] = 255
        thresh_blur = im1

        _, contours, hierarchy = cv2.findContours( thresh_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        d_max = []

        #Select small area again
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if(area < (h/10*w/10)):
                d_min = []
                d_min.append(cnt)
                cv2.drawContours(thresh_blur, d_min, -1, (255,255,255), thickness=-1)
                #Turn the small area into white
                continue
            d_max.append(cnt)
        ##cv2.imwrite("thresh1_flat2.png", thresh_blur)    
        _, contours1, hierarchy1 = cv2.findContours(thresh_blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        #codes to draw the right outline on the initial image
        '''
        c_target = []
        for j in range(len(contours1)):
            cnt1 = contours1[j]
            area1 = cv2.contourArea(cnt1)
            if(area1 > (h/4*w/4) ):
                continue
            c_target.append(cnt1)
            #Put the group of area<1/16 in list c_target
        #Draw the final outline - cv2.drawContours(img,c_target,-1,(0,0,255),5)
        #You can see the outline via - cv2.imwrite(new_name, img)
        '''
        x1=0 # Lazy method to get the position of the rectangle
        y1=0
        w1=0
        h1=0
        count = 0
        for i in range(0,len(contours1)): 
            x, y, w, h = cv2.boundingRect(contours1[i])  
            #The rectangle contains the target area
            #cv2.rectangle(img, (x,y), (x+w,y+h), (153,153,0), 5)
            if (h>800):
                continue
            else: 
                x1=x
                y1=y
                w1=w
                h1=h
                self.processed_image= self.image[y-20:y+h+20,x:x+w]
                #cv2.imwrite(new_name, sub_img)
                count = count + 1
                #Cut and save the weld area
        if (count==0):
            self.processed_image= self.image[950:1400,:]
            y1=970
            h1=410
            x1=0
            w1=2048
            #cv2.imwrite(new_name, sub_img)
        # The upper point of the left side is located at (x,y-20)
        # height = h+40,width = w
        
        
        #self.processed_image=np.copy(self.image)
        self.p_image_shape=np.shape(self.processed_image)
        #                                                                        #
        #                                                                        #
        #                                                                        #
        #                                                                        #
        ##########################################################################
        print('Image pre-processing is done.\n')
        
        print('The recognition process is now running...')
        # Determine the total blocks that required
        self.extra_y=int(not(np.mod(self.p_image_shape[0],(self.block_size[0]-self.overlap[0]))==0))
        self.extra_x=int(not(np.mod(self.p_image_shape[1],(self.block_size[1]-self.overlap[1]))==0))
        self.y_b=int((self.p_image_shape[0]-self.new_block_size[0])/(self.block_size[0]-self.overlap[0]))+self.extra_y
        self.x_b=int((self.p_image_shape[1]-self.new_block_size[1])/(self.block_size[1]-self.overlap[1]))+self.extra_x
        
        # Determine the position of each small block
        self.pt=np.array(np.zeros([self.y_b+2,self.x_b+2,2])).astype(np.int32)
        for i in range(self.x_b+2):
            for j in range(self.y_b+2):
                self.pt[j,i,0]=j*self.new_block_size[0] # height direction
                self.pt[j,i,1]=i*self.new_block_size[1] # width direction
        if self.extra_x:
            for j in range(self.y_b+2):
                self.pt[j,self.x_b+1,1]=self.p_image_shape[1]
        if self.extra_y:
            for i in range(self.x_b+2):
                self.pt[self.y_b+1,i,0]=self.p_image_shape[0]
        
        # Initializing the defect labels and the result image
        self.defect_label=np.array(np.zeros([self.y_b+1,self.x_b+1])).astype(np.int32)
        
        
        TotalWork=self.x_b*self.y_b
        # The defects detection process
        with tf.Session(graph=self.graph1) as sess1:
            with tf.Session(graph=self.graph2) as sess2:
                y_start=0
                for j in range(self.y_b):
                    x_start=0
                    for i in range(self.x_b):
                        
                        # Extracting the block that is going to detect
                        sub_img=np.copy(self.processed_image[y_start:y_start+self.block_size[0],x_start:x_start+self.block_size[1],:])

                        # Rolling the window in the width direction
                        if self.extra_x==1 and i==self.x_b-2:
                            x_start=self.p_image_shape[1]-self.block_size[1]
                        else:
                            x_start=x_start+self.block_size[1]-self.overlap[1]
                                    
                        # Get the result of this block
                        res=int(self.model.predict(sub_img,sess1,sess2,self.threshold))
                        if not res==0:# means that this block is abnormal
                            self.defect_label[j][i]=res
                            self.defect_label[j+1][i]=res
                            self.defect_label[j][i+1]=res
                            self.defect_label[j+1][i+1]=res
                            
                        
                        # The recognition progress
                        ProgressBar=(j*self.x_b+i+1)/TotalWork
                        print('Current Progress:',round(ProgressBar*100,2),' %')
                        
                        # Visualizing this block
                        # Determine the start point and span of height and width of the rolling window
                        if i-self.determine_start[1]<0:
                            i1=0
                        else:
                            i1=i-self.determine_start[1]
                        if j-self.determine_start[0]<0:
                            j1=0
                        else:
                            j1=j-self.determine_start[0]
                        self.Visualize_The_Results(i1,self.rolling_size[1],j1,self.rolling_size[0])
                        rolling_img_path=self.tmp_path+os.path.sep+'_'+str(j)+'_'+str(i)+'_'+str(self.y_ss)+'_'+str(self.x_ss)+'_'+str(self.deltayy)+'_'+str(self.deltaxx)+'.jpg'
                        if not os.path.exists(self.tmp_path):
                            os.makedirs(self.tmp_path)
                        rec1=(self.overlap[1]*(i-self.x_ss),self.overlap[0]*(j-self.y_ss))
                        rec2=(rec1[0]+self.block_size[1],rec1[1]+self.block_size[0])
                        # Blue rectangle mark the block that the app is working at the moment
                        cv2.rectangle(self.rolling_window,rec1,rec2,(255,255,0),5)
                        cv2.imwrite(rolling_img_path,self.rolling_window)
                        self.imageClass = QtGui.QImage(rolling_img_path)
                        self.imageView.setPixmap(QtGui.QPixmap.fromImage(self.imageClass))
                        self.resize(self.imageClass.width(), self.imageClass.height())
                        
                        QtWidgets.QApplication.processEvents() # Refreshing the dialog window
                                   
                    # Rolling the windows in the height direction
                    if self.extra_y==1 and j==self.y_b-2:
                        y_start=self.p_image_shape[0]-self.block_size[0]
                    else:
                        y_start=y_start+self.block_size[0]-self.overlap[0]
                    
        
        # Visualize the whole image
        self.Visualize_The_Results(0,self.x_b+1,0,self.y_b+1)
        self.res_image[y1-20:y1+h1+20,x1:x1+w1]=np.copy(self.rolling_window)
        cv2.rectangle(self.res_image,(x1,y1-20),(x1+w1,y1+h1+20),(255,255,0),2)
        
        # Presenting the final image
        rolling_img_path=self.tmp_path+os.path.sep+'FinalImg.jpg'
        tmp_shape=np.shape(np.array(self.res_image))
        max_side=np.max(tmp_shape)
        tmp_ratio=self._maxWidth/max_side
        size=(int(tmp_ratio*tmp_shape[1]),int(tmp_ratio*tmp_shape[0]))
        self.rolling_window=cv2.resize(self.res_image, size, interpolation=cv2.INTER_CUBIC) 
        cv2.imwrite(rolling_img_path,self.rolling_window)
        self.imageClass = QtGui.QImage(rolling_img_path)
        self.imageView.setPixmap(QtGui.QPixmap.fromImage(self.imageClass))
        self.resize(self.imageClass.width(), self.imageClass.height())
        QtWidgets.QApplication.processEvents()
        
        print('\nThe recognition progress is finished.\n')
        print('Now you can export the finished image or open another image.\n')
        
        # Delete the tmp file
        shutil.rmtree(self.tmp_path)
        
        
    def on_Export_clicked(self):
        self.export_filename=QtWidgets.QFileDialog.getSaveFileName(self, "SaveFile", ".", 
            "Image Files(*.jpg *.jpeg *.png *.bmp)")[0]
        if len(self.export_filename):
            cv2.imwrite(str(self.export_filename),self.res_image)
        print('Image is exported to path:',str(self.export_filename))
    
    
    def Visualize_The_Results(self,x_s,deltax,y_s,deltay):
        
        # The input size maynot be very well
        if x_s+deltax>self.x_b+1:
            x_s=self.x_b+1-deltax
            if x_s<0:
                x_s=0
                deltax=self.x_b+1
        if y_s+deltay>self.y_b+1:
            y_s=self.y_b+1-deltay
            if y_s<0:
                y_s=0
                deltay=self.y_b+1
        self.x_ss=x_s
        self.y_ss=y_s
        self.deltayy=deltay
        self.deltaxx=deltax
        
        
        # Get the new label
        # For the convinience of the visualization
        tmp_whateverTheName=np.copy(self.defect_label[y_s:y_s+deltay,x_s:x_s+deltax])
        label_size=np.shape(np.array(tmp_whateverTheName))
        new_label_size=[label_size[0]+2,label_size[1]+2]
        tmp_defect_label=np.array(np.zeros(new_label_size)).astype(np.int32)
        tmp_defect_label[1:1+label_size[0],1:1+label_size[1]]=tmp_whateverTheName
        
        # Get the sub image
        self.rolling_window=np.copy(self.processed_image[self.pt[y_s,x_s,0]:self.pt[y_s+deltay,x_s+deltax,0],
                                       self.pt[y_s,x_s,1]:self.pt[y_s+deltay,x_s+deltax,1],:])
        
        pt_start=[self.pt[y_s,x_s,0],self.pt[y_s,x_s,1]]
        
        #visualize the results,if there's defects in the sub_image,framed it out
        for j in range(label_size[0]):
            for i in range(label_size[1]):
                
                #firstly,check if there's defects in this block
                if tmp_defect_label[j+1][i+1]==0:
                    pass
                else:
                    #pt:first y then x
                    pos_index=[j,i,j,i+1,j+1,i+1,j+1,i]
                    choice_index=[j,i+1,j+1,i+2,j+2,i+1,j+1,i]
                    #the start point and end point the line to be drawn
                    pt1=[self.pt[y_s+pos_index[2*int(k/2)],x_s+pos_index[2*int(k/2)+1],np.mod(k,2)] for k in range(8)]
                    #to decide whether to draw the line or not in the four direction
                    draw_choice=[tmp_defect_label[choice_index[2*k]]
                                        [choice_index[2*k+1]] for k in range(4)]
                    #draw a rectangle to framed out this block
                    #but not necessarily draw all four lines of this block
                    for k in range(4):
                        #only when the adjacent block is normal,we draw the line
                        if draw_choice[k]==0:
                            #the first parametre for axis x,then y
                            cv2.line(self.rolling_window,(pt1[2*k+1]-pt_start[1],pt1[2*k]-pt_start[0]),
                                     (pt1[np.mod(2*k+3,8)]-pt_start[1],pt1[np.mod(2*k+2,8)]-pt_start[0]),
                                                   self.color_set[tmp_defect_label[j+1][i+1]-1],3)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv) # Creating an instance of QApplication
    dfcw = DefectClassWindow() # Creating an instance of DefectClassWindow
    dfcw.show() # Presenting the main dialog window
    app.exec_() # Looping