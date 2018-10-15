# Weld-Defect-Recognition-App---betav1.1

## 0.About this app

Project name: Weld defect recognition

Version: Beta 1.1


## 1.Operating system

This app can be run on 64bit **windows** operating system, see the section 1~3 to know how to use this app.
 <br>
 <br>

For **Linux** users, I believe that this instruction is totally useless for you, and I also believe that you have already installed everything that is needed for a tensorflow program, so I suggest that you go to another folder named "ForDeveloper", then use the file "GUI_beta_v1_0.py" to run the app on your Linux.

The following instructions is meant for windows operating system users.

## 1.Installation

No need to install anything. All you have to do is to place the folder "App_beta_v1_1" in your computer. 

Noticing: The path that you place the app should not contain any Chinese characters. Also, for windows user, it's better not put this folder in the C:drive(otherwise may cause some operating system permission issues).


## 2.Get start

The entrance of this app is located at:

```
the path you place the folder\App_beta_v1_1\beta1_1.exe
```

Just double left click the beta1_0.exe, then the program get start. Or, it's better to right click the beta1_0.exe,then 'Run as administer'(以管理员身份运行).

Noticing: The anti-virus software like 360 may report a virus, **ignore it**. The app will not do any harm to your computer, don't worry about it.


## 3.How to use the app

If the users open the app successfully, you shall see something like this:

<div align=center>
<img src="https://github.com/ChenZuzhi/Weld-Defect-Recognition-App---betav1.1/blob/master/ImgsForReadme/1.png">
</div>
 <br>
 <br>
 <br>

Then the users can click the 'select an image' button to open an image(Noticing that *.jpg, *.jpeg, *.png, *.bmp image format are supported):

<div align=center>
<img src="https://github.com/ChenZuzhi/Weld-Defect-Recognition-App---betav1.1/blob/master/ImgsForReadme/2.png">
</div>
 <br>
 <br>
 <br>

If the image is opened successfully, the users shall see something like this:

<div align=center>
<img src="https://github.com/ChenZuzhi/Weld-Defect-Recognition-App---betav1.1/blob/master/ImgsForReadme/3.png">
</div>
 <br>
 <br>
 <br>
 
Now the image is ready, the users can click the 'run' button to run the recognition code.

It may take some time to finish the entire progress. Usually, the app will only use the CPU to do the computing, so it will be kind of a slow. 

If the users want to use the GPU computing for this app. Firstly, please make sure that your GPU supports GPU computing. Secondly, you need to install the corresponding version of Cuda and Cudnn to your computer. I recommend that the version of Cuda is 9.0 and the version of Cudnn is 7.0.5.

The running processing looks like this, the blue rectangle marks at which block the app is working at the moment.

<div align=center>
<img src="https://github.com/ChenZuzhi/Weld-Defect-Recognition-App---betav1.1/blob/master/ImgsForReadme/4.png">
</div>
 <br>
 <br>
 <br>

When the progress is done. The users will see the final image in the dialog window.

The users can export the image to anywhere that makes them happy.

The finished dialog looks like this, the blue rectangle in this image shows the area that the app involving:

<div align=center>
<img src="https://github.com/ChenZuzhi/Weld-Defect-Recognition-App---betav1.1/blob/master/ImgsForReadme/5.png">
</div>
 <br>
 <br>
 <br>

Clicking the 'export' button then the users can export the finished image:

<div align=center>
<img src="https://github.com/ChenZuzhi/Weld-Defect-Recognition-App---betav1.1/blob/master/ImgsForReadme/6.png">
</div>
 <br>
 <br>
 <br>

That's it. After that, the users can choose to open another image or exit the app.
