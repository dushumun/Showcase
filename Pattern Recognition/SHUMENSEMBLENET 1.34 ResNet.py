##tkinter imports
from tkinter import *
from tkinter import filedialog
from keras.preprocessing import image
##keras imports
from keras.models import load_model
import numpy as np
from PIL import Image, ImageTk

print("Loading Model")
model = load_model('vgg16_OS_e10_lr001_froze3.h5')
model.summary()

print("Loading Model 2")
model2 = load_model('incepv3_OS_e10_lr001_frozen0.h5')
model2.summary()

#image browse function
def imagebrowse():
    root.fileName = filedialog.askopenfilename(filetypes = (("jpeg", "jpg"),("All Files", "*.*")))
    global image_path
    global im_update 
    image_path = root.fileName
    print("File Selected: " + image_path)
    image_path = image_path
    v.set(image_path)
    vggpred.set("")
    inceppred.set("")
    shumout.set("")
    im_update = ImageTk.PhotoImage(Image.open(image_path))
    cpic_label.configure(image = im_update)
    return



#classify function
def classify(img):
    print(img)
    
    # importing and normalising of values
    img = image.load_img(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x/255
    
    # VGG_16 Model Prediciton
    pred_vgg16 = model.predict(x)
    pred_vgg16.tolist()

    predlist_vgg16 = pred_vgg16[0]
    print("vgg_16 softmax output")
    print(predlist_vgg16)

#    for i in predlist_vgg16:
#        for x in i:
#            print(x)
    
    missing_vgg16 = predlist_vgg16[0]
    no_issue_vgg16 = predlist_vgg16[1]
    obscured_vgg16 = predlist_vgg16[2]

    print(missing_vgg16)
    print(no_issue_vgg16)
    print(obscured_vgg16)
    print("")
    print("VGG-16 Prediction for image: " + image_path)
    ## VGG_16 Prediction
    if missing_vgg16 > no_issue_vgg16 and missing_vgg16 > obscured_vgg16:
        print("VGG-16 Model Predicts the fastener is missing")
        vggpred.set("VGG-16 Model Predicts the fastener is missing")
    elif no_issue_vgg16 > missing_vgg16 and no_issue_vgg16 > obscured_vgg16:
        vggpred.set("VGG-16 Model Predicts their is no issue with the fastener")
        print("VGG-16 Model Predicts their is no issue with the fastener")
    elif obscured_vgg16 > missing_vgg16 and obscured_vgg16 > no_issue_vgg16:
        vggpred.set("VGG-16 Model cannot make prediction, track obscured")
        print("VGG-16 Model cannot make prediction, track obscured")
    
    #InceptionV3 Model Prediciton
    pred_incep3 = model2.predict(x)
    pred_incep3.tolist()
    predlist_incep3 = pred_incep3[0]
    print("ResNet-50 softmax output")
    print(predlist_incep3)
    
    missing_incep3 = predlist_incep3[0]
    no_issue_incep3 = predlist_incep3[1]
    obscured_incep3 = predlist_incep3[2]
    
    print("ResNet-50 Prediction for image: " + image_path)
 
    if missing_incep3 > no_issue_incep3 and missing_incep3 > obscured_incep3:
        print("ResNet-50 Model Predicts the fastener is missing")
        inceppred.set("ResNet-50 Predicts the fastener is missing")
    elif no_issue_incep3 > missing_incep3 and no_issue_incep3 > obscured_incep3:
        print("ResNet-50 Model Predicts their is no issue with the fastener")
        inceppred.set("ResNet-50 Model Predicts their is no issue with the fastener")
    elif obscured_vgg16 > missing_vgg16 and obscured_vgg16 > no_issue_vgg16:
        print("ResNet-50 Model cannot make prediction, track obscured")    
        inceppred.set("ResNet-50 Model cannot make prediction, track obscured")
    

    #Ensemble Predicitons
    missing_ensemble = (missing_vgg16 + missing_incep3)/2
    no_issue_ensemble = (no_issue_vgg16 + no_issue_incep3)/2
    obscured_ensemble = (obscured_vgg16 + obscured_vgg16)/2
    print("ShumEnsemble prediciton for image: " + image_path)
    
    if missing_ensemble > no_issue_ensemble and missing_ensemble > obscured_ensemble:
        print("ShumEnsemble Model Predicts the fastener is missing")
        shumout.set("ShumEnsemble Model Predicts the fastener is missing")
    elif no_issue_ensemble> missing_ensemble and no_issue_ensemble > obscured_ensemble:
        print("ShumEnsemble Model Predicts their is no issue with the fastener")
        shumout.set("ShumEnsemble Model Predicts their is no issue with the fastener")
    elif obscured_ensemble > missing_ensemble and obscured_ensemble > no_issue_ensemble:
        print("ShumEnsemble Model cannot make prediction, track obscured")   
        shumout.set("ShumEnsemble Model cannot make prediction, track obscured")

  
    

#start of gui loop
root = Tk()
root.title("ShumEnsemble 1.1")
#image browse button

frame = Frame(root, width = 250, height = 250)
browse_button = Button(root, text = "Browse Images", command = imagebrowse)
browse_button.grid(row=0, column=0, padx=2, pady=2)
#classify browse button
classify_button = Button(root, text = "Classify Images", command = lambda: classify(image_path))
classify_button.grid(row=0, column=1, padx=2, pady=2)
# display image
display_pic = ImageTk.PhotoImage(Image.open("coverimage.jpg"))
panel = Label(root, image = display_pic)
panel.grid(row=7,column=0, columnspan=2)
## Selected image
statuslabel = Label(root, text = "Image Selected", anchor = W)
statuslabel.grid(row=2, column=0)
v = StringVar()
status = Label(root, textvariable = v, bd=1, relief=SUNKEN, anchor=W)
v.set("No Image Selected")
status.grid(row=2, column=1, padx=2, pady=2)
##VGG output
vgglabel = Label(root, text = "VGG Prediction", anchor = W)
vgglabel.grid(row=3, column = 0)

vggpred = StringVar()
vgg = Label(root, textvariable = vggpred, bd=1, relief = SUNKEN, anchor = W)
vgg.grid(row=3, column = 1, padx=2, pady=2)
vggpred.set("")
 ## incepv3 output
inceplabel = Label(root, text = "ResNet-50 Prediction", anchor = W)
inceplabel.grid(row=4, column=0)

inceppred = StringVar()
incep = Label(root, textvariable = inceppred, bd=1, relief = SUNKEN, anchor = W)
incep.grid(row=4, column = 1, padx=2, pady=2)
inceppred.set("")

#Shumensemble output
shumunlabel = Label(root, text = "ShumEnsemble Prediction", anchor = W)
shumunlabel.grid(row=5, column=0)

shumout = StringVar()
shumunoutput = Label(root, textvariable=shumout, bd=1, relief = SUNKEN, anchor = W)
shumunoutput.grid(row=5, column = 1)
shumout.set("")


## display selected image
current_label = Label(root, text= "Selected Image", anchor = E)
current_label.grid(row = 6, column = 0)

cpic = ImageTk.PhotoImage(Image.open("C:/Users/DUSHUMUN/UNAY/Final Project/active set/noImageSelected.jpg"))
cpic_label = Label(root, image = cpic, padx=2, pady=2, anchor = W)
cpic_label.grid(row=6, column=1)

 
root.mainloop() # end of gui loop


