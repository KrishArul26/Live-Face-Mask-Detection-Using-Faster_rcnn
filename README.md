<h2 align="center"> Live-Face-Mask-Detection-Using-Faster_rcnn</h2>

<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/74568334/140817632-aa63c056-07be-4a14-9620-a34258deab31.png">
</p> 

<h3 align="left"> Fast R-CNN Quick Overview </h3>

<p style= 'text-align: justify;'> **Fast R-CNN** is an object detector that was developed solely by Ross Girshick, a Facebook AI researcher and a former Microsoft Researcher. Fast R-CNN overcomes several issues in R-CNN. As its name suggests, one advantage of the Fast R-CNN over R-CNN is its speed. Here is a summary of the main contributions in 
  
 1. Proposed a new layer called **ROI Pooling** that extracts equal-length feature vectors from all proposals (i.e. ROIs) in the same image.
    Compared to **R-CNN**, which has multiple stages (region proposal generation, feature extraction, and classification using SVM), Faster R-CNN builds a network that has only     a single stage.
2.  **Faster R-CNN** shares computations (i.e. convolutional layer calculations) across all proposals (i.e. ROIs) rather than doing the calculations for each proposal                 independently. #
3.  This is done by using the new **ROI Pooling layer**, which makes **Fast R-CNN faster than R-CNN**.
4.  **Fast R-CNN** does not cache the extracted features and thus does not need so much disk storage compared to **R-CNN**, which needs hundreds of gigabytes.
5.  **Fast R-CNN is more accurate than R-CNN**.
6.  The general architecture of Fast R-CNN is shown above. The model consists of a single-stage, compared to the 3 stages in R-CNN. It just accepts an image as an input and         returns the class probabilities and bounding boxes of the detected objects..</p>


<h3 align="left"> Problem Statment</h3>

<p style= 'text-align: justify;'> Due to the COVID-19 regulation, people have to wear masks to protect themselves. Also, you need to create a system to find out if a person is wearing a mask or not. Also, this system should give a warning or sound when people are not wearing masks. So, I created the model with the help of Fast-R-CNN.</p>


<p align="center">
  <img width="500" src="https://user-images.githubusercontent.com/74568334/123329616-4b6ba800-d53d-11eb-98f7-8006730c863c.jpg">
</p> 


<h4 align="center"> <span style="color:green">Face Mask Detection system built with OpenCV, Keras/TensorFlow using Deep Learning and Computer Vision concepts like faster-rcnn in order to detect face masks in real-time video streams.</span></h4>

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/123328085-7228df00-d53b-11eb-9bc0-0700343af177.gif">
  <img width="350" src="https://user-images.githubusercontent.com/74568334/123328490-e9f70980-d53b-11eb-8980-d0e0e2ab694f.gif">
  
</p> 


### 📁 Data Collection

This project has done up to 50000 epochs with error 0.08 values.Futher, 2200 images were collected among them 1000 images are without mask and 1200 images are with mask.
* With Mask: 1200
* Without Mask: 1000

### 🔑 Prerequisites
* All the dependencies and required libraries are included in the file [requirements.txt](https://github.com/KrishArul26/Live-Face-Mask-Detection-Using-Faster_rcnn/blob/main/requirements.txt)

### 🚀 Installation For Live Mask Detection

1. Clone the repo

* git clone https://github.com/KrishArul26/Live-Face-Mask-Detection-Using-Faster_rcnn.git

2. Change your directory to the cloned repo

```
cd Live-Face-Mask-Detection-Using-Faster_rcnn
```

3. Create a Python 3.6 version of  virtual environment named 'mask' and activate it

 ``` 
pip install virtualenv

 ```

* Create virtual environmental

```
virtualenv mask

```
* Activate that environmental

```
mask\Scripts\activate

```

4. Now, run the following command in your Terminal/Command Prompt to install the libraries required

```
pip install -r requirements.txt

```
### 💡 Working

1. Open terminal. Go into the cloned project directory and type the following command:

```
python mask_detection_video.py

```
### 🔑 Results 

* For this mask detection I have used computer vision trained net work which is faster-rcnn

### 🚀 Installation For Mask Detection for images

1. Clone the repo

* git clone https://github.com/KrishArul26/Live-Face-Mask-Detection-Using-Faster_rcnn.git

2. Change your directory to the cloned repo

```
cd Live-Face-Mask-Detection-Using-Faster_rcnn
```

3. Create a Python 3.6 version of  virtual environment named 'mask' and activate it

 ``` 
pip install virtualenv

 ```

* Create virtual environmental

```
virtualenv mask

```
* Activate that environmental

```
mask\Scripts\activate

```

4. Now, run the following command in your Terminal/Command Prompt to install the libraries required

```
pip install -r requirements.txt

```
### 💡 Working

1. Open terminal. Go into the cloned project directory and type the following command:

```
python mask_detection_image.py

```
### 🔑 Results 

#### Testing-1

<p align="left">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/123304124-d7baa280-d51e-11eb-983c-8945260928e1.jpg">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/123304146-e012dd80-d51e-11eb-9aa1-94b24b78ac59.jpg">
</p> 

#### Testing-2

<p align="left">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/123304686-8828a680-d51f-11eb-8084-14f25b735e70.jpg">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/123304690-8a8b0080-d51f-11eb-88f6-19b7997d573d.jpg">
</p> 

#### Testing-3

<p align="left">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/123307848-43067380-d523-11eb-8978-a8217559b142.jpg">
  <img width="350" src="https://user-images.githubusercontent.com/74568334/123307846-426ddd00-d523-11eb-9af8-233ea4838198.jpg">
</p> 

