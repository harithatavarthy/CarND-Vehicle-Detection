##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png

[image8]: ./output_images/HOG_image.png
[image9]: ./output_images/car_detection_test_images.png
[image10]: ./output_images/heatmap_test_images.png
[image11]: ./output_images/multiscale_windows.png
[image12]: ./output_images/sliding_window.png


[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the  code cell IN-5 of the IPython notebook `project.ipynb`

I started by reading in all the `vehicle` and `non-vehicle` images provided by Udacity for training the Linear SVM. The code to train the SVM can be found in code cell IN-207 of ipython notebook `project.ipynb`


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(4, 4)` and `cells_per_block=(2, 2)`:


![alt text][image8]

####2. Explain how you settled on your final choice of HOG parameters.

Initially i tried to train the linear SVM using HOG features obtained from various color spaces . However RGB Color space gave me an better accuracy ( `99.6%`) over other color spaces and for that reason I have fixed on to 'RGB' color space. Then I experimented with various combinations of HOG parameters including a pixels_per_cell combination of both 8 and 16. Though the value '8' gave me a better accuracy over '16', it took a lot longer for the SVM to be trained as well as for it to predict the features. I have chosen a pixel_per_cell value of '16' there by sacrifising tiny bit of diffence in accuracy in order to reap the benefit of gain in overall performance.

  `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Though Udacity suggested to use LinearSVM for this project, i  experimented with various other classifiere (including naive bayes, SVM with Grid Search and Decision Trees). Training took a lot longer using Decision Tree compared to SVM and Naive Bayes performed poorly over SVM. For that reason , i decided to use SVM and tried to experiment with various parameter combinations of SVM through Grid Search and found out that Linear SVM with a 'C' value of 1.0 is preferred over other combination.

I have hence used LinearSVM to train my classifier.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Initially i have used the udacity supplied code to perform sliding window search . You can see the `slide_window`function defined in code cell IN-383 of Ipython notebook `project.ipynb`.  As you can see in the visual below, this function can be called to perform fixed sized window size on an image. However for multi scale window search, this function will not work. 


![alt text][image12]

Later, i have defined a function by name `find_cars` that can peform both feature_extraction as well as multi scale search on a given image channel. The code for this function can be found on code cell IN-319 of Ipython notebook `project.ipynb`. This function when called with appropriate input parameters will return a list of multi size window positions that contain the detected vehicles. Below visual shows the  multi scale window (irrespective of whether cars are detected or not) search process on an image channel.

![alt text][image11]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Though i experimented with various color spaces and parameters, I have ultimately used 'RGB' color space to extract Color and HOG features. To improve the SVM training speed, i have set the pixels_per_cell value to '16' (each cell is 16x16 pixels). By doing so, the training speed improved by five fold. Also, the speed at which the prediction are made increased by three fold.

My pipeline to predict the vehicles on test images is as follows -
 a. For each test image, I will call the function `vehicle_detection_tracking_img()` . This function will inturn call function `find_cars()` for different sections of the input image with different scaling size , which essentially is the core of multi scale window search.
 b. Each time `find_cars()` funtion is called, it returns a list of cordinates with detected vehicles. I will save these coordinates for different window scales and will later use HEATMAPS with thresholds to  merge overlapping windows and remove false positives.
 c. Finally, i will label the windows and draw boxes/widows over the image to represent the detected vehicles.
 
 Below is an example of boxes drawn over the detected vehicles on test images.

![alt text][image9]


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` 

### Here are heatmaps corresponding to the six test_images provided by udacity.

![alt text][image10]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

