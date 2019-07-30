# **Finding Lane Lines on the Road** 


**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[img_steps_grayscale]: ./img_results/steps_gray.png "Grayscale"
[img_steps_blurred]: ./img_results/steps_blurred.png "Blurred"
[img_steps_canny]: ./img_results/steps_canny.png "Canny"
[img_steps_masked]: ./img_results/steps_masked.png "Masked"
[img_steps_hough_lines]: ./img_results/steps_hough_lines.png "Hough lines"
[img_steps_weighted]: ./img_results/steps_weighted.png "Original image with lines"

[img_ld_01]: ./img_results/line_detection_01.png "Line detection - example 1."
[img_ld_02]: ./img_results/line_detection_02_yellow.png "Line detection - example 2."
[img_ld_03]: ./img_results/line_detection_03_yellow_curve.png "Line detection - example 3."
[img_ld_04]: ./img_results/line_detection_04_yellow_curve2.png "Line detection - example 4."
[img_ld_05]: ./img_results/line_detection_05_white_curve.png "Line detection - example 5."
[img_ld_06]: ./img_results/line_detection_06_car_switch.png "Line detection - example 6."

[img_rl_01]: ./img_results/right_left_01_white.png "Right/Left detection - example 1."
[img_rl_02]: ./img_results/right_left_03_yellow_curve.png "Right/Left detection - example 2."
[img_rl_03]: ./img_results/right_left_06_car_switch.png "Right/Left detection - example 3."

[img_extrapolation_01]: ./img_results/extrapolation_01_white.png "Line extrapolation - example 1."
[img_extrapolation_02]: ./img_results/extrapolation_02_yellow.png "Line extrapolation - example 1."
[img_extrapolation_03]: ./img_results/extrapolation_03_yellow_curve.png "Line extrapolation - example 1."
[img_extrapolation_04]: ./img_results/extrapolation_04_yellow_curve2.png "Line extrapolation - example 1."
[img_extrapolation_05]: ./img_results/extrapolation_05_white_curve.png "Line extrapolation - example 1."
[img_extrapolation_06]: ./img_results/extrapolation_06_car_switch.png "Line extrapolation - example 1."


---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

#### Building the pipeline

Firstly, I've had to build the required pipeline which would apply the different helping functions provided in the notebook, in the correct order, which would be the following:

1. **Color Image to grayscale.**
    The first thing we do, as always is to convert the RGB original image to a one channel grayscale image. This corresponds with the `grayscale` helping function. The result would seem something like this:
    
    ![img_steps_grayscale]

2. **Blur image.**
    Similar to the course exercises, we apply a Gaussian filter. This corresponds with the `gaussian_blur` helping function. 
    
    ![img_steps_blurred]
    
3. **Canny edge detector.**
    We apply the canny algorithm to detect edges on the image. This corresponds with the `canny` helping function. 

    ![img_steps_canny]

4. **Region of interest.**
    We only take those regions of the image that could be of interest for us, to prevent other information to introduce noise in the process. This corresponds with the `region_of_interest` helping function. 
    
    ![img_steps_masked]

5. **Hough lines.**
    Next, we use the helping function in charge of detecting straight lines in the picture. This corresponds with the `hough_lines` helping function, which at the same time uses the `draw_lines` function.
    
    ![img_steps_hough_lines]

6. **Weighted image.**
    Finally, we apply the `weighted_img` function to draw the lines over the original image. This corresponds with the `weighted_img` helping function.
    
    ![img_steps_weighted]


So our pipeline will be a function that applies all these helping function in this specified order. This is done in the notebook with the implementation of the following code:

```python
def lines_pipeline(img, params):
    canny_low_threshold = params['canny_low_threshold']
    canny_high_threshold = params['canny_high_threshold']
    gaussian_kernel_size = params['gaussian_kernel_size']
    roi_vertices = params['roi_vertices']
    hough_rho = params['hough_rho']
    hough_theta = params['hough_theta']
    hough_threshold = params['hough_threshold']
    hough_min_line_len = params['hough_min_line_len']
    hough_max_line_gap = params['hough_max_line_gap']
    draw_all_lines = params['draw_all_lines']
    draw_extrapolation = params['draw_extrapolation']
    
    img_gray = grayscale(img)
    img_blurred = gaussian_blur(img_gray, gaussian_kernel_size)
    img_canny = canny(img_blurred, canny_low_threshold, canny_high_threshold)
    img_masked = region_of_interest(img_canny, roi_vertices)
    img_lines = hough_lines(img_masked, hough_rho, hough_theta, hough_threshold, hough_min_line_len, hough_max_line_gap, 
                           y_bottom=np.max(roi_vertices[:,:,1]), y_top = np.min(roi_vertices[:,:,1]),
                           draw_all_lines = draw_all_lines, draw_extrapolation = draw_extrapolation)
    img_weighted = weighted_img(img_lines, img, α=0.8, β=2., γ=0.)
    
    return img_weighted
```

The function takes as an input the image to be processed, and a dictionary that contains the value for all the parameters needed for the helping functions input. We store all these parameters in variables and execute, one by one, all the helping functions in the order specified before. One thing to notice here, is the presence of several new input parameters in the `hough_lines` function (`y_bottom`, `y_top`, `draw_all_lines` and `draw_extrapolation`), but this will be explained later.   

To apply our pipeline, we would do something like this, assigning the desired value for each parameter:

```python
imgshape = image.shape
roi_top = 340
params = {
    "canny_low_threshold": 50,
    "canny_high_threshold": 150,
    "gaussian_kernel_size": 5,
    "roi_vertices": np.array([[(100,imgshape[0]), (400,roi_top), (600,roi_top), (900, imgshape[0])]], dtype=np.int32), # vertices of mask
    "roi_top": roi_top,
    "hough_rho": 2,
    "hough_theta": np.pi/180, 
    "hough_threshold": 25, # min intersections in Hough grid cell
    "hough_min_line_len": 40, # min pixels of a line
    "hough_max_line_gap": 20, # max pixels for gap within line
    "draw_all_lines": False,
    "draw_extrapolation": True
}

processed_img = lines_pipeline(image, params)
plt.imshow(processed_img)
cv2.imwrite('test_images_output/'+img_name, processed_img)
```

With this pipeline and tuned parameters, using the original `draw_lines` function (without extrapolation), these are some of the results we obtain:

![img_ld_01]

![img_ld_02]

![img_ld_03]

![img_ld_04]

![img_ld_05]

![img_ld_06]


#### Improving the `draw_lines()` function

The next task of this project would be to improve the `draw_lines()` function in order to take all the lanes detected an extrapolate them to two lines; one for the left lane and one for the right. These lines should go from the bottom of the region of interest to the top.

The code structure for developing this utility will have the following steps:

1. **Detect right lines and left lines candidates.**
    
    Firstly, we will detect, from all the lines detected in the image, which could correspond to the left lane and which to the right one (and which to neither of them).
    We will do it by establish a range of possible slopes for each lane and accepting only those lines that fit this range, essentially.
    
    To visualize this step, we have coloured the two lines with different colors. A quick implementation of this step would be the following (this would be also enhanced later):
    
    ```python
    def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    m_threshold = 0.5
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1))
            line_color = color
            if slope<(-1*m_threshold):
                line_color = [0, 0, 255]
            elif slope>m_threshold:
                line_color = [0, 255, 0]
            cv2.line(img, (x1, y1), (x2, y2), line_color, thickness)
    ```
    Some of the images obtained with this small function are the following:
    
    
    ![img_rl_01]
    
    ![img_rl_02]
    
    ![img_rl_03]
   
2. **Extrapolate the lines.**

    With the previous step, we have managed to create two different lists of lines; one for the lines corresponding to the left lane, and list of lines corresponding to the right lane.
    We can use these two lists of lines to extrapolate them and obtain two lines of them, one for the left lane and one for the right one.
    
    We have tried different approaches for it (some using the `np.polyfit` function with the points of the lines to get a linear regression of this points), but in most of them we found that, although they worked very well in nearly all the test images and videos of the project, it was a bit more difficult to make them work okay in the *Optional Challenge*. 
    
    At the end, we decided to do the following for extrapoling the lines (giving one of these two lists):
    
    * Calculate mean of slopes.
    * Use only those lines with slope near to the mean (in a certain percentile of distance to mean slope).
    * Take both initial and final points of each line and create a `points` array.
    * Find the line with fixed slope (mean slope) which best fits these points.
    * Finally, from the m*X+b equation, we get the points x_bottom and x_top from the points y_bottom and y_top. All of them combined, will give us the points (x0, y0, x1, y1) that will define the line we are looking for.

So, after improving these attempts, we managed to get the following `draw_lines` function, that we applied to both the regular tests of the project, and the `Optional Challenge`:

```python
def draw_lines(img, lines, y_bottom, y_top, color=[255, 0, 0], thickness=2, m_left=-1., m_right=1., m_threshold=0.6, 
               draw_all_lines = False, draw_extrapolation = True):
               
    left_lines = []
    right_lines = []
    
    if lines is None:
        return
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1))
            line_color = color
            if np.isclose(slope, m_left, atol=m_threshold): # Is a left line (visual slope in image is positive)
                left_lines.append([x1,y1,x2,y2])
                line_color = [0, 0, 255]
            elif np.isclose(slope, m_right, atol=m_threshold): # Is a right line (visual slope in image is negative)
                right_lines.append([x1,y1,x2,y2])
                line_color = [0, 255, 0]
            
            if draw_all_lines:
                cv2.line(img, (x1, y1), (x2, y2), line_color, thickness)
        
    if len(right_lines)>0 and draw_extrapolation:
        x_bottom, x_top = extrapolate_lines(right_lines, y_bottom, y_top)
        cv2.line(img, (x_bottom, y_bottom), (x_top, y_top), color, thickness*4)
                 
    if len(left_lines)>0 and draw_extrapolation:
        x_bottom, x_top = extrapolate_lines(left_lines, y_bottom, y_top)
        cv2.line(img, (x_bottom, y_bottom), (x_top, y_top), color, thickness*4)
            

def extrapolate_lines(lines_array, y_bottom, y_top):
    lines = np.array(lines_array)
    slopes = (lines[:,3]-lines[:,1])/(lines[:,2]-lines[:,0])
    slopes_mean = np.mean(slopes)
    slope_distance = np.abs(slopes-slopes_mean)
    lines_near_mean = lines[slope_distance<=np.percentile(slope_distance, 40)]
    points = lines_near_mean.reshape((lines_near_mean.shape[0]*2, 2))

    b = np.mean(points[:,1]-slopes_mean*points[:,0])
    x_bottom = int((y_bottom-b)/slopes_mean)
    x_top = int((y_top-b)/slopes_mean)
    return x_bottom, x_top
```

As we can see, apart from the redesign of the `draw_lines` function, we have included an auxiliary function, `extrapolate_lines`. This function will do, step by step, all the process mentioned in the previous *Extrapolate the lines.* section. It will take as an input the array of lines, and the upper and lower bounds of the region of interest, and will return the horizontal values (X) of the line to be drawn where the Y value is equal to the lower and the upper bounds of the roi (x_bottom and x_top, respectively). Traduced to code, these steps would look like this:
* Calculate mean of slopes:
```python
lines = np.array(lines_array)
slopes = (lines[:,3]-lines[:,1])/(lines[:,2]-lines[:,0])
slopes_mean = np.mean(slopes)
```
* Use only those lines with slope near to the mean (in a certain percentile of distance to mean slope, 40 in the example).
```python
slope_distance = np.abs(slopes-slopes_mean)
lines_near_mean = lines[slope_distance<=np.percentile(slope_distance, 40)]
```
* Take both initial and final points of each line and create a `points` array:
```python
b = np.mean(points[:,1]-slopes_mean*points[:,0])
```
* Finally, from the m*X+b equation, we get the points x_bottom and x_top from the points y_bottom and y_top. All of them combined, will give us the points (x0, y0, x1, y1) that will define the line we are looking for.
```python
b = np.mean(points[:,1]-slopes_mean*points[:,0])
x_bottom = int((y_bottom-b)/slopes_mean)
x_top = int((y_top-b)/slopes_mean)
``` 

For the `draw_lines` function itself, the first thing we notice is the introduction of several new input parameters:
 * `y_top`: upper bound of the region of interest. This will be the higher Y coordinate that the extrapolation lines will reach.
 * `y_top`: lower bound of the region of interest. This will be the lower Y coordinate that the extrapolation lines will reach.
 * `m_left`: approximate slope that we expect to find for the lines in the left lane.
 * `m_right`: approximate slope that we expect to find for the lines in the right lane.
 * `m_threshold`: tolerance in the slope at the moment to accept line candidates for the left and right lanes.
 * `draw_all_lines`: (boolean) Whether or not we draw all Hough lines we find in the image. Useful for debugging.
 * `draw_extrapolation`: (boolean) Whether or not we draw the extrapolated lines in the image.

The first thing we do in this function is to classify the lines between right lines and left lines (or neither of them), depending if the slope is close to one or the other expected slope values, with a certain tolerance. We store each line in its correspondent list. Then, for each list of lines, we try to find a line extrapolation, by calling our auxiliar function. We draw these lines in the image.

#### Results obtained in testing images
As said, we found that the resulting output of applying the pipeline in the test images and videos are really what we would expected. Here are the results for the images:

![img_extrapolation_01]

![img_extrapolation_02]

![img_extrapolation_03]

![img_extrapolation_04]

![img_extrapolation_05]

![img_extrapolation_06]

#### 

#### Results obtained in testing videos

// TODO


#### Results obtained in the *Optional Chanllenge*.

Finally, we have been able to apply this function (and pipeline) to the *Optional Challenge*.

In the next snippet code we build the function that will be applied to each frame of the video. We show here the value we have use for each of the pipeline parameters in this case:

```
def process_image_challenge(image):
    imgshape = image.shape
    params = {
        "canny_low_threshold": 50,
        "canny_high_threshold": 150,
        "gaussian_kernel_size": 5,
        "roi_vertices": np.array([[(int(imgshape[1]*0.25), int(imgshape[0]*0.9)), 
                                   (int(imgshape[1]*0.4), int(imgshape[0]*0.65)), 
                                   (int(imgshape[1]*0.6), int(imgshape[0]*0.65)), 
                                   (int(imgshape[1]*0.9), int(imgshape[0]*0.9))]], dtype=np.int32), # vertices of mask
        "roi_top": int(imgshape[0]*0.65),
        "hough_rho": 2,
        "hough_theta": np.pi/180, 
        "hough_threshold": 30, # min intersections in Hough grid cell
        "hough_min_line_len": 30, # min pixels of a line
        "hough_max_line_gap": 20, # max pixels for gap within line
        "draw_all_lines": False,
        "draw_extrapolation": True
    }

    result = lines_pipeline(image, params)
    
    return result
```

The results obtained with these parameters and our previous `draw_lines` function look very promising. We have been able to detect lines perfectly for most time of the video, except for one moment in the left line just after the car exited the first tree, where the roads changes.

<video width="960" height="540" controls>
  <source src="img_results/challenge.mp4" type="video/mp4">
</video>



### 2. Identify potential shortcomings with your current pipeline



One potential shortcoming that could be found in the future would be what would happen with our implementation when the roads had close curves. Right now we're leaning on the lanes angles in order to detect right and left lines candidates, and remove noise. If we find that this angle range is very wide, we may have problems if we don't increase the tolerance of our line classification algorithm, and, if we do increase it, we will be removing some of the advantage we have obtained from its implementation.

Another future work focus maybe more than shortcoming with these approach, could be the robustness against environments with more noise (cars changing lines, different lightning conditions, etc.), which hasn't been tested too much really. 


### 3. Suggest possible improvements to your pipeline

At first glance, we can distinguish where are the two main processes that we can improve in order to obtain better results in our implementation:

* Firstly, we have found that there is a moment the test in the *Optional Challenge* where our algorithm is incapable of locate the left lane. Debugging it, we have found that, the first time we detect lines with the 'Hough lines' method, we're not detecting nothing there either. So it's clear that, for us, the main improvement to the implementation could come from improving this first line detection.
* When extrapolating the line, we have applied a method that we have found to work great in our use cases. There are a lot of possibilities here to apply, so we could improve our model greatly with other functions for line extrapolation.