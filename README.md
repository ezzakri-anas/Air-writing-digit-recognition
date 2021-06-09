# Air-writing-digit-recognition
This project is a combination of computer vision and Hand written digits recognition. It recognizes hand gestures by detecting and tracking hand landmarks using [Mediapipe](https://google.github.io/mediapipe/) hand tracking solutions and [OpenCV](https://opencv.org/) computer vision library. This data is passed into a Hand written digits recognition model to predict produced patterns.


## Dependencies installation
Using `pip` install following packages:
  - matplotlib==3.2.1
  - tensorflow==2.4.0
  - opencv-python==4.5.2.54
  - mediapipe==0.8.5
  - numpy==1.19.5

**Or** use the following command instead:
  ```shell
  pip install -r requirements.txt
  ```
  
## Datasets
Made use of the followings datasets:
- [MNIST Dataset](https://zenodo.org/record/1188976): contains a training set of 60,000 examples, and a test set of 10,000 examples of hand writting digits images of 28x28 pixels.


## How to Use:
- `Index finger` **up** and other fingers down: Starts drawing on video frames to produce a digit pattern by tracking the index finger and collecting its path on a canva.
- `Pinky finger` **up** and other fingers down: Reinitializes the canva and erases the drawn pattern.
- `Index and middle finger` **up** and other fingers down: Resizes the canva and passes it into the hand written digits recognition model.
