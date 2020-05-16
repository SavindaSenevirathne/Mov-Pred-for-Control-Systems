# Movement-Prediction-for-Control-Systems
This system will predict human movement and will give appropriate control signals

## Versions
- Python - 3.5.1
- OpenCV - 4.1.0
- firebase-admin - for firebase database integration

## Models
Model can be downloaded from the following link. Model is in the project file in the google drive.

 <https://drive.google.com/file/d/1cnTp0pKdgM1ivPffllbc8YAOdM3iJVxM/view?usp=sharing>


Add the downloaded model file inside pose/coco folder

## Installation
Clone this repository and cd into the project folder and checkout to the ***semester8_co425*** branch using

``` git checkout semester8_co425 ```

Install the necessary dependencies using the given requirements.txt file using the following command

``` pip install -r requirements.txt ```

Download the model file from the given google drive link and place it in pose/coco folder.

### Running full application
The entry point for the application is Detection_and_Tracking.py. Run the file using,

``` python  Detection_and_Tracking.py ``` command
- First the facial recoginition window popups and shows the recognition data. press ``` x ``` when satisfied with the recoginition. 
- if the application detects the persons is not in the database, it will prompt to take five images of the person. Press ``` space ``` to save the images and take the images at different angles.
- Press ``` esc ``` when you take necessary amount of images (more than 5)
- Then the human detection will start.
- Once a human is detected, it will show in a different window.
- Close the window to continue tracking the detected person
- Press ``` c ``` to start the posture and distance detection algorithm on current frame. This will output the results in a seperate window. Close or press ``` x ``` to continue tracking

### Data logging
DataLogger.py is used to log the data to output.txt file as well as in firebase database.
Run the file using 

``` python DataLogger.py ``` command

This is implemented in a seperately because of the performance issue in the posture detection which is not problematic in realtime monitoring

## Creators

**Kushan Senanayaka**

* <https://github.com/ksenanayaka>

**Savinda Senevirathne**

* <https://github.com/SavindaSenevirathne>

**Gayan Ranaweera**

* <https://github.com/GayanRanaweera>