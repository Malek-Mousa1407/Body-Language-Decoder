## 1. First extract joints/coordinates position into the CSV file.
`python Train.py` <br>
On running this command, the program will first prompt you to enter a 'classname' which is essentially the pose name that your trying to train your model to detect. Next, simply pose in front of your camera in your desired position. To exit the program press `q`.

## 2. Train the classification ML model.
After completing the first step, the script will automatically extract the joint coordinates to a CSV file. <br>Running this command: `python ClassificatioModel.py` will fit the CSV data to train the classification ML model to make predictions based on the joints' coordinates positions.

## 3. Make pose detections
Running the `Detect.py` script, will predict your pose and display it on screen, based on pre-existing data. 