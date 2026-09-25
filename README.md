# Week_6-Task-6
Train YOLOv8 to detect and label defect type + location on metal part images

For training large datasets, a good GPU is the best option. Hence, we are using Google Colab, which provides a T4 GPU for training. This makes the training process fast and easy. I am providing the Colab link and the code below. Just upload the annotated dataset to this Colab link and run the cells in the order they appear.

Colab Link: 
https://colab.research.google.com/drive/1CvpUFf0GnGukSUu9XNY3KHwV7jj4u9R6?usp=sharing

Drive Download link for the Dataset:
https://drive.google.com/file/d/1dqs952oSqCP4zSdBbW0LxEU_TnWWlPDO/view?usp=sharing

Upload the metal_defect dataset to Google Colab, train, and download the Model:
  1. Open Google Colab.
  2. Go to the runtime environment and change the runtime to T4 GPU.
  3. On the left side bar, there is a folder symbol. It has an option to upload the data.zip file.
  4. Run the respective Cobab cells to unzip and train the model.
  5. At the end, download the my_model file.

After downloading the my_model file. Open Google Chrome and download Anaconda. Download the one suitable for your Operating System. (Windows, Linux, or Mac OS).

Steps in Anaconda Prompt Window:
  1. Search for Anaconda Prompt on your PC
  2. In the Anaconda Prompt, create an environment by running the command:
     create --name yolo_env1 python=3.12
  3. Then activate the environment and follow the commands:
     conda activate yolo_env1
  4. Install the library pip install ultralytics
