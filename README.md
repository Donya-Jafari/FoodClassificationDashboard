# Food22 Classification Dashboard

Welcome to the Food22 Classification Dashboard repository! This project features a web-based dashboard for image classification, showcasing a deep learning model (Resnet50) trained to classify food images. The dashboard allows users to upload their images for classification, view class distributions, and examine the model's performance metrics.

## Features

- **Image Classification**: Upload an image and get predictions from the trained model.
- **Dataset Visualization**: View class distributions and dataset size.
- **Model Performance**: Examine the confusion matrix and other performance metrics.
- **Interactibg with dataset**: Choose an image from the dataset and see whats the dataset label and what is the model prediction.

## Live Demo

Check out the live demo of the Food22 Classification Dashboard: [classifier](https://food22classifier.dpzone.top)

## Running Locally

### Requirements

To run this project locally, you'll need the following Python libraries:

- `tensorflow`
- `numpy`
- `matplotlib`
- `flask`
- `gunicorn`
- `huggingface_hub`
- `datasets`
- `Pillow`

You can run this on your local by following these steps:

```bash
git clone https://github.com/Donya-Jafari/FoodClassificationDashboard.git
pip install -r requirements.txt
gunicorn app:app
