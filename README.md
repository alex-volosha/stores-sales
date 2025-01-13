# Retail store sales prediction
* The dataset I used is from our Kaggle competition: https://www.kaggle.com/competitions/ml-zoomcamp-2024-competition/data  

* This dataset contains sales information from four stores of one of the retailers over 25 months. Participants are expected to use these files to develop models that can predict customer demand.   

## Goal   
* Predict items sales on a specific dates to help retail stores optimize stock management and reduce operational inefficiencies.  

## Data preparation and EDA:  
* All EDA, ML models trainings, and Visualisations you can find in `notebook.ipynb`

## Run the project:  
* `git clone https://github.com/alex-volosha/stores-sales/`  

## 1. Run the app as a web service locally:  
After cloning the repo you can install virtual environment dedicated for this project with all dependancies.  
(Make sure to go to the project directory in you terminal before you run this):  
`pip install pipenv`  

Then install Pipfile/Pipfile.lock files by:  
`pipenv install`  

Narrow down into newly created virtual environment:  
`pipenv shell`  

And now you can run `python predict.py` script.
Open a new terminal and send request to running predict.py request by calling `python item.py`  

## 2. Run the app locally within a Docker container  
> :warning: **Warning:** First make sure Docker is installed and running so you can connect it. 
<a href="https://docs.docker.com/engine/install/" target="_blank">Check Docker website to Install Docker Engine</a>  

- Build docker image  
(No need to run pipenv, Dockerfile will do it itself) by running this command in your terminal  
`docker build -t  sales .`  

- Run the docker image  
`docker run -it --rm -e PORT=9696 -p 9696:9696 sales`  

- Open a new terminal window and run the prediction request  
`python item.py`  
And you will get the prediction of quantity to be sold (notice it might not be the whole numbers, but selling in bulks of unpacked goods)  


> :bulb: **Options:** As shown in the features importance chart, some of the main important features are store_format and item_class for example.  
So, by changing the variable in the `item.py` file from:  
'store_format' : 'Format-1'  
to:  
'store_format' : 'Format-2'  
The predicted amount of items will change.  




