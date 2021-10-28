### AI Energy Forecast using LSTM 

It basically takes some smartmeter data (5 cols, > 12mil. instances, cols: id, device_name, property, value, timestamp) and creates a custom forecast based on selected window. 
The file is available in .py and .ipynb format, so you can choose according to your preferences.

Please notice that once you load up the smartmeter data, there are inputs created on the timestamp col like wd_input (the weekday of the timestamp), as well as a cos(inus) and sin(us)
time inputs, giving the model the ability to keep track of the daytime of each instance. Finally, the inputs are merged to an input df, standardized and differenced.
After that, some functions are used to give the user the ability to use time windows from the data. Based on these, the model generates forecasts.   
   
![Model](https://github.com/infinimesh/ai/blob/main/energy-forcast/model.png?raw=true)
   
The first models created are a simple baseline model, used for evaluating the performance of the later on built LSTM model. The baseline model simply shifts the values by t=1. Hence,
there is no t=0 and each timestamp uses the value from t-1.
Finally, there's the 2-layer plain vanilla LSTM. After 11 epochs, I reached a loss of 10.86 which is rather mediocre. However, the main idea here is to build a basic forecasting model
for which this seems appropriate.   


![LSTM](https://github.com/infinimesh/ai/blob/main/energy-forcast/LTSM.png?raw=true)   
   
Blogpost: https://infinitedevices.de/en/time-series-forecast-with-lstm/
