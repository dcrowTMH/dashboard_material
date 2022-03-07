# dashboard_material

Here is the repository of the course about building an interactive dashboard using Python.

It will be used to store the code material/ exercise involved in this course.

Course: (Udemy)
https://www.udemy.com/course/build-an-interactive-data-analytics-dashboard-with-python/

It it great to practice the coding skill and data handling skill with the flow of how data science work.

All the materials was used to be a reference for my routine work/ future work as well.

All the python files was used to clean the mindset of dealing with the time series data.

All the function was defined to handle same type/ kinds of data and encapsulated into classes later.

Data is special for the related COVID-19 data set and adjustment is needed for other set of data.

After that, HTML & CSS was used to build the dashboard and deploy for production.

1) prepare.py
- Demostration of how to get ready the code for getting data, data cleaning, transformation & smoothing
- Dealing with bad data for time series (cummax & mask function)

2) smooth_data.py
- Used for smooth the data for time series data type
- Applied lowess function to smooth the data for variation
- Data can be variated due to the weekends/ reporting/ holidays

3) exponential_growth_decline_model.py
- Using simple exponential fucntion to demostrate the growth or decline of the data
- make the prediction , correct the prediction & plot it for visualization

4) logistic_growth_model.py
- Using logistic function/ sigmoid fuction for the prediction
- Build a model with Assymptotes of beginning & end of the prediction
- Parameters for vertical/ horizontal moving & upper asymptote & growth rate
- Generalized logistic function to prevent overestimated/ underestimated
