#Need to install plotly package using the following command typed within a cell of a Jupyter Notebook
# !pip install plotly
#Tested on Jupyter Notebook with Python - 3 on a Macbook
# Using a special Library to plot Assymetric Error Plots. Matplotlib doesn't have a standard implementation
# Using junk data for demonstration purpose only

import plotly.graph_objects as go

fig = go.Figure(data=go.Scatter(
        x=[1, 2, 3, 4], #Sample Size in Log Scale
        y=[2, 1, 3, 4], # Mean Error
        error_y=dict(
            type='data',
            symmetric=False,
            array=[0.1, 0.2, 0.1, 0.1], # (Max - Mean) Error
            arrayminus=[0.2, 0.4, 1, 0.2]) # (Mean - Min) Error
        ))

fig.show()
