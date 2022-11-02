#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.graph_objects as px
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.write("hello")
# In[2]:


circuits= pd.read_csv('circuits.csv')
constructor_results= pd.read_csv('constructor_results.csv')
constructor_standings= pd.read_csv('constructor_standings.csv')
constructors= pd.read_csv('constructors.csv')
driver_standings= pd.read_csv('driver_standings.csv')
drivers= pd.read_csv('drivers.csv')
lap_times= pd.read_csv('lap_times.csv')
pit_stops= pd.read_csv('pit_stops.csv')
qualifying= pd.read_csv('qualifying.csv')
races= pd.read_csv('races.csv')
results= pd.read_csv('results.csv')
seasons= pd.read_csv('seasons.csv')
sprint_results= pd.read_csv('sprint_results.csv')
status= pd.read_csv('status.csv')


# In[3]:


racesdf = races.copy()
racesdf = racesdf.drop(columns = ['url',
       'fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time',
       'quali_date', 'quali_time', 'sprint_date', 'sprint_time', 'time'])
racesdf = racesdf.rename(columns ={'name':'race_name'})


# In[4]:


circuitsdf = circuits.copy()
circuitsdf = circuitsdf.drop(columns = ['alt', 'url'])
circuitsdf = circuitsdf.rename(columns={'name':'circuit_name', 'location':'city'})


# In[5]:


results_copy_df = results.copy()


# In[6]:


driversdf = drivers.copy()
driversdf =driversdf.drop(columns =['driverRef', 'number', 'code', 'url'])
driversdf['driver_name'] = driversdf['forename'] + ' ' + driversdf['surname']
driversdf = driversdf.drop(columns =['forename', 'surname'])


# In[7]:


constructorsdf =constructors.copy()
constructorsdf =constructorsdf.drop(columns = ['url','constructorRef'])
constructorsdf =constructorsdf.rename(columns = {'name':'constructors_name'})


# In[8]:


fastestlapdf = racesdf.merge(circuitsdf , on = 'circuitId')
fastestlapdf = fastestlapdf.merge(results_copy_df, on = 'raceId')


# In[9]:


merged_df = results.merge(status , on = 'statusId')
merged_df = merged_df.merge(racesdf, on = 'raceId')
merged_df = merged_df.merge(driversdf, on = 'driverId')
merged_df = merged_df.merge(constructorsdf, on = 'constructorId')
merged_df = merged_df.merge(circuitsdf , on = 'circuitId')

merged_df = merged_df.rename(columns= {'nationality_x':'driver_nationality','nationality_y':'constructor_nationality'})

driver_analysis_df = merged_df.groupby(['year','driver_name']).agg({'points': ['sum'],'raceId':['count'], 'positionOrder':['mean','std'] }).reset_index()

driver_analysis_df.columns = ['_'.join(col).strip() for col in driver_analysis_df.columns.values]
driver_analysis_df = driver_analysis_df.rename( columns = {'year_':'year', 'driver_name_' : 'driver_name'})

champion_df= driver_analysis_df.groupby(['year', 'driver_name']).agg({'points_sum':sum}).reset_index()

champion_df = champion_df.sort_values(['year','points_sum'], ascending = False).groupby('year').head(1)

champion_df = champion_df.drop(3155)

most_races = merged_df.groupby('driver_name')[['raceId']].count().reset_index()
most_races = most_races.sort_values('raceId', ascending= False).head(10)
most_races = most_races.rename(columns ={'raceId': 'total_races'})

con_analysis_df = merged_df.groupby(['year','constructors_name']).agg({'points': ['sum'],'raceId':['count'],'positionOrder':['mean','std'] }).reset_index()
con_analysis_df.columns = ['_'.join(col).strip() for col in con_analysis_df.columns.values]


# In[10]:


condf= con_analysis_df[['year_','constructors_name_', 'points_sum']].tail(10)
condf= condf.sort_values(by='points_sum', ascending=False)


# # Circuit plots

# In[11]:


most_circuits = fastestlapdf.groupby(['year','circuit_name']).count().reset_index()

m = most_circuits['circuit_name'].value_counts().reset_index().head(20)
m =  m.rename(columns ={'circuit_name': 'count', 'index': 'circuit_name'})


# In[12]:


import plotly.express as px
fig = px.bar(m, x='count', y='circuit_name')

fig.update_layout(title='Top 20 meest voorkomende circuits in Formule 1',
                   xaxis_title='Circuits',
                   yaxis_title='Rondetijden',
                   template = "plotly_dark")

fig.show()


# In[13]:


fastestlapdf['f_lap_1'] = fastestlapdf['fastestLapTime'].apply(lambda x : (x.split('.')[-1]))
fastestlapdf['f_lap_2'] = fastestlapdf['fastestLapTime'].apply(lambda x : (x.split('.')[0]))
fastestlapdf['f_lap_3'] = fastestlapdf['f_lap_2'].apply(lambda x: (x.split(':')[-1]))
fastestlapdf['f_lap_4'] = fastestlapdf['f_lap_2'].apply(lambda x: (x.split(':')[0]))

fastestlapdf['f_lap_1'] = fastestlapdf['f_lap_1'].str.strip()
fastestlapdf['f_lap_3'] = fastestlapdf['f_lap_3'].str.strip()
fastestlapdf['f_lap_4'] = fastestlapdf['f_lap_4'].str.strip()

fastestlapdf['f_lap_1'] = pd.to_numeric(fastestlapdf['f_lap_1'] , errors = 'coerce')
fastestlapdf['f_lap_3'] = pd.to_numeric(fastestlapdf['f_lap_3'] , errors = 'coerce')
fastestlapdf['f_lap_4'] = pd.to_numeric(fastestlapdf['f_lap_4'] , errors = 'coerce')

fastestlapdf['fastest_lap'] = fastestlapdf['f_lap_1'] + fastestlapdf['f_lap_3']*1000 + fastestlapdf['f_lap_4']*60*1000

fastestlapdf = fastestlapdf.drop(columns = ['f_lap_4','f_lap_3', 'f_lap_2','f_lap_1'])

x = fastestlapdf.sort_values('year')

fastestlapdf = fastestlapdf[(fastestlapdf['year'].between(2004,2021, inclusive = 'both'))]

h = fastestlapdf.groupby(['year','circuit_name']).count().reset_index()

lap_time_monza = fastestlapdf[fastestlapdf['circuit_name'] == 'Autodromo Nazionale di Monza']
lap_time_monaco = fastestlapdf[fastestlapdf['circuit_name']== 'Circuit de Monaco']
lap_time_silverstone = fastestlapdf[fastestlapdf['circuit_name']== 'Silverstone Circuit']
lap_time_catalunya = fastestlapdf[fastestlapdf['circuit_name']=='Circuit de Barcelona-Catalunya']
lap_time_hungaroring = fastestlapdf[fastestlapdf['circuit_name']== 'Hungaroring']
lap_time_spa = fastestlapdf[fastestlapdf['circuit_name']== 'Circuit de Spa-Francorchamps']

lap = lap_time_silverstone.groupby('year')[['fastest_lap']].min().reset_index()
lap1 =  lap_time_monaco.groupby('year')[['fastest_lap']].min().reset_index()
lap2 =  lap_time_monza.groupby('year')[['fastest_lap']].min().reset_index()
lap3 = lap_time_catalunya.groupby('year')[['fastest_lap']].min().reset_index()
lap4 = lap_time_hungaroring.groupby('year')[['fastest_lap']].min().reset_index()
lap5 = lap_time_spa.groupby('year')[['fastest_lap']].min().reset_index()


# In[14]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=lap.year, y=lap.fastest_lap,
                    mode='lines',
                    name='Silverstone'))
fig.add_trace(go.Scatter(x=lap1.year, y=lap1.fastest_lap,
                    mode='lines',
                    name='Monaco'))
fig.add_trace(go.Scatter(x=lap2.year, y=lap2.fastest_lap,
                    mode='lines',
                    name='Monza'))
fig.add_trace(go.Scatter(x=lap3.year, y=lap3.fastest_lap,
                    mode='lines',
                    name='Catalunya'))
fig.add_trace(go.Scatter(x=lap4.year, y=lap4.fastest_lap,
                    mode='lines',
                    name='Hungaroring'))
fig.add_trace(go.Scatter(x=lap5.year, y=lap5.fastest_lap,
                    mode='lines',
                    name='Spa-Francorchamps'))

fig.update_layout(title='Snelste rondetijden per Circuit per paar',
                   xaxis_title='Jaar',
                   yaxis_title='Rondetijden',
                   template = "plotly_dark")


# In[15]:


pitstopsdf = pit_stops.merge(races , on = 'raceId')
pitstopsdf = pitstopsdf.merge(circuits, on ="circuitId") 

lap_time_monza = pitstopsdf[pitstopsdf['name_y'] == 'Autodromo Nazionale di Monza']
lap_time_monaco = pitstopsdf[pitstopsdf['name_y']== 'Circuit de Monaco']
lap_time_silverstone = pitstopsdf[pitstopsdf['name_y']== 'Silverstone Circuit']
lap_time_catalunya = pitstopsdf[pitstopsdf['name_y']=='Circuit de Barcelona-Catalunya']
lap_time_hungaroring = pitstopsdf[pitstopsdf['name_y']== 'Hungaroring']
lap_time_spa = pitstopsdf[pitstopsdf['name_y']== 'Circuit de Spa-Francorchamps']

lap = lap_time_silverstone.groupby('year')[['milliseconds']].min().reset_index()
lap1 =  lap_time_monaco.groupby('year')[['milliseconds']].min().reset_index()
lap2 =  lap_time_monza.groupby('year')[['milliseconds']].min().reset_index()
lap3 = lap_time_catalunya.groupby('year')[['milliseconds']].min().reset_index()
lap4 = lap_time_hungaroring.groupby('year')[['milliseconds']].min().reset_index()
lap5 = lap_time_spa.groupby('year')[['milliseconds']].min().reset_index()


# In[16]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=lap.year, y=lap.milliseconds,
                    mode='lines',
                    name='Silverstone'))
fig.add_trace(go.Scatter(x=lap1.year, y=lap1.milliseconds,
                    mode='lines',
                    name='Monaco'))
fig.add_trace(go.Scatter(x=lap2.year, y=lap2.milliseconds,
                    mode='lines',
                    name='Monza'))
fig.add_trace(go.Scatter(x=lap3.year, y=lap3.milliseconds,
                    mode='lines',
                    name='Catalunya'))
fig.add_trace(go.Scatter(x=lap4.year, y=lap4.milliseconds,
                    mode='lines',
                    name='Hungaroring'))
fig.add_trace(go.Scatter(x=lap5.year, y=lap5.milliseconds,
                    mode='lines',
                    name='Spa-Francorchamps'))

fig.update_layout(title='Snelste pitstoptijden per Circuit per jaar',
                   xaxis_title='Jaren',
                   yaxis_title='Pitstoptijden',
                   template = "plotly_dark")


# In[17]:


circuitsdf['text'] = circuitsdf['circuit_name'] + ', ' + 'Country: ' + circuitsdf['country'].astype(str)

fig = go.Figure(data=go.Scattergeo(
        lon = circuitsdf['lng'],
        lat = circuitsdf['lat'],
        text = circuitsdf['text'],
        mode = 'markers',
        marker_colorscale = "thermal"
        
        ))

fig.update_geos(projection_type="orthographic")

fig.update_layout(
        #title = 'Circuits of Formula 1 across the world<br>(Hover for info)',
        height=500,
        margin={"r":0,"t":0,"l":0,"b":0},
        template = "plotly_dark"
    )


fig.show()


# # Driver plots

# In[18]:


driver_country = driversdf.groupby('nationality').driver_name.nunique().reset_index() 
driver_country = driver_country.rename(columns = {'driver_name': 'driver_counts'})
driver_country1 = driver_country[driver_country.driver_counts >= 30].sort_values('driver_counts' ,ascending = False )
driver_country1.loc[len(driver_country1.index)] = ['Others', (driver_country.driver_counts.sum() - driver_country1.driver_counts.sum())]
driver_country1


# In[19]:


labels = driver_country1['nationality']
values = driver_country1['driver_counts']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, pull=[0, 0, 0, 0, 0, 0, 0.2], hoverinfo='label+value')])

fig.update_layout(title='Verdeling van nationaliteit per seizoen', template = "plotly_dark")

fig.show()


# In[20]:


drivers = driver_analysis_df[['year','driver_name', 'points_sum']]
drivers = drivers.sort_values(by='points_sum', ascending=False)
drivers1 = drivers[drivers['year']==2021]


# In[21]:


fig = px.bar(drivers1, y='points_sum', x='driver_name', text_auto='.2s')

fig.update_layout(title='Aantal punten vd Coureurs per Seizoen',
                   xaxis_title='Team',
                   yaxis_title='Punten',
                   template = "plotly_dark")

fig.show()


# In[22]:


scat_df = merged_df[['raceId', 'driver_name', 'constructors_name', 'grid', 'position', 'circuit_name']]
#scat_df = scat_df[scat_df.constructors_name == 'Williams']
scat_df = scat_df[scat_df.driver_name == 'Lewis Hamilton']
scat_df = scat_df[scat_df.position != r'\N']
scat_df = scat_df.sort_values(by= ['position', 'grid'], ascending = [True, True]).head(500)


# In[23]:


fig = px.scatter(scat_df, x="grid", y="position")

fig.update_layout(title='Correlatie tussen Qualificatie positie en eindpositie',
                   xaxis_title='Qualificatie positie',
                   yaxis_title='Eindpositie',
                   template = "plotly_dark")

fig.update_yaxes(type='linear')

fig.show()


# In[24]:


alle_drivers= scat_df['driver_name'].unique()


# In[25]:


fig = go.Figure()

teller = 0
buttonlist = [dict(label = "Kies een coureur", method='update', args=[{"visible": [True*len(alle_drivers)]}])]

for i in alle_drivers:
    df2= scat_df[scat_df['driver_name'] == i]
    
    fig.add_trace(go.Scatter(x=scat_df["grid"], y=scat_df["position"], mode='markers', name=str(i)))
    
    lijst = [False]*len(alle_drivers)
    lijst[teller] = True
    teller = teller + 1
    
    one_button = dict(label = str(i), method='update', args=[{"visible": lijst}])
    buttonlist.append(one_button)
    
fig.update_layout(
updatemenus=[
        dict(
            buttons=buttonlist,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=1.1,
            xanchor="left",
            y=1.2,
            yanchor="top"
        ),        
    ]
)

fig.update_layout(title='Correlatie tussen Qualificatie positie en eindpositie',
                   xaxis_title='Qualificatie positie',
                   yaxis_title='Eindpositie',
                   template = "plotly_dark")

fig.update_yaxes(type='linear')
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Constructeur plots

# In[26]:


condf= con_analysis_df[['year_','constructors_name_', 'points_sum']].tail(10)
condf= condf.sort_values(by='points_sum', ascending=False)


# In[27]:


fig = px.bar(condf, y='points_sum', x='constructors_name_',  text_auto='.2s')

fig.update_layout(title='Aantal Constructeurs punten per Seizoen',
                   xaxis_title='Team',
                   yaxis_title='Punten',
                   template = "plotly_dark")

fig.show()


# In[28]:


condf1= con_analysis_df[['year_','constructors_name_', 'points_sum']]
condf1 = condf1[condf1['points_sum']>= 70]
condf1= condf1.sort_values(by='points_sum', ascending=False)


# In[29]:


fig = px.bar(condf1, y='points_sum', x='constructors_name_',  text_auto='.2s')

fig.update_layout(title='Aantal Constructeurs punten per Seizoen',
                   xaxis_title='Team',
                   yaxis_title='Punten',
                   template = "plotly_dark")

fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Legends plots

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




