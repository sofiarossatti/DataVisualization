import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import geopandas as gpd
from wordcloud import WordCloud
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.figure_factory as ff
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from IPython.display import Markdown, display, IFrame
import base64

paygap = pd.read_csv("pay_gap_Europe.csv")
paygap.info()
columns_to_drop = ['GDP', 'Urban_population'] # I'm dropping these columns because I'm not interested in them for the project
paygap = paygap.drop(columns_to_drop, axis=1)

# I create smaller dataframes for each countries to work better with the Nan values
df_austria = paygap[(paygap['Country'] == 'Austria')]
df_Belgium = paygap[(paygap['Country'] == 'Belgium')]
df_Bulgaria =paygap[(paygap['Country'] == 'Bulgaria')]
df_Croatia = paygap[(paygap['Country'] == 'Croatia')]
df_Cyprus = paygap[(paygap['Country'] == 'Cyprus')]
df_CZ = paygap[(paygap['Country'] == 'Czech Republic')]
df_Denmark = paygap[(paygap['Country'] == 'Denmark')]
df_Estonia = paygap[(paygap['Country'] == 'Estonia')]
df_Finland = paygap[(paygap['Country'] == 'Finland')]
df_France = paygap[(paygap['Country'] == 'France')]
df_Germany = paygap[(paygap['Country'] == 'Germany')]
df_Hungary = paygap[(paygap['Country'] == 'Hungary')]
df_Italy = paygap[(paygap['Country'] == 'Italy')]
df_Latvia = paygap[(paygap['Country'] == 'Latvia')]
df_Lithuania = paygap[(paygap['Country'] == 'Lithuania')]
df_Luxembourg = paygap[(paygap['Country'] == 'Luxembourg')]
df_Malta = paygap[(paygap['Country'] == 'Malta')]
df_Netherlands = paygap[(paygap['Country'] == 'Netherlands')]
df_Norway = paygap[(paygap['Country'] == 'Norway')]
df_Poland = paygap[(paygap['Country'] == 'Poland')]
df_Portugal = paygap[(paygap['Country'] == 'Portugal')]
df_Romania = paygap[(paygap['Country'] == 'Romania')]
df_Slovakia = paygap[(paygap['Country'] == 'Slovakia')]
df_Slovenia = paygap[(paygap['Country'] == 'Slovenia')]
df_Spain = paygap[(paygap['Country'] == 'Spain')]
df_Sweden = paygap[(paygap['Country'] == 'Sweden')]
df_Switzerland = paygap[(paygap['Country'] == 'Switzerland')]

df_austria.fillna(df_austria.mean(numeric_only=True), inplace=True)
df_Belgium.fillna(df_Belgium.mean(numeric_only=True), inplace=True)
df_Bulgaria.fillna(df_Bulgaria.mean(numeric_only=True), inplace=True)
df_Croatia.fillna(df_Croatia.mean(numeric_only=True), inplace=True)
df_Cyprus.fillna(df_Cyprus.mean(numeric_only=True), inplace=True)
df_CZ.fillna(df_CZ.mean(numeric_only=True), inplace=True)
df_Denmark.fillna(df_Denmark.mean(numeric_only=True), inplace=True)
df_Estonia.fillna(df_Estonia.mean(numeric_only=True), inplace=True)
df_Finland.fillna(df_Finland.mean(numeric_only=True), inplace=True)
df_France.fillna(df_France.mean(numeric_only=True), inplace=True)
df_Germany.fillna(df_Germany.mean(numeric_only=True), inplace=True)
df_Hungary.fillna(df_Hungary.mean(numeric_only=True), inplace=True)
df_Italy.fillna(df_Italy.mean(numeric_only=True), inplace=True)
df_Latvia.fillna(df_Latvia.mean(numeric_only=True), inplace=True)
df_Lithuania.fillna(df_Lithuania.mean(numeric_only=True), inplace=True)
df_Luxembourg.fillna(df_Luxembourg.mean(numeric_only=True), inplace=True)
df_Malta.fillna(df_Malta.mean(numeric_only=True), inplace=True)
df_Netherlands.fillna(df_Netherlands.mean(numeric_only=True), inplace=True)
df_Norway.fillna(df_Norway.mean(numeric_only=True), inplace=True)
df_Poland.fillna(df_Poland.mean(numeric_only=True), inplace=True)
df_Portugal.fillna(df_Portugal.mean(numeric_only=True), inplace=True)
df_Romania.fillna(df_Romania.mean(numeric_only=True), inplace=True)
df_Slovakia.fillna(df_Slovakia.mean(numeric_only=True), inplace=True)
df_Slovenia.fillna(df_Slovenia.mean(numeric_only=True), inplace=True)
df_Spain.fillna(df_Spain.mean(numeric_only=True), inplace=True)
df_Sweden.fillna(df_Sweden.mean(numeric_only=True), inplace=True)
df_Switzerland.fillna(df_Switzerland.mean(numeric_only=True), inplace=True)

mean_public_administration = paygap['Public_administration'].mean()

df_austria['Public_administration'].fillna(mean_public_administration, inplace=True)
df_Belgium['Public_administration'].fillna(mean_public_administration, inplace=True)
df_Portugal['Public_administration'].fillna(mean_public_administration, inplace=True)

paygap = pd.concat([df_austria, df_Belgium, df_Bulgaria, df_Croatia, df_Cyprus, df_CZ, df_Denmark, df_Estonia, df_Finland, df_France, df_Germany, df_Hungary, df_Italy, df_Latvia, df_Lithuania, df_Luxembourg, df_Malta, df_Netherlands, df_Norway, df_Poland, df_Portugal, df_Romania, df_Slovakia, df_Slovenia, df_Spain, df_Sweden, df_Switzerland], ignore_index=True)
numeric_columns = paygap.columns[2:]

working_fields = numeric_columns  
years = paygap["Year"]

paygap['Max_Value'] = paygap[working_fields].max(axis=1) # I create a new column in the DataFrame to store the maximum value across working fields
paygap['Max_Variable'] = paygap[working_fields].idxmax(axis=1) # here I get the column name with the maximum value

max_values = paygap.groupby(['Country', 'Year'])[['Max_Value', 'Max_Variable']].max().reset_index()


paygap = paygap.drop(["Max_Value"], axis = 1)

paygap.head()

#st.subheader("WordCloud: the biggest gaps in fields")
Industry = paygap["Industry"].mean()
Business= paygap["Business"].mean()
Mining= paygap["Mining"].mean()
Manufacturing= paygap["Manufacturing"].mean()
Electricity_supply= paygap["Electricity_supply"].mean()
Water_supply= paygap["Water_supply"].mean()
Construction = paygap["Construction"].mean()
Retail_trade= paygap["Retail trade"].mean()
Transportation=paygap["Transportation"].mean()
Accommodation= paygap["Accommodation"].mean()
Information= paygap["Information"].mean()
Financial= paygap["Financial"].mean()
Real_Estate= paygap["Real estate "].mean()
Professional_scientific = paygap["Professional_scientific"].mean()
Administrative = paygap["Administrative"].mean()
Public_administration = paygap["Public_administration"].mean()
Education = paygap["Education"].mean()
Human_health = paygap["Human_health"].mean()
Arts = paygap["Arts"].mean()
Other = paygap["Other"].mean()


app = dash.Dash(__name__, suppress_callback_exceptions=True)

#Layout for explanation text
layout_explanation = html.Div([
    html.H1("Explanation 📜"),
    dcc.Markdown("""
        This is a presentation about the gender paygap in Europe between 2010 and 2021. 
        The aim of this project is to offer interesting insights about the topic that could lead to further considerations.
        In this presentation I will show you:

        *Choropleth Map Tab:* Explore the gender pay gap across different countries using an interactive choropleth map.

        *WordCloud Tab:* Visualize the working fields with the highest pay gap using a WordCloud representation.

        *Grouped Bar Chart Tab:* View grouped bar charts to compare pay gaps across variables for specific countries and years.

        *Interactive Map Tab:* Explore interactive maps showing the intensity of variables across countries and years.

        *Line Plot Tab:* Visualize the change over time for specific variables in different countries.

        *Correlation Heatmap Tab:* View the correlation between variables using an interactive heatmap.

        *Clusterization Tab:* Explore how countries cluster based on the gender pay gap.
                 
        
    """)
])

# Layout for Choropleth Map
layout_choropleth = html.Div([
    html.H1("Choropleth Map 🌎"),
    dcc.Dropdown(
        id='year-dropdown-choropleth',
        options=[{'label': str(year), 'value': year} for year in paygap['Year'].unique()],
        value=paygap['Year'].min()
    ),
    dcc.Graph(id='choropleth-map')
])

@app.callback(
    Output('choropleth-map', 'figure'),
    Input('year-dropdown-choropleth', 'value')
)
def update_choropleth(selected_year):
    filtered_data = paygap[paygap['Year'] == selected_year]
    
    fig = px.choropleth(max_values, 
                    locations="Country", 
                    locationmode='country names',
                    color="Max_Variable",
                    hover_data=["Max_Variable", "Max_Value"],  # Add variables to hover data
                    color_continuous_scale=px.colors.qualitative.Plotly
                    )
    
    return fig

# Layout for Wordcloud Map

working_sectors = {"Industry": Industry, "Business": Business, "Mining": Mining, "Manufacturing": Manufacturing, "Electricity supply": Electricity_supply, "Water supply": Water_supply, "Construction": Construction, "Retail trade": Retail_trade, "Transportation": Transportation, "Accommodation": Accommodation, "Information": Information, "Financial": Financial, "Real estate": Real_Estate ,  "Prefessional scientific": Professional_scientific, "Administrative": Administrative, "Public administration": Public_administration,  "Education": Education,  "Human health": Human_health,  "Arts": Arts, "Other": Other }
wordcloud = WordCloud(width=400, height=200, background_color='white').generate_from_frequencies(working_sectors)
# Convert the WordCloud image to a base64-encoded image
img = wordcloud.to_image()
img_byte_array = io.BytesIO()
img.save(img_byte_array, format='PNG')
img_base64 = base64.b64encode(img_byte_array.getvalue()).decode()

layout_wordcloud = html.Div([
    html.H1("WordCloud 💭"),
    html.Img(src='data:image/png;base64,{}'.format(img_base64), style={'width': '60%', 'height': 'auto', 'display': "block", 'margin-left': 'auto', 'margin-right': 'auto'})
])

#END WORDCLOUD

layout_page1 = html.Div([
    html.H1("Grouped Bar Chart 📊"),
    dcc.Dropdown(
        id='country-dropdown-page1',
        options=[{'label': country, 'value': country} for country in paygap['Country'].unique()],
        value=paygap['Country'].unique()[0]
    ),
    dcc.Graph(id='grouped-bar-chart-page1'),
    dcc.Slider(
        id='year-slider-page1',
        min=min(years),
        max=max(years),
        value=min(years),
        marks={str(year): str(year) for year in years},
        step=None
    )
])

# Callback for updating grouped bar chart
@app.callback(
    Output('grouped-bar-chart-page1', 'figure'),
    Input('country-dropdown-page1', 'value'),
    Input('year-slider-page1', 'value')
)
def update_grouped_bar_chart(selected_country, selected_year):
    filtered_data = paygap[(paygap['Country'] == selected_country) & (paygap['Year'] == selected_year)]
    trace_list = []
    for variable in working_fields:
        trace = go.Bar(x=[variable], y=[filtered_data[variable].values[0]], name=variable)
        trace_list.append(trace)

    return {
        'data': trace_list,
        'layout': go.Layout(title=f'Variable Values for {selected_country} in {selected_year}',
                            xaxis={'title': 'Variable'},
                            yaxis={'title': 'Value'},
                            barmode='group')
    }

layout_page2 = html.Div([
    html.H1("Scatter geo map 🔍"),
    dcc.Dropdown(
        id='year-dropdown-page2',
        options=[{'label': str(year), 'value': year} for year in paygap['Year'].unique()],
        value=paygap['Year'].min()
    ),
    dcc.Dropdown(
        id='variable-dropdown-page2',
        options=[{'label': variable, 'value': variable} for variable in paygap.columns[2:]],
        value=paygap.columns[2]
    ),
    dcc.Graph(id='interactive-map-page2', style={'width': '100%', 'height': '600px'})
])

# Callback for updating interactive map
@app.callback(
    Output('interactive-map-page2', 'figure'),
    Input('year-dropdown-page2', 'value'),
    Input('variable-dropdown-page2', 'value')
)
def update_interactive_map(selected_year, selected_variable):
    filtered_data = paygap[paygap['Year'] == selected_year]
    
    fig = px.scatter_geo(
        filtered_data,
        locations='Country',
        locationmode='country names',
        hover_name='Country',
        size=filtered_data[selected_variable],
        color=filtered_data[selected_variable],
        color_continuous_scale='Viridis',
        title=f'Interactive Map: {selected_variable} in {selected_year}'
    )
    
    fig.update_geos(
        projection_type="natural earth",
        showcoastlines=True, coastlinecolor="Black",
        showland=True, landcolor="white",
        showcountries=True, countrycolor="Black", countrywidth=0.5
    )
    
    return fig

# Layout for Page 3: Line Plot
layout_page3 = html.Div([
    html.H1("Line Plot 📉"),
    dcc.Dropdown(
        id='variable-dropdown-page3',
        options=[{'label': column, 'value': column} for column in paygap.columns[2:]],
        value=paygap.columns[2]
    ),
    dcc.Dropdown(
        id='country-dropdown-page3',
        options=[{'label': country, 'value': country} for country in paygap['Country'].unique()],
        value=paygap['Country'].unique()[0]
    ),
    dcc.Graph(id='line-plot-page3')
])

# Callback for updating line plot
@app.callback(
    Output('line-plot-page3', 'figure'),
    Input('variable-dropdown-page3', 'value'),
    Input('country-dropdown-page3', 'value')
)
def update_line_plot(selected_variable, selected_country):
    filtered_data = paygap[paygap['Country'] == selected_country]
    
    fig = px.line(filtered_data, x='Year', y=selected_variable,
                  labels={'Your_Variable_Column': 'Variable Value'},
                  markers=True)

    fig.update_layout(title=f'Change Over Time for {selected_country}',
                      xaxis_title='Year', yaxis_title='Variable Value')
    
    return fig

# Layout for Page 4: 3D Scatter Plot
layout_page4 = html.Div([
    html.H1("3D Scatter Plot 📌"),
    dcc.Dropdown(
        id='year-dropdown-page4',
        options=[{'label': str(year), 'value': year} for year in paygap['Year'].unique()],
        value=paygap['Year'].min()
    ),
    dcc.Dropdown(
        id='x-variable-dropdown-page4',
        options=[{'label': variable, 'value': variable} for variable in paygap.columns[2:]],
        value=paygap.columns[2]
    ),
    dcc.Dropdown(
        id='y-variable-dropdown-page4',
        options=[{'label': variable, 'value': variable} for variable in paygap.columns[2:]],
        value=paygap.columns[3]
    ),
    dcc.Graph(id='interactive-3d-scatter-page4')
])

# Callback for updating 3D scatter plot
@app.callback(
    Output('interactive-3d-scatter-page4', 'figure'),
    Input('year-dropdown-page4', 'value'),
    Input('x-variable-dropdown-page4', 'value'),
    Input('y-variable-dropdown-page4', 'value')
)
def update_3d_scatter(selected_year, selected_x_variable, selected_y_variable):
    filtered_data = paygap[paygap['Year'] == selected_year]
    
    fig = px.scatter_3d(
        filtered_data,
        x=selected_x_variable,
        y=selected_y_variable,
        z='Country',
        color=selected_x_variable,
        size=selected_y_variable,
        text='Country',
        title=f'Interactive 3D Scatter Plot: {selected_x_variable} vs {selected_y_variable} in {selected_year}'
    )
    
    return fig

# Layout for Page 5: Heatmap
layout_page5 = html.Div([
    html.H1("Heatmap 🟨🟩🟪"),
    dcc.Dropdown(
        id='year-dropdown-page5',
        options=[{'label': str(year), 'value': year} for year in paygap['Year'].unique()],
        value=paygap['Year'].min()
    ),
    dcc.Graph(id='cluster-heatmap-page5')
])

# Callback for updating cluster heatmap
@app.callback(
    Output('cluster-heatmap-page5', 'figure'),
    Input('year-dropdown-page5', 'value')
)
def update_cluster_heatmap(selected_year):
    filtered_data = paygap[paygap['Year'] == selected_year]
    correlation_matrix = filtered_data[numeric_columns].corr()
    
    fig = px.imshow(correlation_matrix,
                    labels=dict(x="Variable", y="Variable"),
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    color_continuous_scale='Viridis')

    fig.update_layout(title=f'Correlation Heatmap for {selected_year}', width=1200, height=600)

    return fig

# Layout for Page 6: Clusterization
layout_page6 = html.Div([
    html.H1("Clusterization 🗺"),
    dcc.Dropdown(
        id='year-dropdown-page6',
        options=[{'label': str(year), 'value': year} for year in paygap['Year'].unique()],
        value=paygap['Year'].min()
    ),
    dcc.Graph(id='cluster-map-page6')
])

# Callback for updating cluster map
@app.callback(
    Output('cluster-map-page6', 'figure'),
    Input('year-dropdown-page6', 'value')
)
def update_cluster_map(selected_year):
    paygap1 = paygap.drop(["Max_Variable"], axis = 1)
    filtered_data = paygap1[paygap1['Year'] == selected_year]
    # Perform clustering on filtered_data
    data = filtered_data.drop(['Country', 'Year'], axis=1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    num_clusters = 6
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(scaled_data)
    filtered_data['Cluster'] = clusters
    
    # Create a choropleth cluster map
    fig = px.choropleth(
        filtered_data,
        locations='Country',
        locationmode='country names',
        color='Cluster',
        color_continuous_scale='Viridis',
        hover_name='Country',
        title=f'Cluster Map for Year {selected_year}',
    )

    tickvals = list(range(num_clusters))
    ticktext = [str(cluster) for cluster in tickvals]
    fig.update_layout(coloraxis_colorbar=dict(tickvals=tickvals, ticktext=ticktext))
    
    return fig

app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab-0', children=[
        dcc.Tab(label='Explanation', value='tab-0'),
        dcc.Tab(label='Choropleth Map', value='tab-1'),
        dcc.Tab(label='WordCloud', value='tab-2'),
        dcc.Tab(label='Grouped bar Chart', value='tab-3'),
        dcc.Tab(label='Variables intensity', value='tab-4'),
        dcc.Tab(label='Line plot', value='tab-5'),
        dcc.Tab(label='Correlation between variables', value='tab-6'),
        dcc.Tab(label='Heatmap', value='tab-7'),
        dcc.Tab(label='Clusterization', value='tab-8'),
    ]),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'), Input('tabs', 'value'))
def display_page(tab):
    if tab == 'tab-0':
        return layout_explanation  
    if tab == 'tab-1':
        return layout_choropleth  
    elif tab == 'tab-2':
        return layout_wordcloud
    if tab == 'tab-3':
        return layout_page1
    elif tab == 'tab-4':
        return layout_page2
    elif tab == 'tab-5':
        return layout_page3
    elif tab == 'tab-6':
        return layout_page4
    elif tab == 'tab-7':
        return layout_page5
    elif tab == 'tab-8':
        return layout_page6

if __name__ == '__main__':
    app.run_server(debug=True, port=8061)