import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import pandas as pd
import requests

# Initialize the Dash application
app = dash.Dash(__name__, external_scripts=['http://localhost:8000/copilot/index.js'])

# Styles
COLORS = {
    'primary': '#119DFF',
    'secondary': '#4CAF50',
    'background': '#f8f9fa',
    'text': '#333333',
}

STYLES = {
    'page_container': {
        'display': 'flex',
        'justify-content': 'flex-start',
        'padding': '10px',
    },
    'content_container': {
        'width': '60%',  
        'max-width': '500px',
        'background-color': COLORS['background'],
        'padding': '15px',
        'border-radius': '8px',
        'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
    },
    'header': {
        'text-align': 'center',
        'font-family': 'Arial, sans-serif',
        'color': COLORS['text'],
    },
    'tab': {
        'borderBottom': '1px solid #d6d6d6',
        'padding': '4px',
        'fontWeight': 'bold',
        'font-size': '16px',
    },
    'tab_selected': {
        'borderTop': '1px solid #d6d6d6',
        'borderBottom': '1px solid #d6d6d6',
        'backgroundColor': COLORS['primary'],
        'color': 'white',
        'padding': '6px',
    },
    'input_container': {
        'display': 'flex',
        'flex-direction': 'column',
        'gap': '20px',
        'margin-top': '20px',
    },
    'input': {
        'width': '100%',
        'padding': '10px',
        'font-size': '16px',
        'border': f'1px solid {COLORS["primary"]}',
        'border-radius': '4px',
    },
    'textarea': {
        'width': '100%',
        'height': '200px',
        'padding': '10px',
        'font-size': '16px',
        'border': f'1px solid {COLORS["primary"]}',
        'border-radius': '4px',
        'resize': 'vertical',
    },
    'button': {
        'background-color': COLORS['secondary'],
        'color': 'white',
        'padding': '10px 20px',
        'font-size': '16px',
        'border': 'none',
        'border-radius': '4px',
        'cursor': 'pointer',
    },
    'response': {
        'margin-top': '10px',
        'font-size': '16px',
        'color': COLORS['text'],
    }
}

# App layout
app.layout = html.Div(style=STYLES['page_container'], children=[
    html.Div(style=STYLES['content_container'], children=[
        html.H1('Code Generator', style=STYLES['header']),
        dcc.Tabs([
            dcc.Tab(
                label="Input Form",
                style=STYLES['tab'],
                selected_style=STYLES['tab_selected'],
                children=[
                    html.H2('Info for the form', style=STYLES['header']),
                    html.Div(style=STYLES['input_container'], children=[
                        html.Label('App Name', style=STYLES['header']),
                        dcc.Input(id='fieldA', type='text', style=STYLES['input']),
                        
                        html.Label('App Description', style=STYLES['header']),
                        dcc.Input(id='fieldB', type='text', style=STYLES['input']),
                        
                        html.Label('Code Snippet', style=STYLES['header']),
                        dcc.Textarea(id='fieldC', style=STYLES['textarea']),
                        
                        html.Div(style={'text-align': 'center'}, children=[
                            html.Button('Submit', id='add-val', style=STYLES['button']),
                            html.Div(id='submit-response', children='Click to submit', style=STYLES['response'])
                        ])
                    ])
                ]
            )
        ])
    ])
])

# Callback for submit button
@app.callback(
    Output('submit-response', 'children'),
    Input('add-val', 'n_clicks'),
    State('fieldA', 'value'),
    State('fieldB', 'value'),
    State('fieldC', 'value'),
    prevent_initial_call=True
)
def submit_form(n_clicks, app_name, app_description, code_snippet):
    if n_clicks:
        # Here you can add your form submission logic
        return f"Submitted: App Name: {app_name}, Description: {app_description}, Code length: {len(code_snippet) if code_snippet else 0}"
    return "Click to submit"

# Run the Dash server
if __name__ == '__main__':
    app.run_server(debug=True)