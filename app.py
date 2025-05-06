import dash
from dash import Dash, dcc, html, Input, Output
import numpy as np
import plotly.graph_objs as go

app = Dash(__name__)

def stochastic_sir(S0, I0, R0, beta, gamma, nu, rho, sigma, lam, t):
    dt = t[1] - t[0]
    S, I, R = [S0], [I0], [R0]
    for i in range(1, len(t)):
        dW = np.random.normal(0, np.sqrt(dt))
        dN = np.random.poisson(lam * dt)
        dS = -beta * S[-1] * I[-1] * dt + sigma * dW - nu * S[-1] * dt
        dI = beta * S[-1] * I[-1] * dt - gamma * I[-1] * dt + dN + rho * R[-1] * dt
        dR = gamma * I[-1] * dt - rho * R[-1] * dt + nu * S[-1] * dt
        S.append(S[-1] + dS)
        I.append(I[-1] + dI)
        R.append(R[-1] + dR)
    return np.array(S), np.array(I), np.array(R)

app.layout = html.Div([
    html.H1("Стохастична SIR-модель: Візуалізація", style={'textAlign': 'center'}),
    html.Div([
        html.Div([
            html.Label("S0"),
            dcc.Slider(0.1, 1.0, 0.1, value=0.9, id='S0-slider', tooltip={"placement": "bottom"}, updatemode='drag'),
            html.Label("I0"),
            dcc.Slider(0.01, 0.5, 0.01, value=0.1, id='I0-slider', tooltip={"placement": "bottom"}, updatemode='drag'),
            html.Label("β (інфекційність)"),
            dcc.Slider(0.05, 1.0, 0.05, value=0.3, id='beta-slider', tooltip={"placement": "bottom"}, updatemode='drag'),
            html.Label("γ (одужання)"),
            dcc.Slider(0.05, 1.0, 0.05, value=0.1, id='gamma-slider', tooltip={"placement": "bottom"}, updatemode='drag')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("ν (вакцинація)"),
            dcc.Slider(0.0, 0.5, 0.01, value=0.05, id='nu-slider', tooltip={"placement": "bottom"}, updatemode='drag'),
            html.Label("ρ (реінфекція)"),
            dcc.Slider(0.0, 0.3, 0.01, value=0.02, id='rho-slider', tooltip={"placement": "bottom"}, updatemode='drag'),
            html.Label("σ (Вінерове збурення)"),
            dcc.Slider(0.0, 0.1, 0.005, value=0.01, id='sigma-slider', tooltip={"placement": "bottom"}, updatemode='drag'),
            html.Label("λ (Пуассонове збурення)"),
            dcc.Slider(0.0, 1.0, 0.05, value=0.2, id='lam-slider', tooltip={"placement": "bottom"}, updatemode='drag')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '10px'})
    ], style={'padding': '10px', 'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'}),

    html.Div([
        html.Button("Графік 3D-траєкторії", id='btn-3d', n_clicks=0, style={'fontSize': '20px', 'margin': '10px', 'padding': '10px 20px'}),
        html.Button("Пік інфікованих (β, γ)", id='btn-infected', n_clicks=0, style={'fontSize': '20px', 'margin': '10px', 'padding': '10px 20px'}),
        html.Button("Поверхня ризику (ν, ρ)", id='btn-risk', n_clicks=0, style={'fontSize': '20px', 'margin': '10px', 'padding': '10px 20px'})
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    dcc.Graph(id='display-graph', style={'height': '70vh'})
])

@app.callback(
    Output('display-graph', 'figure'),
    Input('btn-3d', 'n_clicks'),
    Input('btn-infected', 'n_clicks'),
    Input('btn-risk', 'n_clicks'),
    Input('S0-slider', 'value'),
    Input('I0-slider', 'value'),
    Input('beta-slider', 'value'),
    Input('gamma-slider', 'value'),
    Input('nu-slider', 'value'),
    Input('rho-slider', 'value'),
    Input('sigma-slider', 'value'),
    Input('lam-slider', 'value')
)
def update_figure(n3d, ninfected, nrisk, S0, I0, beta, gamma, nu, rho, sigma, lam):
    R0 = 1.0 - S0 - I0
    t = np.linspace(0, 100, 200)
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'btn-3d'

    if button_id == 'btn-3d':
        S, I, R = stochastic_sir(S0, I0, R0, beta, gamma, nu, rho, sigma, lam, t)
        trace = go.Scatter3d(x=S, y=I, z=R, mode='lines', line=dict(color='firebrick'))
        fig = go.Figure(data=[trace])
        fig.update_layout(scene=dict(xaxis_title='S', yaxis_title='I', zaxis_title='R'), title="3D Траєкторія SIR")
        return fig

    elif button_id == 'btn-infected':
        beta_vals = np.linspace(0.1, 0.8, 20)
        gamma_vals = np.linspace(0.05, 0.5, 20)
        Z = np.zeros((len(beta_vals), len(gamma_vals)))
        for i, b in enumerate(beta_vals):
            for j, g in enumerate(gamma_vals):
                _, I_sim, _ = stochastic_sir(S0, I0, R0, b, g, nu, rho, sigma, lam, np.linspace(0, 100, 100))
                Z[i, j] = max(I_sim)
        surf_fig = go.Figure(data=[go.Surface(z=Z, x=beta_vals, y=gamma_vals, colorscale='Viridis')])
        surf_fig.update_layout(title="Пік інфікованих залежно від β і γ", scene=dict(xaxis_title='β', yaxis_title='γ', zaxis_title='Imax'))
        return surf_fig

    elif button_id == 'btn-risk':
        nu_vals = np.linspace(0.0, 0.5, 20)
        rho_vals = np.linspace(0.0, 0.3, 20)
        Z2 = np.zeros((len(nu_vals), len(rho_vals)))
        for i, nu_ in enumerate(nu_vals):
            for j, rho_ in enumerate(rho_vals):
                _, I_sim, _ = stochastic_sir(S0, I0, R0, beta, gamma, nu_, rho_, sigma, lam, np.linspace(0, 100, 100))
                Z2[i, j] = max(I_sim)
        risk_fig = go.Figure(data=[go.Surface(z=Z2, x=nu_vals, y=rho_vals, colorscale='Reds')])
        risk_fig.update_layout(title="Ризик інфекції залежно від вакцинації та реінфекції", scene=dict(xaxis_title='ν', yaxis_title='ρ', zaxis_title='Imax'))
        return risk_fig

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
