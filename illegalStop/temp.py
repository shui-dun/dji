import plotly.graph_objects as go

if __name__ == '__main__':
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[1, 2], y=[3, 4]
    ))
    fig.write_html("a.html")
