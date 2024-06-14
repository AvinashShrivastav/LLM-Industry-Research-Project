from plot_from_json_static import plot_from_ai_output
output = '''<data> [ { "type": "line", "x": ["Oct-21", "Nov-22", "Dec-21", "Jan-22", "Feb-22", "Mar-22", "Apr-22", "May-22", "June-22", "Jul-22"], "y": [0.85, 1.87, 4.05, 5.43, 5.85, 7.68, 8.31, 7.97, 7.75, 6.69], "mode": "lines", "name": "Inflation Rate", "line": { "color": "rgba(75, 192, 192, 1)", "width": 2 } } ] </data> <layout> { "title": { "text": "All India year on year inflation Rate", "font": { "family": "Arial, sans-serif", "size": 24, "color": "#000000" } }, "xaxis": { "title": { "text": "Month", "font": { "family": "Arial, sans-serif", "size": 18, "color": "#000000" } }, "showgrid": true, "gridcolor": "rgba(0, 0, 0, 0.1)", "zeroline": true, "zerolinecolor": "rgba(0, 0, 0, 0.1)", "tickangle": 45 }, "yaxis": { "title": { "text": "Inflation Rate", "font": { "family": "Arial, sans-serif", "size": 18, "color": "#000000" } }, "showgrid": true, "gridcolor": "rgba(0, 0, 0, 0.1)", "zeroline": true, "zerolinecolor": "rgba(0, 0, 0, 0.1)" }, "legend": { "orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.2, "font": { "family": "Arial, sans-serif", "size": 12, "color": "#000000" } }, "margin": { "l": 60, "r": 30, "b": 60, "t": 60 }, "plot_bgcolor": "#ffffff", "paper_bgcolor": "#ffffff" } </layout> <config> { "responsive": true, "displayModeBar": true, "modeBarButtonsToRemove": ["toImage"], "scrollZoom": true } </config>'''
plot_from_ai_output(output, 'image.png')