import plotly.graph_objs as go
import plotly.io as pio
import json
import re

def extract_json_section(input_string, tag):
    """Extract JSON section between specified XML-like tags."""
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, input_string, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def parse_json(json_str):
    """Attempt to parse JSON with relaxed rules."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try replacing single quotes with double quotes and parsing again
        try:
            fixed_json_str = json_str.replace("'", '"')
            return json.loads(fixed_json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

def plot_from_ai_output(input_string, output_path):
    # Extract JSON sections
    data_json = extract_json_section(input_string, "data")
    layout_json = extract_json_section(input_string, "layout")
    config_json = extract_json_section(input_string, "config")

    # Parse JSON strings with relaxed rules
    data = parse_json(data_json) if data_json else None
    layout = parse_json(layout_json) if layout_json else None
    config = parse_json(config_json) if config_json else None

    if not data or not layout:
        return input_string
        raise ValueError("Invalid or missing data or layout JSON.")

    # Prepare traces for the plot
    traces = []
    for trace_data in data:
        trace_type = trace_data.get('type')

        if trace_type == 'bar':
            trace = go.Bar(
                x=trace_data.get('x', []),
                y=trace_data.get('y', []),
                name=trace_data.get('name', ''),
                marker=dict(color=trace_data['marker']['color']) if 'marker' in trace_data else None
            )
        elif trace_type == 'line':
            trace = go.Scatter(
                x=trace_data.get('x', []),
                y=trace_data.get('y', []),
                mode='lines',
                name=trace_data.get('name', ''),
                line=dict(color=trace_data['line']['color'], width=trace_data['line']['width']) if 'line' in trace_data else None,
                hoverinfo='text',
                text=trace_data['text'] if 'text' in trace_data else None
            )
        elif trace_type == 'scatter':
            trace = go.Scatter(
                x=trace_data.get('x', []),
                y=trace_data.get('y', []),
                mode=trace_data.get('mode', 'markers'),
                name=trace_data.get('name', ''),
                marker=dict(color=trace_data['marker']['color']) if 'marker' in trace_data else None,
                line=dict(color=trace_data['line']['color'], width=trace_data['line']['width']) if 'line' in trace_data else None,
                hoverinfo='text',
                text=trace_data['text'] if 'text' in trace_data else None
            )
        elif trace_type == 'pie':
            trace = go.Pie(
                labels=trace_data.get('labels', []),
                values=trace_data.get('values', []),
                name=trace_data.get('name', ''),
                textinfo=trace_data.get('textinfo', 'percent+label'),
                hoverinfo=trace_data.get('hoverinfo', 'label+percent+name')
            )
        else:
            raise ValueError(f"Unsupported trace type: {trace_type}")

        traces.append(trace)

    # Create figure with the extracted layout and data
    fig = go.Figure(data=traces, layout=layout)
    # Save the figure as an image (optional)
    fig.write_image(output_path)