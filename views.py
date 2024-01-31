from django.shortcuts import render
import plotly.express as px
import json
from .models import Picture
from .forms import PatientForm

def heart_disease_prediction(request):
    if request.method == 'POST':
        form = PatientForm(request.POST)
        if form.is_valid():
            # Process form data
            form.save()
            return render(request, 'prediction_result.html', {'form': form})
    else:
        form = PatientForm()
    return render(request, 'heart_disease_prediction.html', {'form': form})


def choropleth_map(request):
    with open('heartProjectApp/files/merged.geojson') as geojson_file:
        geojson_data = json.load(geojson_file)

    # Convert 'HeartDiseasePercentage' values to floats
    for feature in geojson_data['features']:
        percentage_str = feature['properties'].get('HeartDiseasePercentage', None)
        try:
            percentage_float = float(percentage_str)
            feature['properties']['HeartDiseasePercentage'] = percentage_float
        except (TypeError, ValueError):
            feature['properties']['HeartDiseasePercentage'] = None

    locations = [feature['properties']['NAME'] for feature in geojson_data['features']]
    hover_names = [feature['properties']['NAME'] for feature in geojson_data['features']]
    color_values = [feature['properties']['HeartDiseasePercentage'] for feature in geojson_data['features']]

    fig = px.choropleth(
        geojson_data,
        geojson=geojson_data,
        locations=locations,
        featureidkey="properties.NAME",
        color=color_values,
        hover_name=hover_names,
        color_continuous_scale="reds",
    )

    fig.update_geos(fitbounds="locations", visible=False)
    # Set the initial zoom level
    fig.update_layout(
        autosize=False,
        margin = dict(
                l=0,
                r=0,
                b=0,
                t=0,
                pad=4,
                autoexpand=True
            ),
            width=800,
    )

    graph = fig.to_html(full_html=False)

    return render(request, 'choropleth_map.html', {'graph': graph})

def home(request):
    # Fetch the choropleth map HTML
    choropleth_map_html = choropleth_map(request).content.decode("utf-8")

    # Your existing code...
    images = Picture.objects.all()

    return render(request, 'home.html', {'images': images, 'choropleth_map': choropleth_map_html})
def display_images(request, category):
    images = Picture.objects.filter(category=category)
    return render(request, 'image_list.html', {'images': images})
