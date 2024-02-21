from django import forms
from django.forms import Select
from django.utils.safestring import mark_safe
#The ExtendedInfoWidget class defined which extends the Django 'Select' form widget
class ExtendedInfoWidget(Select):
    #Constructor of this class, takes info_text and other arguments
    def __init__(self, info_text, *args, **kwargs):
        #initializes info_text as popover if passed through form element
        self.info_text = info_text
        #calls init method for initialization
        super().__init__(*args, **kwargs)
    #Overrides default 'Selectt' initialization, 
    def render(self, name, value, attrs=None, renderer=None):
        #Renders widget using parent render, stores rendered html in rendered_widget
        rendered_widget = super().render(name, value, attrs, renderer)
        #Populates the rendered widget element with a button that displays the info_text, and does it with
        #mark_safe to ensure django that string is safe for HTML rendering
        return mark_safe(rendered_widget + f'<button type="button" class="info-button" data-toggle="popover" data-content="{self.info_text}"><img src="../static/heartProjectApp/home-images/info.svg"></button>')
#Creating the HeartDiseaseForm class
class HeartDiseaseForm(forms.Form):
   #Creating the forms elements and specifying what kind of values are allowed
    BMI = forms.ChoiceField(
        choices=[(i, str(i)) for i in range(1, 101)],
        label='BMI',
    
    )
    #ExtendedInfoWidget is added to these two elements because they need extra information, info_text is displayed as a popover text
    PhysicalHealth = forms.ChoiceField(
        choices=[(i, str(i)) for i in range(1, 32)],
        label='Physical Health',
       widget=ExtendedInfoWidget(info_text='Number of good physical health days in a month'),
    
    )

    MentalHealth = forms.ChoiceField(
        choices=[(i, str(i)) for i in range(1, 32)],
        label='Mental Health',
        widget=ExtendedInfoWidget(info_text='Number of good mental health days in a month'),
    
    )

    SleepTime = forms.ChoiceField(
        choices=[(i, str(i)) for i in range(0, 25)], 
        label='Sleep Time',
        widget=forms.Select(attrs={'class': 'form-control'}),
    )

    Smoking = forms.ChoiceField(
        choices=[('No', 'No'), ('Yes', 'Yes')],
        label='Smoking',
        widget=forms.Select(attrs={'class': 'form-control'}),
    )

    AlcoholDrinking = forms.ChoiceField(
        choices=[('No', 'No'), ('Yes', 'Yes')],
        label='Alcohol Drinking',
        widget=forms.Select(attrs={'class': 'form-control'}),
    )

    Stroke = forms.ChoiceField(
        choices=[('No', 'No'), ('Yes', 'Yes')],
        label='Stroke',
        widget=forms.Select(attrs={'class': 'form-control'}),
    )

    DiffWalking = forms.ChoiceField(
        choices=[('No', 'No'), ('Yes', 'Yes')],
        label='Difficulty Walking',
        widget=forms.Select(attrs={'class': 'form-control'}),
    )

    Sex = forms.ChoiceField(
        choices=[('Female', 'Female'), ('Male', 'Male')],
        label='Sex',
        widget=forms.Select(attrs={'class': 'form-control'}),
    )

    AgeCategory = forms.ChoiceField(
        choices=[
            ('18-24', '18-24'),
            ('25-29', '25-29'),
            ('30-34', '30-34'),
            ('35-39', '35-39'),
            ('40-44', '40-44'),
            ('45-49', '45-49'),
            ('50-54', '50-54'),
            ('55-59', '55-59'),
            ('60-64', '60-64'),
            ('65-69', '65-69'),
            ('70-74', '70-74'),
            ('75-79', '75-79'),
            ('80 or older', '80 or older'),
        ],
        label='Age Category',
        widget=forms.Select(attrs={'class': 'form-control'}),
    )

    Race = forms.ChoiceField(
        choices=[
            ('American Indian/Alaskan Native', 'American Indian/Alaskan Native'),
            ('Asian', 'Asian'),
            ('Black', 'Black'),
            ('Hispanic', 'Hispanic'),
            ('Multiracial', 'Multiracial'),
            ('Other', 'Other'),
            ('White', 'White'),
        ],
        label='Race',
        widget=forms.Select(attrs={'class': 'form-control'}),
    )

    Diabetic = forms.ChoiceField(
        choices=[('No', 'No'), ('Yes', 'Yes')],
        label='Diabetic',
        widget=forms.Select(attrs={'class': 'form-control'}),
    )

    PhysicalActivity = forms.ChoiceField(
        choices=[('No', 'No'), ('Yes', 'Yes')],
        label='Physical Activity',
        widget=forms.Select(attrs={'class': 'form-control'}),
    )

    GenHealth = forms.ChoiceField(
        choices=[
            ('Excellent', 'Excellent'),
            ('Fair', 'Fair'),
            ('Good', 'Good'),
            ('Poor', 'Poor'),
            ('Very good', 'Very good'),
        ],
        label='General Health',
        widget=forms.Select(attrs={'class': 'form-control'}),
    )

    Asthma = forms.ChoiceField(
        choices=[('No', 'No'), ('Yes', 'Yes')],
        label='Asthma',
        widget=forms.Select(attrs={'class': 'form-control'}),
    )

    KidneyDisease = forms.ChoiceField(
        choices=[('No', 'No'), ('Yes', 'Yes')],
        label='Kidney Disease',
        widget=forms.Select(attrs={'class': 'form-control'}),
    )

    SkinCancer = forms.ChoiceField(
        choices=[('No', 'No'), ('Yes', 'Yes')],
        label='Skin Cancer',
        widget=forms.Select(attrs={'class': 'form-control'}),
    )

    def __init__(self, *args, csv_columns=None, **kwargs):
        super(HeartDiseaseForm, self).__init__(*args, **kwargs)
        # Add 'form-control' class to the first three fields
        for field_name in ['BMI', 'PhysicalHealth', 'MentalHealth']:
            self.fields[field_name].widget.attrs['class'] = 'form-control'

        # Set default values for specific fields that would thought to allow for faster completion
        self.fields['BMI'].initial = 25
        self.fields['PhysicalHealth'].initial = 15
        self.fields['MentalHealth'].initial = 15
        self.fields['SleepTime'].initial = 8 
        self.fields['Race'].initial = 'Multiracial'
