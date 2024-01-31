from django import forms
from .models import Patient

class PatientForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ['heart_disease', 'bmi', 'smoking', 'alcohol_drinking', 'stroke',
                  'physical_health', 'mental_health', 'diff_walking', 'sex', 'age_category',
                  'race', 'diabetic', 'physical_activity', 'gen_health', 'sleep_time',
                  'asthma', 'kidney_disease', 'skin_cancer']
