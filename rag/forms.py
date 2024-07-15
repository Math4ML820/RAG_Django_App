# In forms.py (create this file if it doesn't exist)
from django import forms

class UploadFileForm(forms.Form):
    pdf_file = forms.FileField(label='Upload PDF')