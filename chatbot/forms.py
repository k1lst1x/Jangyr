from django import forms

class DocumentUploadForm(forms.Form):
    file = forms.FileField(label="Документ или ZIP‑архив", required=False)
