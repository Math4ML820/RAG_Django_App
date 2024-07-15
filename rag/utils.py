import os
from django.conf import settings

def get_uploaded_files():
    upload_dir = os.path.join(settings.BASE_DIR, 'rag', 'uploads')
    uploaded_files = []
    for root, dirs, files in os.walk(upload_dir):
        for file in files:
            if file.endswith('.pdf'):
                uploaded_files.append(file)
    return uploaded_files