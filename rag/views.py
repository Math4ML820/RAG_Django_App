# views.py
from django.shortcuts import render, redirect
from django.urls import reverse
import os
from django.conf import settings
import sys
from .utils import get_uploaded_files
from .forms import UploadFileForm
from django.contrib.auth.decorators import login_required
from .rag import *  # Import your main function from rag.py
import logging

logger = logging.getLogger(__name__)

# @login_required(login_url='/admin')
def llm(request):
    upload_form = UploadFileForm()
    uploaded_files = []

    # Directory to save uploads
    upload_dir = os.path.join(settings.BASE_DIR, 'rag', 'uploads')

    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    uploaded_files = get_uploaded_files()

    question_history = request.session.get('question_history', [])
    logger.info(f'question_history is {question_history}')
    selected_file = request.session.get('selected_file', '')
    logger.info(f'selected_file is {selected_file}')

    if request.method == 'POST':
        if 'pdf_file' in request.FILES:
            # Handle file upload
            upload_form = UploadFileForm(request.POST, request.FILES)
            if upload_form.is_valid():
                uploaded_file = request.FILES['pdf_file']
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, 'wb+') as destination:
                    for chunk in uploaded_file.chunks():
                        destination.write(chunk)
                if 'question_history' in request.session:
                    request.session['question_history'] = []
                    question_history = request.session['question_history']
                if 'selected_file' in request.session:
                    request.session['selected_file'] = []
                    selected_file = request.session['selected_file']
                uploaded_files = get_uploaded_files()
                request.session.clear()
                # redirect('llm')
        elif 'question' in request.POST:
            # Handle question submission
            question = request.POST['question']
            selected_file = request.POST['question_on_selected_file']
            request.session['selected_file'] = selected_file
            # logger.info(f'selected_file is {request.session['selected_file']}')
            if selected_file and question:
                uploaded_file_path = os.path.join(upload_dir, selected_file)
                # Call your main function here with the uploaded_file_path and question
                answer = main(uploaded_file_path, question)
                # Add the question and answer to the session history
                if 'question_history' not in request.session:
                    logger.info('looks like question_history not there....')
                    request.session['question_history'] = []
                request.session['question_history'].append({'question': question, 'answer': answer})
                question_history = request.session.get('question_history', [])
            else:
                logger.info('did not get the question or selected_file properly')

    return render(request, 'llm.html', {
        'upload_form': upload_form,
        'uploaded_files': uploaded_files,
        'question_history': question_history,
        'selected_file': selected_file
    })