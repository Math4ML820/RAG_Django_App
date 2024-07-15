document.addEventListener('DOMContentLoaded', function () {
    var modal = document.getElementById("uploadModal");
    var btn = document.getElementById("uploadBtn");
    var span = document.getElementsByClassName("close")[0];

    btn.onclick = function () {
        modal.style.display = "block";
    }

    span.onclick = function () {
        modal.style.display = "none";
    }

    window.onclick = function (event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }
});


document.addEventListener("DOMContentLoaded", function () {
    const fileRadioButtons = document.querySelectorAll('input[type="radio"][name="selected_file"]');
    const submitButton = document.querySelector('.submitQuestionButton');  // Assuming there's a submit button with id 'submitQuestionButton'
    const loadingIndicator = document.querySelector('#loadingIndicator');
    const overLay = document.getElementById("overlay");
    const questionTextArea = document.getElementById("question");

    submitButton.addEventListener('click', function (event) {
        let fileSelected = false;
        let selectedFileName = '';
        let questionText = questionTextArea.value.trim();

        fileRadioButtons.forEach(function (radioButton) {
            if (radioButton.checked) {
                fileSelected = true;
                selectedFileName = radioButton.value;  // Assuming value is the file name
                // console.log(selectedFileName);
            }
        });

        if (!fileSelected) {
            alert('Please select a file before submitting your question.');
            event.preventDefault();  // Prevent form submission if no file is selected
        } else if (questionText === '') {
            alert('Please enter a question before submitting.');
            event.preventDefault();  // Prevent form submission if the question is empty
        }
        else {
            // Add the selected file name to the hidden input field
            document.querySelector('#question_on_selected_file').value = selectedFileName;
            console.log(document.querySelector('#question_on_selected_file').value);
            loadingIndicator.style.display = 'block';
            overLay.style.display = "block";
            // localStorage.setItem('selectedFile', selectedRadioButton.value);
        }
    });

    // Optionally, you can hide the loading indicator when the page finishes loading
    window.addEventListener('load', function () {
        loadingIndicator.style.display = 'none';
        overLay.style.display = "none";
    });
});
