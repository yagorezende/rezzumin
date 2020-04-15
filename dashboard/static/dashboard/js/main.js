$(document).ready(function () {
    $('#pdf-spinner').hide();
    setSliderListener();
    $('#upload-pdf-btn').click(function () {
        $('#pdf-file').trigger('click');
        console.log("ready!");
    });

    $('#pdf-upload-submit').click(function () {
        $('#pdf-spinner').show("fast");
        $('#upload-btn').hide("slow");
        console.log("Done");

        var formData = new FormData();
        var file = $('#pdf-file')[0].files[0];
        formData.append('file', file);
        formData.append('percent', $('#file-coverage-slider').val());
        formData.append('csrfmiddlewaretoken', document.getElementsByName('csrfmiddlewaretoken')[0].value);
        console.log("Here!");

        jQuery.ajax({
            url: '/result',
            type: 'post',
            enctype: 'multipart/form-data',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response){
                if(response !== 0){
                   alert('file uploaded');
                }
                else{
                    alert('file not uploaded');
                }
            },
        });
    });
});

function setSliderListener() {
    // Display the default slider value
    $("#file-coverage-output").html($("#file-coverage-slider").val() + "%");

    // Update the current slider value (each time you drag the slider handle)
    $(document).on('change', '#file-coverage-slider', function() {
        $('#file-coverage-output').html( $(this).val() + "%");
    });
}