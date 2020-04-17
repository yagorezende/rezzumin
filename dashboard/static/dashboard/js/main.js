$(document).ready(function () {
    $('#pdf-spinner').hide();
    $('#progress-bar-wrapper').hide();
    setSliderListener();
    $('#upload-pdf-btn').click(function () {
        $('#pdf-file').trigger('click');
        console.log("ready!");
    });

    $('#pdf-upload-submit').click(function () {
        $('#pdf-spinner').show("fast");
        $('#progress-bar-wrapper').show("fast");
        $('#upload-btn').hide("slow");


        var rid = Math.random().toString(36).replace(/[^a-z]+/g, '').substr(0, 10);

        console.log(rid.toUpperCase());

        var formData = new FormData();
        var file = $('#pdf-file')[0].files[0];
        formData.append('file', file);
        formData.append('rid', rid.toUpperCase());
        formData.append('percent', $('#file-coverage-slider').val());
        formData.append('csrfmiddlewaretoken', document.getElementsByName('csrfmiddlewaretoken')[0].value);
        console.log("Here!");

        jQuery.ajax({
            url: '/process',
            type: 'post',
            enctype: 'multipart/form-data',
            data: formData,
            contentType: false,
            processData: false,
            success: function(response){
                if(response !== 0){
                    // answer = jQuery.parseJSON(response);
                    if(response.result != null){
                        window.location.replace("/result?id="+response.result);
                    }
                }
                else{
                    alert('file not uploaded');
                }
            },
        });
        getStatus(rid.toUpperCase());
    });
});

function timeout(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function getStatus(id) {
    i = 0;
    while(i < 100){
        await timeout(10000);

        jQuery.ajax({
            url: '/get_status?id='+id,
            type: 'get',
            contentType: false,
            processData: false,
            success: function(response){
                if(response !== 0){
                    console.log(response.status+"%");
                    $('#progress-bar').css("width", response.status+"%");
                    $('#progress-bar').html(response.status+"%");

                    i = response.status;
                    // if(response.answer !== 200) i = 100;
                }
                else{
                    alert('error reading update, please reload page');
                }
            },
        });
    }

}

function setSliderListener() {
    // Display the default slider value
    $("#file-coverage-output").html($("#file-coverage-slider").val() + "%");

    // Update the current slider value (each time you drag the slider handle)
    $(document).on('change', '#file-coverage-slider', function() {
        $('#file-coverage-output').html( $(this).val() + "%");
    });
}