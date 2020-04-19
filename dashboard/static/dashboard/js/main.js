let PHRASES = [
    "Fabricando Infra-estrutura Imaginária",
    "Reverificando Almas",
    "Classificando Efeitos de Feitiços",
    "Verificando Almas",
    "Inserindo Gerador de Caos",
    "Escavando Terreno Local",
    "Garantindo a Sinergia Transplanar",
    "Recalculando Matriz Mamífera",
    "Compilando Almas Verificadas",
    "Compondo Eufonia Melódica"
]

let COLORS = [
    "text-primary",
    "text-secondary",
    "text-success",
    "text-primary",
    "text-danger",
    "text-warning",
    "text-info",
    "text-light",
]

$(document).ready(function () {
    $('#pdf-spinner').hide();
    $('#progress-bar-wrapper').hide();
    setSliderListener();

    $('#pdf-file').on("change", function(){
        var file = $('#pdf-file')[0].files[0];
        if(file.type !== "application/pdf"){
            $('#upload-file-footer').html("arquivo: "+file.name+" não é um pdf válido!");
            $('#pdf-upload-submit').addClass("btn-outline-danger");
            $('#pdf-upload-submit').addClass("disabled");
            $('#pdf-upload-submit').click(null);

        }else{
            $('#pdf-upload-submit').removeClass("btn-outline-danger");
            $('#pdf-upload-submit').removeClass("disabled");
            $('#upload-file-footer').html("arquivo: "+file.name);
            addUploadPdfAction();
        }
        console.log(file);
    });

    $('#text-upload-submit').click(function () {
        $('#text-spinner').show("fast");
        $('#progress-text-bar-wrapper').show("fast");
        $('#text-field').hide("slow");
        $('#upload-text-slider').hide("slow");
        $('#text-upload-submit').hide("slow");

        var rid = Math.random().toString(36).replace(/[^a-z]+/g, '').substr(0, 10);

        console.log(rid.toUpperCase());
        console.log("ready to submit text!");

        var formData = new FormData();
        formData.append('percent', $('#text-coverage-slider').val());
        formData.append('text', $('#text-field').val());
        formData.append('rid', rid.toUpperCase());
        formData.append('csrfmiddlewaretoken', document.getElementsByName('csrfmiddlewaretoken')[0].value);

        jQuery.ajax({
            url: '/process_text',
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
                    alert('Problemas ao fazer o resumo, por favor tente outra vez');
                }
            },
        });
        getStatus(rid.toUpperCase(), false);
    });

    $('#upload-pdf-btn').click(function () {
        $('#pdf-file').trigger('click');
        console.log("ready!");

    });
});

function copyToClipboard() {
    var $temp = $("<input>");
    $("body").append($temp);
    $temp.val($("#abstract-content").text()).select();
    document.execCommand("copy");
    $temp.remove();

    alert("Resumo copiado!");
}

function addUploadPdfAction() {
    $('#pdf-upload-submit').click(function () {
        $('#pdf-spinner').show("fast");

        $('#progress-bar-wrapper').show("fast");
        $('#upload-btn').hide("slow");
        $('#pdf-upload-submit').hide("slow");

        var rid = Math.random().toString(36).replace(/[^a-z]+/g, '').substr(0, 10);

        console.log(rid.toUpperCase());

        var formData = new FormData();
        var file = $('#pdf-file')[0].files[0];
        formData.append('file', file);
        formData.append('rid', rid.toUpperCase());
        formData.append('percent', $('#file-coverage-slider').val());
        formData.append('sbc', $('#sbc-checkbox:checkbox:checked').length > 0);
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
        getStatus(rid.toUpperCase(), true);
    });
}

function timeout(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

var spinner = null;
var progressbar = null;
var progressbar_label = null;


async function getStatus(id, isPdf) {
    i = 0;
    var oldColor = "text-warning";

    if(isPdf == true){
        spinner = $("#pdf-spinner");
        progressbar = $("#progress-bar");
        progressbar_label = $("#progress-bar-label");
    }else{
        spinner = $("#text-spinner");
        progressbar = $("#progress-text-bar");
        progressbar_label = $("#progress-text-bar-label");
    }

    while(i < 100){
        var phrase = PHRASES[Math.floor(Math.random() * PHRASES.length)];
        var newColor = COLORS[Math.floor(Math.random() * COLORS.length)];

        progressbar_label.html(phrase);
        spinner.removeClass(oldColor);
        spinner.addClass(newColor);
        oldColor = newColor;

        await timeout(10000);

        jQuery.ajax({
            url: '/get_status?id='+id,
            type: 'get',
            contentType: false,
            processData: false,
            success: function(response){
                if(response !== 0){
                    console.log(response.status+"%");
                    progressbar.css("width", response.status+"%");
                    progressbar.html(response.status+"%");

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
    $("#text-coverage-output").html($("#text-coverage-slider").val() + "%");

    // Update the current slider value (each time you drag the slider handle)
    $(document).on('change', '#file-coverage-slider', function() {
        $('#file-coverage-output').html( $(this).val() + "%");
    });

    // Update the current slider value (each time you drag the slider handle)
    $(document).on('change', '#text-coverage-slider', function() {
        $('#text-coverage-output').html( $(this).val() + "%");
    });
}

//Get the button:
mybutton = document.getElementById("go-top-btn");

// When the user scrolls down 20px from the top of the document, show the button
window.onscroll = function() {scrollFunction()};

function scrollFunction() {
  if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
    mybutton.style.display = "block";
  } else {
    mybutton.style.display = "none";
  }
}

// When the user clicks on the button, scroll to the top of the document
function topFunction() {
  document.body.scrollTop = 0; // For Safari
  document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
}