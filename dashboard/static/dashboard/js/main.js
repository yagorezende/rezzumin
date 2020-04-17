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
}

function timeout(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function getStatus(id) {
    i = 0;
    var oldColor = "text-warning";

    while(i < 100){
        var phrase = PHRASES[Math.floor(Math.random() * PHRASES.length)];
        var newColor = COLORS[Math.floor(Math.random() * COLORS.length)];

        $("#progress-bar-label").html(phrase);
        $("#pdf-spinner").removeClass(oldColor);
        $("#pdf-spinner").addClass(newColor);
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